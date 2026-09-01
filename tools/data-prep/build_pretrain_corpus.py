#!/usr/bin/env python3
"""Shared, config-driven pretrain-corpus builder — the generalized replacement for every
project's own hardcoded copy of "build_pretrain_split.py". Takes source files and
output paths as arguments instead of a Python-literal SOURCES list baked into one
project's copy of the script, so the same file works for any project, any subset of
`_shared_data/raw/`'s sources, any output location.

Held out per-source (not one global shuffle, to keep test representative of every
source): the LAST `--test-fraction` of each source's documents become test, the rest
become train. Documents are already separated by "\\n\\n" (DOCUMENT_SEPARATOR) in every
`_shared_data/raw/<source>/text.txt` file — this script only re-partitions at those
same boundaries, never re-splits inside a document.

Processes and writes **one source at a time**, immediately freeing that source's
in-memory list before moving to the next — a real, measured requirement here, not
theoretical: an earlier version of this script accumulated every source's documents
into one big list before writing anything, and got OOM-killed (SIGKILL, exit 137)
trying to hold ~28 GB of raw text (5 sources: cosmopedia v1+v2, finemath-4plus,
wikipedia-hi, open-web-math) in memory at once on a 24 GB Mac. Streaming per-source
peaks at roughly one source's own size, not the sum of all of them.

A JSONL source (e.g. tools/corpus-extractor's book/repo-derived output) is supported
directly via --jsonl-source, read one {"text": ...} record per line — never naively
`.split("\\n\\n")` on the plain-text sibling of a JSONL source, since each JSONL record
may itself contain internal blank lines (this was a real bug caught in this exact
codebase — see custom-gpt-50m-ddp/docs/DATA_AND_TRAINING_SOP.md's "why this was
rebuilt from scratch").

Usage, one project's real example:

    uv run python ../../tools/data-prep/build_pretrain_corpus.py \\
        --source ../_shared_data/raw/HuggingFaceTB__cosmopedia/text.txt \\
        --source ../_shared_data/raw/HuggingFaceTB__smollm-corpus__cosmopedia-v2/text.txt \\
        --source ../_shared_data/raw/HuggingFaceTB__finemath__finemath-4plus/text.txt \\
        --source ../_shared_data/raw/wikimedia__wikipedia__hi/text.txt \\
        --source ../_shared_data/raw/open-web-math__open-web-math/text.txt \\
        --jsonl-source ../_shared_data/raw/books/dataset.jsonl \\
        --out-train data/profiles/pretrain/train.txt \\
        --out-test data/profiles/pretrain/test.txt
"""

import argparse
import json
from pathlib import Path

SEP = "\n\n"


def _write_one_doc(fh, doc, is_first_write):
    """Write a single document to an already-open file handle, with a leading SEP
    unless this is truly the first thing ever written to this file."""
    if not is_first_write:
        fh.write(SEP)
    fh.write(doc)
    return False


CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB — bounds peak memory to ~this, not the file size


def iter_documents(path):
    """Yield one document at a time from a large SEP-joined text file, reading in
    fixed-size chunks rather than the whole file at once — genuinely required here,
    not a style preference: an earlier version of this function read the full file
    into one string and called .split(SEP) on it, which peaks at roughly TWO full
    copies in memory at once (the original string plus the resulting list of
    documents) — for this project's ~15 GB cosmopedia-v2 source alone, that's enough
    to exceed a 24 GB Mac's RAM and get the process SIGKILLed (exit 137), confirmed
    twice in this exact codebase before this fix. A small leftover buffer carries
    any partial document across a chunk boundary — SEP straddling two chunk reads is
    handled correctly, not just documents that happen to fit inside one chunk.
    """
    leftover = ""
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            leftover += chunk
            pieces = leftover.split(SEP)
            # The last piece might be an incomplete document (the chunk boundary cut
            # it mid-way) — hold it back and prepend it to the next read, rather than
            # yielding a truncated document now.
            leftover = pieces[-1]
            for doc in pieces[:-1]:
                yield doc
    if leftover:
        yield leftover


def stream_plaintext_source(path, test_fraction, train_fh, test_fh, train_is_first, test_is_first):
    """Every Nth document (N = round(1/test_fraction)) goes to test, spread evenly
    through the file rather than only the tail — a side benefit of not knowing the
    total document count in advance (genuine streaming can't), and arguably more
    representative than a single tail slice if the source has any ordering by
    topic/collection."""
    period = max(1, round(1 / test_fraction))
    n_train = n_test = 0
    for i, doc in enumerate(iter_documents(path)):
        if i % period == 0:
            test_is_first = _write_one_doc(test_fh, doc, test_is_first)
            n_test += 1
        else:
            train_is_first = _write_one_doc(train_fh, doc, train_is_first)
            n_train += 1
    print(f"  [{path}] {n_train + n_test:,} docs -> train {n_train:,} / test {n_test:,}", flush=True)
    return train_is_first, test_is_first


def stream_jsonl_source(path, test_fraction, train_fh, test_fh, train_is_first, test_is_first):
    """One real chunk/record per line, streamed — never re-derived by splitting a
    plain-text sibling file on blank lines, which would shatter multi-paragraph
    records (see module docstring). Same every-Nth-record test split as the
    plaintext path, for the same reason (spread through the file, no need to know
    the total count up front)."""
    period = max(1, round(1 / test_fraction))
    n_train = n_test = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)["text"]
            if i % period == 0:
                test_is_first = _write_one_doc(test_fh, doc, test_is_first)
                n_test += 1
            else:
                train_is_first = _write_one_doc(train_fh, doc, train_is_first)
                n_train += 1
    print(f"  [{path}] {n_train + n_test:,} records -> train {n_train:,} / test {n_test:,}", flush=True)
    return train_is_first, test_is_first


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=[],
                    help="Path to a plain-text source (documents separated by blank lines). Repeatable.")
    p.add_argument("--jsonl-source", action="append", default=[],
                    help="Path to a JSONL source (one {'text': ...} record per line, e.g. "
                         "tools/corpus-extractor output). Repeatable.")
    p.add_argument("--test-fraction", type=float, default=0.01,
                    help="Fraction of each source's own documents held out as test (default 0.01, "
                         "matching this codebase family's established convention).")
    p.add_argument("--out-train", required=True)
    p.add_argument("--out-test", required=True)
    args = p.parse_args()

    if not args.source and not args.jsonl_source:
        p.error("pass at least one --source or --jsonl-source")

    out_train = Path(args.out_train)
    out_test = Path(args.out_test)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    train_is_first, test_is_first = True, True
    with open(out_train, "w", encoding="utf-8") as train_fh, \
         open(out_test, "w", encoding="utf-8") as test_fh:
        for src in args.source:
            train_is_first, test_is_first = stream_plaintext_source(
                src, args.test_fraction, train_fh, test_fh, train_is_first, test_is_first)
        for src in args.jsonl_source:
            train_is_first, test_is_first = stream_jsonl_source(
                src, args.test_fraction, train_fh, test_fh, train_is_first, test_is_first)

    print(f"\ndone -> {out_train} ({out_train.stat().st_size / 1e6:.0f} MB), "
          f"{out_test} ({out_test.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
