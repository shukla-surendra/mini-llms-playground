#!/usr/bin/env python3
"""Stream a HuggingFaceTB/smollm-corpus config into a pretraining .txt, up to a token
budget, without downloading the whole (multi-billion-token) config.

This is the "2.5B-token pretraining corpus" DATASET.md describes as not yet existing —
deliberately a *separate* file from data/train.txt (the chat/fine-tuning corpus), never
pooled with it (see DATASET.md's "The corpus you already have" section for why).

Uses r50k_base + encode_ordinary to count tokens, matching exactly what
`corpus-extractor tokenize` / `gpt-tokenize` will produce later — the budget here is
real tokens, not an estimate. Documents are joined with "\\n\\n", matching this
project's actual DOCUMENT_SEPARATOR (see prepare.py and corpus-extractor's
tokenize_cmd.rs) rather than a literal "<|endoftext|>", which encode_ordinary would
just BPE-encode as ordinary text rather than treat as a boundary.

    uv run python scripts/fetch_pretrain_corpus.py --dataset HuggingFaceTB/smollm-corpus \\
        --config cosmopedia-v2 --token-budget 600_000_000 \\
        --out-train data/train.txt --out-test data/test.txt

Repeat --config to pull from several configs of the same dataset repo (e.g. cosmopedia
v1's 8 topic splits have no unified "train" config) — the budget is split evenly across
however many configs are given.

Train/test split happens HERE, streaming, at true document granularity — every
`test_every_n`-th document (in original stream order) goes to test, everything else to
train, decided the instant each document is read and before either output file's own
"\\n\\n" joining is written. This is deliberate: re-splitting an already-joined .txt
file after the fact is NOT safe, because the "\\n\\n" DOCUMENT_SEPARATOR used to join
documents is indistinguishable from a document's own internal paragraph breaks —
naively splitting on every "\\n\\n" (an earlier version of this pipeline's approach)
shatters documents into paragraph-sized pieces and can put two paragraphs of the SAME
original document on both sides of the split, leaking train content into "held-out"
test. Splitting at fetch time, before any joining happens, has no such ambiguity: each
row read from the HF stream is a single atomic unit, routed to exactly one file.
"""

import argparse
import time
from pathlib import Path

import tiktoken
from datasets import load_dataset

DOCUMENT_SEPARATOR = "\n\n"


def _human(n):
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


def fetch_one(dataset, config, token_budget, text_field, tokenizer, train_file, test_file,
              test_every_n, doc_index, started, skip_docs=0):
    ds = load_dataset(dataset, config, split="train", streaming=True)
    total_tokens = 0
    docs_written = 0
    test_docs_written = 0
    docs_skipped = 0
    last_report = time.time()

    for row in ds:
        text = row.get(text_field, "")
        if not text:
            continue

        # HF streaming iteration order is deterministic (same shard order every run) —
        # re-running this script from scratch would re-fetch the exact same leading
        # documents already sitting in train.txt/test.txt, not new ones. skip_docs
        # advances past however many documents a prior run already consumed for this
        # config before writing anything, so a second "enrichment" run appends genuinely
        # new content instead of duplicating what's already there.
        if docs_skipped < skip_docs:
            docs_skipped += 1
            now = time.time()
            if now - last_report >= 10.0:
                print(f"  [{config}] skipping already-fetched docs: {docs_skipped:,}/{skip_docs:,}", flush=True)
                last_report = now
            continue

        n = len(tokenizer.encode_ordinary(text))

        is_test = (doc_index[0] % test_every_n) == 0
        target, seen_key = (test_file, "test") if is_test else (train_file, "train")
        if target.tell() > 0:
            target.write(DOCUMENT_SEPARATOR)
        target.write(text)
        doc_index[0] += 1
        total_tokens += n
        docs_written += 1
        if is_test:
            test_docs_written += 1

        now = time.time()
        if now - last_report >= 10.0:
            rate = total_tokens / max(now - started, 1e-6)
            pct = 100.0 * total_tokens / token_budget
            print(
                f"  [{config}] {_human(total_tokens)} / {_human(token_budget)} tokens "
                f"({pct:.1f}%) | {docs_written:,} docs ({test_docs_written:,} test) | "
                f"{_human(rate)} tok/s",
                flush=True,
            )
            last_report = now

        if total_tokens >= token_budget:
            break

    if skip_docs:
        print(f"  [{config}] skipped {docs_skipped:,} already-fetched docs before writing new ones", flush=True)

    return total_tokens, docs_written, test_docs_written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="HF dataset repo id")
    parser.add_argument("--config", action="append", required=True,
                        help="Config/subset name. Repeatable — budget splits evenly across all given.")
    parser.add_argument("--token-budget", type=int, required=True, help="Total across all --config values")
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-test", required=True)
    parser.add_argument("--test-every-n", type=int, default=100,
                        help="Every Nth document (stream order) goes to test. Default 100 -> ~1%% test.")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing train/test files instead of truncating "
                             "(for pooling multiple --dataset runs into the same two files).")
    parser.add_argument("--skip-docs-per-config", type=int, default=0,
                        help="Advance past this many documents (per --config, before "
                             "counting toward --token-budget) before writing anything — "
                             "use this on a second/enrichment run so it appends genuinely "
                             "new documents instead of re-fetching the same leading ones "
                             "HF's deterministic stream order would otherwise repeat. Set "
                             "to (roughly) the doc count each config already contributed "
                             "last time — see the target folder's SOURCE.md.")
    args = parser.parse_args()

    train_path, test_path = Path(args.out_train), Path(args.out_test)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding("r50k_base")
    per_config_budget = args.token_budget // len(args.config)
    mode = "a" if args.append else "w"

    started = time.time()
    total_tokens = 0
    total_docs = 0
    total_test_docs = 0
    doc_index = [0]  # mutable counter shared/threaded through fetch_one calls
    with train_path.open(mode, encoding="utf-8") as train_file, \
         test_path.open(mode, encoding="utf-8") as test_file:
        for config in args.config:
            n, d, td = fetch_one(args.dataset, config, per_config_budget, args.text_field,
                                  tokenizer, train_file, test_file, args.test_every_n,
                                  doc_index, started, skip_docs=args.skip_docs_per_config)
            total_tokens += n
            total_docs += d
            total_test_docs += td

    elapsed = time.time() - started
    train_mb = train_path.stat().st_size / (1024 * 1024)
    test_mb = test_path.stat().st_size / (1024 * 1024)
    print(
        f"\ndone   {_human(total_tokens)} tokens, {total_docs:,} docs "
        f"({total_test_docs:,} test / {total_docs - total_test_docs:,} train), {elapsed:,.0f}s"
    )
    print(f"  {train_path} ({train_mb:,.0f} MB)")
    print(f"  {test_path} ({test_mb:,.0f} MB)")


if __name__ == "__main__":
    main()
