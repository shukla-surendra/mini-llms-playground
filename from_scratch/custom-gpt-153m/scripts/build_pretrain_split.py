#!/usr/bin/env python3
"""One-off: build data/train.txt + data/test.txt for the pretraining corpus from the
five sources already fetched into data/raw/*/text.txt (see each folder's SOURCE.md).
Rebuilds this project's OWN train/test from the full enriched raw pool (cosmopedia-v1/v2
+ finemath-4plus enrichment, Hindi Wikipedia, and open-web-math all now filled to their
2026-08-17 targets) — supersedes the smaller 1.174B-token build this replaces.

Held out per-source (not one global shuffle, to keep test representative of every
source): the LAST ~1% of each source's documents become test, the rest become train.
Documents are separated by "\\n\\n" (DOCUMENT_SEPARATOR) in the source files already;
this only re-partitions at those same boundaries, never splits a document.

Memory: never holds a whole source file in memory. Two earlier versions of this script
did (`text.split(SEP)` across all 5 files at once ~13GB combined, then a plain
`f.read()` per file peaking near 2x the largest single file) and both were OOM-killed on
this machine (only ~8GB free at the time). This version finds the ~99% split point by
reading a small window (a few MB) around a byte-offset estimate, then stream-copies each
half in fixed-size chunks — peak memory is the chunk size, not the file size.

    uv run python scripts/build_pretrain_split.py
"""

import os

SOURCES = [
    "data/raw/HuggingFaceTB__smollm-corpus__cosmopedia-v2/text.txt",
    "data/raw/HuggingFaceTB__cosmopedia/text.txt",
    "data/raw/HuggingFaceTB__finemath__finemath-4plus/text.txt",
    "data/raw/wikimedia__wikipedia__hi/text.txt",
    "data/raw/open-web-math__open-web-math/text.txt",
]
TEST_FRACTION = 0.01
SEP = b"\n\n"
WINDOW = 8 * 1024 * 1024   # 8MB window to search for the boundary near the 99% mark
CHUNK = 64 * 1024 * 1024   # 64MB copy chunks


def find_split_byte_offset(path, size):
    """Byte offset of a real SEP boundary nearest the 99%-of-file mark."""
    approx = int(size * (1.0 - TEST_FRACTION))
    start = max(0, approx - WINDOW // 2)
    with open(path, "rb") as f:
        f.seek(start)
        window = f.read(WINDOW)
    local_approx = approx - start
    idx = window.rfind(SEP, 0, local_approx)
    if idx == -1:
        idx = window.find(SEP, local_approx)
    return start + idx if idx != -1 else size


def stream_copy(path, out_f, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            n = min(CHUNK, remaining)
            data = f.read(n)
            if not data:
                break
            out_f.write(data)
            remaining -= len(data)


def process_source(path, train_f, test_f):
    size = os.path.getsize(path)
    split_at = find_split_byte_offset(path, size)
    stream_copy(path, train_f, 0, split_at)
    stream_copy(path, test_f, split_at + len(SEP), size)
    print(f"  {path}: {size:,} bytes -> train {split_at:,} bytes, "
          f"test {size - split_at - len(SEP):,} bytes")


def main():
    with open("data/train.txt", "wb") as train_f, \
         open("data/test.txt", "wb") as test_f:
        for i, src in enumerate(SOURCES):
            if i > 0:
                train_f.write(SEP)
                test_f.write(SEP)
            process_source(src, train_f, test_f)

    print("\ndone")


if __name__ == "__main__":
    main()
