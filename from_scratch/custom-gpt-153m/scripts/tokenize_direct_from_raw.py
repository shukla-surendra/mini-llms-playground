#!/usr/bin/env python3
"""One-off: build data/train.bin + data/test.bin directly from data/raw/*/text.txt,
skipping the combined data/train.txt/test.txt intermediate entirely.

Why: the combined-text approach (build_pretrain_split.py -> gpt-tokenize) needs the full
~30GB train.txt + its ~15GB train.bin on disk at the same time — after round 2's
enrichment the raw pool grew to ~28GB and disk ran out (down to ~5GB free mid-run).
This script processes ONE source at a time: split it (same 99%-mark boundary method as
build_pretrain_split.py), write only THAT source's train/test slice to a temp .txt,
tokenize it with `build_token_bin` into a temp .bin, delete the temp .txt immediately,
keep the temp .bin. After all 5 sources, concatenate the per-source .bin files into
data/train.bin / data/test.bin — `.bin` files are flat uint16 arrays with no header, so
concatenation is exactly equivalent to tokenizing the combined text (documented
already in the sibling 350m project's DATASET.md "Mechanics" section).

    uv run python scripts/tokenize_direct_from_raw.py
"""

import json
import os
from pathlib import Path

import tiktoken

from gpt.config import TOKENIZER_NAME
from gpt.data.dataset import bin_meta_path, build_token_bin, tokenizer_fingerprint

SOURCES = [
    "data/raw/HuggingFaceTB__smollm-corpus__cosmopedia-v2/text.txt",
    "data/raw/HuggingFaceTB__cosmopedia/text.txt",
    "data/raw/HuggingFaceTB__finemath__finemath-4plus/text.txt",
    "data/raw/wikimedia__wikipedia__hi/text.txt",
    "data/raw/open-web-math__open-web-math/text.txt",
]
TEST_FRACTION = 0.01
SEP = b"\n\n"
WINDOW = 8 * 1024 * 1024
CHUNK = 64 * 1024 * 1024
TMP_DIR = Path("data/_tokenize_tmp")


def find_split_byte_offset(path, size):
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


def stream_copy(path, out_path, start, end):
    with open(path, "rb") as f, open(out_path, "wb") as out_f:
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            n = min(CHUNK, remaining)
            data = f.read(n)
            if not data:
                break
            out_f.write(data)
            remaining -= len(data)


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)

    train_parts, test_parts = [], []
    total_train_tokens, total_test_tokens = 0, 0

    for i, src in enumerate(SOURCES):
        src_path = Path(src)
        size = os.path.getsize(src_path)
        split_at = find_split_byte_offset(src_path, size)

        train_txt = TMP_DIR / f"{i}_train.txt"
        test_txt = TMP_DIR / f"{i}_test.txt"
        train_bin = TMP_DIR / f"{i}_train.bin"
        test_bin = TMP_DIR / f"{i}_test.bin"

        print(f"[{i+1}/{len(SOURCES)}] {src}: {size:,} bytes, split at {split_at:,}")

        stream_copy(src_path, train_txt, 0, split_at)
        n = build_token_bin(tokenizer, train_txt, train_bin)
        total_train_tokens += n
        train_txt.unlink()
        print(f"  train: {n:,} tokens -> {train_bin} ({train_bin.stat().st_size / (1024**3):.2f} GB)")

        stream_copy(src_path, test_txt, split_at + len(SEP), size)
        n = build_token_bin(tokenizer, test_txt, test_bin)
        total_test_tokens += n
        test_txt.unlink()
        print(f"  test:  {n:,} tokens -> {test_bin} ({test_bin.stat().st_size / (1024**3):.2f} GB)")

        train_parts.append(train_bin)
        test_parts.append(test_bin)

    print("\nConcatenating per-source .bin files -> data/train.bin / data/test.bin")
    with open("data/train.bin", "wb") as out:
        for p in train_parts:
            with open(p, "rb") as f:
                while chunk := f.read(CHUNK):
                    out.write(chunk)
    with open("data/test.bin", "wb") as out:
        for p in test_parts:
            with open(p, "rb") as f:
                while chunk := f.read(CHUNK):
                    out.write(chunk)

    for p in train_parts + test_parts:
        p.unlink()
    TMP_DIR.rmdir()

    fp = tokenizer_fingerprint(tokenizer)
    bin_meta_path(Path("data/train.bin")).write_text(json.dumps(
        {"tokenizer": fp, "tokens": total_train_tokens, "source": "data/raw/* (direct, round 2)"},
        indent=2))
    bin_meta_path(Path("data/test.bin")).write_text(json.dumps(
        {"tokenizer": fp, "tokens": total_test_tokens, "source": "data/raw/* (direct, round 2)"},
        indent=2))

    print(f"\ndone: train {total_train_tokens:,} tokens, test {total_test_tokens:,} tokens, "
          f"total {total_train_tokens + total_test_tokens:,}")


if __name__ == "__main__":
    main()
