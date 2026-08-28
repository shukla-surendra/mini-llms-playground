#!/usr/bin/env python3
"""Build the full prose pretraining pool for custom-gpt-350m from _shared_data/raw.

Scope decision (2026-08-27): prose/text sources only — books, Cosmopedia, Cosmopedia-v2,
FineMath-4+, open-web-math, Hindi Wikipedia. Chat/instruction sources (UltraChat, OASST1,
Dolly, No Robots, SmolTalk, GSM8K, LMSYS-Chat-1M, OpenHermes-2.5) are deliberately excluded
here — DATASET.md already documents why mixing them into a pretraining stream dilutes
gradient signal; they remain available under _shared_data/raw for a post-pretrain
fine-tuning stage instead.

Streams each source straight into data/pretrain/train.bin + test.bin (99/1 split per
source, by token count) without ever materializing a merged .txt — keeps peak extra disk
usage to roughly the tokenized output size (~0.5x the raw text size), since only one
small per-source scratch .bin exists at a time.

This is the FULL available pool, not the training mix — a later step samples ~7B tokens
out of it (see manifest.json's per-source counts for what's available to draw from).

    uv run python scripts/build_pretrain_corpus.py
"""

import json
import time
from pathlib import Path

from gpt.config import TOKENIZER_PATH
from gpt.data.dataset import bin_meta_path, build_token_bin
from gpt.tokenizer import load_tokenizer

SHARED_RAW = Path("../_shared_data/raw")
OUT_DIR = Path("data/pretrain")
TRAIN_RATIO = 0.99
BYTES_PER_TOKEN = 2  # TOKEN_DTYPE is uint16
COPY_CHUNK = 4 * 1024 * 1024

SOURCES = [
    ("books", SHARED_RAW / "books/dataset.txt"),
    ("cosmopedia", SHARED_RAW / "HuggingFaceTB__cosmopedia/text.txt"),
    ("cosmopedia_v2", SHARED_RAW / "HuggingFaceTB__smollm-corpus__cosmopedia-v2/text.txt"),
    ("finemath_4plus", SHARED_RAW / "HuggingFaceTB__finemath__finemath-4plus/text.txt"),
    ("open_web_math", SHARED_RAW / "open-web-math__open-web-math/text.txt"),
    ("wikipedia_hi", SHARED_RAW / "wikimedia__wikipedia__hi/text.txt"),
]


def _write_manifest(manifest_path, manifest):
    total_tokens = sum(m["tokens"] for m in manifest.values())
    total_train = sum(m["train_tokens"] for m in manifest.values())
    total_test = sum(m["test_tokens"] for m in manifest.values())
    manifest_path.write_text(json.dumps({
        "sources": manifest,
        "total_tokens": total_tokens,
        "total_train_tokens": total_train,
        "total_test_tokens": total_test,
        "train_ratio": TRAIN_RATIO,
        "tokenizer": TOKENIZER_PATH,
    }, indent=2))


def split_into(per_source_bin, split_byte, train_f, test_f):
    size = per_source_bin.stat().st_size
    with open(per_source_bin, "rb") as src:
        remaining = split_byte
        while remaining > 0:
            chunk = src.read(min(COPY_CHUNK, remaining))
            if not chunk:
                break
            train_f.write(chunk)
            remaining -= len(chunk)
        while True:
            chunk = src.read(COPY_CHUNK)
            if not chunk:
                break
            test_f.write(chunk)
    assert train_f.tell() >= 0 and test_f.tell() >= 0
    return size


def main():
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.bin"
    test_path = OUT_DIR / "test.bin"
    manifest_path = OUT_DIR / "manifest.json"

    # Resume support: a source already recorded in manifest.json is already merged into
    # train.bin/test.bin (split_into runs before the manifest entry is written below), so
    # skipping it AND opening the .bin files in append mode is what makes a killed run
    # continue instead of re-tokenizing from byte zero — books/cosmopedia together took
    # ~62 minutes; redoing them on every resume would waste that every time.
    manifest = json.loads(manifest_path.read_text())["sources"] if manifest_path.exists() else {}
    done = set(manifest)
    if done:
        print(f"[resume] already done, skipping: {', '.join(done)}")
    mode = "ab" if done else "wb"

    # A stale, never-finalized scratch file from a killed run (build_token_bin's own
    # <name>.tmp, or an incomplete _scratch_<name>.bin left by an interrupted split_into)
    # is not resumable mid-file — build_token_bin has no byte-offset-aware resume — so the
    # source it belongs to just restarts from its own beginning next time it's reached.
    for stale in OUT_DIR.glob("_scratch_*"):
        print(f"[cleanup] removing stale partial: {stale.name}")
        stale.unlink()

    t_start = time.time()

    with open(train_path, mode) as train_f, open(test_path, mode) as test_f:
        for name, text_path in SOURCES:
            if name in done:
                continue
            if not text_path.exists():
                print(f"[skip] {name}: {text_path} not found")
                continue

            per_source_bin = OUT_DIR / f"_scratch_{name}.bin"
            t0 = time.time()
            print(f"[tokenize] {name} <- {text_path} "
                  f"({text_path.stat().st_size / (1024**3):.2f} GB)")

            last_report = [0.0]

            def progress(count, _name=name, _last=last_report):
                now = time.time()
                if now - _last[0] > 5:
                    print(f"  [{_name}] {count:,} tokens", flush=True)
                    _last[0] = now

            count = build_token_bin(tokenizer, text_path, per_source_bin, progress=progress)
            split_at = int(count * TRAIN_RATIO)
            split_byte = split_at * BYTES_PER_TOKEN

            split_into(per_source_bin, split_byte, train_f, test_f)

            per_source_bin.unlink()
            bin_meta_path(per_source_bin).unlink(missing_ok=True)
            train_f.flush()
            test_f.flush()

            elapsed = time.time() - t0
            manifest[name] = {
                "tokens": count,
                "train_tokens": split_at,
                "test_tokens": count - split_at,
                "source": str(text_path),
                "seconds": round(elapsed, 1),
            }
            print(f"  done: {count:,} tokens ({split_at:,} train / "
                  f"{count - split_at:,} test) in {elapsed:,.0f}s")

            # Persisted after EVERY source, not just at the end — this is what a resumed
            # run reads back to decide what's already safely inside train.bin/test.bin.
            _write_manifest(manifest_path, manifest)

    total_tokens = sum(m["tokens"] for m in manifest.values())
    total_train = sum(m["train_tokens"] for m in manifest.values())
    total_test = sum(m["test_tokens"] for m in manifest.values())

    print("\n=== Full pretrain pool built ===")
    for name, m in manifest.items():
        pct = 100 * m["tokens"] / total_tokens if total_tokens else 0
        print(f"  {name:<16} {m['tokens']:>14,} tokens  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<16} {total_tokens:>14,} tokens")
    print(f"\n  train.bin: {train_path.stat().st_size / (1024**3):.2f} GB "
          f"({total_train:,} tokens)")
    print(f"  test.bin:  {test_path.stat().st_size / (1024**3):.2f} GB "
          f"({total_test:,} tokens)")
    print(f"  manifest:  {OUT_DIR / 'manifest.json'}")
    print(f"\n  wall time: {time.time() - t_start:,.0f}s")


if __name__ == "__main__":
    main()
