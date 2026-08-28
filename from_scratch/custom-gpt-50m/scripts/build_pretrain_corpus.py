#!/usr/bin/env python3
"""Build a prose pretraining corpus for custom-gpt-50m from _shared_data/raw, using this
project's own tokenizer (GPT-2 via tiktoken, vocab 50,257) — NOT a copy of custom-gpt-350m's
already-tokenized .bin, which is indexed against a completely different 32,768-entry custom
vocabulary. Reusing those token ids here would be silent corruption: every id is still a
valid embedding-table row, just the wrong one (see gpt.data.dataset's tokenizer_fingerprint
docstring on the sibling project for exactly this failure mode).

Scope: books + Cosmopedia only (~1.03B tokens under GPT-2 BPE, close to what these two
sources produced under the 350m project's tokenizer). That alone is already close to
Chinchilla-optimal for a 50M-param model (~20 tokens/param = 1B) — no need to also pull in
FineMath/open-web-math/wikipedia_hi at this size the way the 350m pool does.

Same streaming-per-source-into-one-.bin mechanism as custom-gpt-350m/scripts/
build_pretrain_corpus.py: never materializes a merged .txt, and is resumable — a source
already recorded in data/manifest.json is skipped and train.bin/test.bin are opened in
append mode, so a killed run doesn't redo completed sources.

    uv run python scripts/build_pretrain_corpus.py
"""

import json
import time
from pathlib import Path

import tiktoken

from gpt.data.dataset import bin_meta_path, build_token_bin

SHARED_RAW = Path("../_shared_data/raw")
OUT_DIR = Path("data")
TRAIN_RATIO = 0.99
BYTES_PER_TOKEN = 2  # TOKEN_DTYPE is uint16 (GPT-2's 50,257 vocab fits)
COPY_CHUNK = 4 * 1024 * 1024

SOURCES = [
    ("books", SHARED_RAW / "books/dataset.txt"),
    ("cosmopedia", SHARED_RAW / "HuggingFaceTB__cosmopedia/text.txt"),
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
        "tokenizer": "gpt2 (tiktoken)",
    }, indent=2))


def split_into(per_source_bin, split_byte, train_f, test_f):
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


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.bin"
    test_path = OUT_DIR / "test.bin"
    manifest_path = OUT_DIR / "manifest.json"

    manifest = json.loads(manifest_path.read_text())["sources"] if manifest_path.exists() else {}
    done = set(manifest)
    if done:
        print(f"[resume] already done, skipping: {', '.join(done)}")
    mode = "ab" if done else "wb"

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

            _write_manifest(manifest_path, manifest)

    total_tokens = sum(m["tokens"] for m in manifest.values())
    total_train = sum(m["train_tokens"] for m in manifest.values())
    total_test = sum(m["test_tokens"] for m in manifest.values())

    print("\n=== Pretrain corpus built ===")
    for name, m in manifest.items():
        pct = 100 * m["tokens"] / total_tokens if total_tokens else 0
        print(f"  {name:<16} {m['tokens']:>14,} tokens  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<16} {total_tokens:>14,} tokens")
    print(f"\n  train.bin: {train_path.stat().st_size / (1024**3):.2f} GB "
          f"({total_train:,} tokens)")
    print(f"  test.bin:  {test_path.stat().st_size / (1024**3):.2f} GB "
          f"({total_test:,} tokens)")
    print(f"  wall time: {time.time() - t_start:,.0f}s")


if __name__ == "__main__":
    main()
