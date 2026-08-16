"""`gpt-tokenize` — turn `data/*.txt` into flat uint16 `.bin` token files.

Run once after the corpus changes. Training reads the `.bin` as a `numpy.memmap`, so
the corpus never has to fit in RAM or VRAM (see `gpt/data/dataset.py`).

Why this is a separate step rather than something training does implicitly: at this
project's target scale (~2.5B tokens) tokenizing is minutes of CPU work, and a rented
GPU billed by the hour is the worst possible place to repeat it on every launch or
crash-restart.

    gpt-tokenize                 # build train.bin + test.bin if missing or stale
    gpt-tokenize --force         # rebuild unconditionally
    gpt-tokenize --file data/extra.txt
"""

import argparse
from pathlib import Path
import time


from ..config import TOKENIZER_PATH, load_settings
from ..data.dataset import TOKEN_DTYPE, build_token_bin, token_bin_path
from ..tokenizer import load_tokenizer


def _human(n):
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


def _is_stale(text_path, bin_path):
    if not bin_path.exists():
        return True
    return bin_path.stat().st_mtime < text_path.stat().st_mtime


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize corpus text files into flat uint16 .bin token files."
    )
    parser.add_argument("--preset", default=None, help="Model size preset (for data paths)")
    parser.add_argument("--file", action="append", default=None,
                        help="Tokenize this file instead of train/test (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the .bin is newer than its source text")
    args = parser.parse_args()

    _, _, paths, _ = load_settings(args.preset)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    targets = [Path(f) for f in args.file] if args.file else [
        paths.train_data, paths.test_data
    ]

    total_tokens = 0
    for text_path in targets:
        bin_path = token_bin_path(text_path)
        if not text_path.exists():
            print(f"skip   {text_path} (not found)")
            continue
        if not args.force and not _is_stale(text_path, bin_path):
            print(f"ok     {bin_path} is up to date")
            continue

        src_mb = text_path.stat().st_size / (1024 * 1024)
        print(f"build  {text_path} ({src_mb:,.0f} MB) -> {bin_path}")
        started = time.time()

        last_report = [0.0]

        def progress(count, _started=started, _last=last_report):
            now = time.time()
            if now - _last[0] < 5.0:
                return
            _last[0] = now
            rate = count / max(now - _started, 1e-6)
            print(f"       {_human(count)} tokens  ({_human(rate)}/s)", flush=True)

        count = build_token_bin(tokenizer, text_path, bin_path, progress=progress)
        elapsed = time.time() - started
        size_mb = bin_path.stat().st_size / (1024 * 1024)
        total_tokens += count
        print(
            f"done   {_human(count)} tokens in {elapsed:,.0f}s "
            f"-> {size_mb:,.0f} MB ({TOKEN_DTYPE.__name__})"
        )

    if total_tokens:
        print(f"\nTotal tokenized this run: {total_tokens:,}")


if __name__ == "__main__":
    main()
