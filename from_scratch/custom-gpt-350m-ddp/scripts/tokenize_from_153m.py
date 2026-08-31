#!/usr/bin/env python3
"""One-off: tokenize custom-gpt-153m's rebuilt corpus directly into this project's own
data/train.bin + test.bin, using this project's own tokenizer (tokenizer/tokenizer.json,
trained on a sample of that same corpus via `make tokenizer`).

Reads 153m's data/train.txt / test.txt in place and writes only the tokenized output
here — deliberately does NOT copy the ~13GB source text into this project first (disk
was down to ~19GB free when this was written; a text copy plus its own .bin would not
have fit). `build_token_bin` streams the source rather than reading it whole, so this
is safe regardless of file size.

    uv run python scripts/tokenize_from_153m.py
"""

from pathlib import Path

from gpt.config import TOKENIZER_PATH
from gpt.data.dataset import build_token_bin
from gpt.tokenizer import load_tokenizer

SOURCES = [
    (Path("../custom-gpt-153m/data/train.txt"), Path("data/train.bin")),
    (Path("../custom-gpt-153m/data/test.txt"), Path("data/test.bin")),
]


def main():
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    for text_path, bin_path in SOURCES:
        print(f"tokenizing {text_path} -> {bin_path}")

        def progress(count, _text_path=text_path):
            print(f"  [{_text_path.name}] {count:,} tokens", flush=True)

        count = build_token_bin(tokenizer, text_path, bin_path, progress=progress)
        print(f"  done: {count:,} tokens -> {bin_path} "
              f"({bin_path.stat().st_size / (1024*1024):,.0f} MB)")


if __name__ == "__main__":
    main()
