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
        --config cosmopedia-v2 --token-budget 600_000_000 --out data/pretrain_cosmopedia_v2.txt

Repeat --config to pull from several configs of the same dataset repo (e.g. cosmopedia
v1's 8 topic splits have no unified "train" config) — the budget is split evenly across
however many configs are given.
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


def fetch_one(dataset, config, token_budget, text_field, tokenizer, out_file, started):
    ds = load_dataset(dataset, config, split="train", streaming=True)
    total_tokens = 0
    docs_written = 0
    last_report = time.time()

    for row in ds:
        text = row.get(text_field, "")
        if not text:
            continue
        n = len(tokenizer.encode_ordinary(text))
        out_file.write(DOCUMENT_SEPARATOR)
        out_file.write(text)
        total_tokens += n
        docs_written += 1

        now = time.time()
        if now - last_report >= 10.0:
            rate = total_tokens / max(now - started, 1e-6)
            pct = 100.0 * total_tokens / token_budget
            print(
                f"  [{config}] {_human(total_tokens)} / {_human(token_budget)} tokens "
                f"({pct:.1f}%) | {docs_written:,} docs | {_human(rate)} tok/s",
                flush=True,
            )
            last_report = now

        if total_tokens >= token_budget:
            break

    return total_tokens, docs_written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="HF dataset repo id")
    parser.add_argument("--config", action="append", required=True,
                        help="Config/subset name. Repeatable — budget splits evenly across all given.")
    parser.add_argument("--token-budget", type=int, required=True, help="Total across all --config values")
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding("r50k_base")
    per_config_budget = args.token_budget // len(args.config)

    started = time.time()
    total_tokens = 0
    total_docs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for config in args.config:
            n, d = fetch_one(args.dataset, config, per_config_budget, args.text_field, tokenizer, f, started)
            total_tokens += n
            total_docs += d

    elapsed = time.time() - started
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"\ndone   {_human(total_tokens)} tokens, {total_docs:,} docs, "
        f"{elapsed:,.0f}s -> {out_path} ({size_mb:,.0f} MB)"
    )


if __name__ == "__main__":
    main()
