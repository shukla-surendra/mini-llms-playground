"""
Download TinyStories, train a small custom BPE tokenizer on it, and write out
tokenized train/val splits as compact binary files.

Why TinyStories and why a custom small vocabulary instead of GPT-2's 50,257-token one:
see docs/DATASET_AND_TOKENIZER.md for the full reasoning.

Usage:
    python prepare_dataset.py --max-samples 100000 --vocab-size 4096
"""
import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

EOT_TOKEN = "<|endoftext|>"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare TinyStories dataset + tokenizer")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--max-samples", type=int, default=100_000,
                    help="Number of TRAINING stories to use (out of ~2.1M available)")
    p.add_argument("--max-val-samples", type=int, default=2_000,
                    help="Number of VALIDATION stories to use (out of ~22k available)")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--reuse-tokenizer", default=None,
                    help="Path to an existing tokenizer.json to tokenize this data with, "
                         "instead of training a new one. REQUIRED if you plan to continue "
                         "training an existing checkpoint on this new data — see "
                         "docs/CONTINUING_TRAINING_ON_NEW_DATA.md. Using a different "
                         "tokenizer than the checkpoint was trained with silently corrupts "
                         "generation, since token IDs would no longer mean the same things "
                         "to the model's embedding table.")
    return p.parse_args()


def train_tokenizer(texts, vocab_size, out_path):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", EOT_TOKEN],
        min_frequency=2,
    )
    print(f"[tokenizer] training BPE, target vocab_size={vocab_size} on {len(texts)} stories...")
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(out_path))
    print(f"[tokenizer] saved to {out_path} (actual vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def tokenize_split(tokenizer, texts, out_bin_path):
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    all_ids = []
    for text in tqdm(texts, desc=f"tokenizing -> {out_bin_path.name}"):
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
        all_ids.append(eot_id)  # separate stories so the model learns document boundaries
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(out_bin_path)
    print(f"[tokenize] wrote {len(arr):,} tokens -> {out_bin_path}")
    return len(arr)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading {args.dataset} (train[:{args.max_samples}], "
          f"validation[:{args.max_val_samples}])...")
    train_ds = load_dataset(args.dataset, split=f"train[:{args.max_samples}]")
    val_ds = load_dataset(args.dataset, split=f"validation[:{args.max_val_samples}]")

    train_texts = [row["text"] for row in train_ds]
    val_texts = [row["text"] for row in val_ds]
    print(f"[data] {len(train_texts):,} train stories, {len(val_texts):,} val stories")

    if args.reuse_tokenizer:
        print(f"[tokenizer] reusing existing tokenizer from {args.reuse_tokenizer} "
              f"(NOT training a new one)")
        tokenizer = Tokenizer.from_file(args.reuse_tokenizer)
        tokenizer_path = out_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))  # keep a copy alongside this data for reference
    else:
        tokenizer_path = out_dir / "tokenizer.json"
        tokenizer = train_tokenizer(train_texts, args.vocab_size, tokenizer_path)

    train_tokens = tokenize_split(tokenizer, train_texts, out_dir / "train.bin")
    val_tokens = tokenize_split(tokenizer, val_texts, out_dir / "val.bin")

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_stories": len(train_texts),
        "val_stories": len(val_texts),
    }
    import json
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {meta}")


if __name__ == "__main__":
    main()
