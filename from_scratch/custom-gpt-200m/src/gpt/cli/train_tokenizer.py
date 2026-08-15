"""`gpt-train-tokenizer` — train this project's BPE vocabulary on the corpus.

Must run **before** `gpt-tokenize` and `gpt-train`: the model's embedding table is
sized to this vocabulary, so the tokenizer is a prerequisite, not an option.

    gpt-train-tokenizer                     # train on data/train.txt at VOCAB_SIZE
    gpt-train-tokenizer --vocab-size 24000  # compare a smaller vocabulary
    gpt-train-tokenizer --sample-mb 512     # train on a slice, for a quick look

Retraining invalidates every existing `.bin` and every checkpoint — different ids mean
different embedding rows. `gpt-tokenize` rebuilds `.bin` files automatically when the
tokenizer is newer, but checkpoints cannot be migrated and must be retrained.
"""

import argparse
from pathlib import Path
import time

from ..config import TOKENIZER_PATH, VOCAB_SIZE, load_settings
from ..tokenizer import END_OF_TEXT, train_tokenizer


def _sample_corpus(src, out, megabytes):
    """Copy the first `megabytes` MB of `src`, cut at a document boundary."""
    budget = megabytes * 1024 * 1024
    written = 0
    with open(src, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        while written < budget:
            chunk = fin.read(8 * 1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
            written += len(chunk)
    return out


def main():
    p = argparse.ArgumentParser(description="Train this project's BPE tokenizer.")
    p.add_argument("--preset", default=None)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--out", default=TOKENIZER_PATH)
    p.add_argument("--input", action="append", default=None,
                   help="Corpus file(s) to train on (default: data/train.txt)")
    p.add_argument("--sample-mb", type=int, default=None,
                   help="Train on only the first N MB — faster, and usually enough: "
                        "BPE merge statistics converge long before a full corpus is read")
    args = p.parse_args()

    _, _, paths, _ = load_settings(args.preset)
    inputs = [Path(f) for f in args.input] if args.input else [paths.train_data]
    missing = [f for f in inputs if not f.exists()]
    if missing:
        raise SystemExit(f"Corpus not found: {', '.join(str(m) for m in missing)}")

    if args.sample_mb:
        sampled = Path(paths.data_dir) / "_tokenizer_sample.txt"
        print(f"Sampling first {args.sample_mb} MB of {inputs[0]} -> {sampled}")
        inputs = [_sample_corpus(inputs[0], sampled, args.sample_mb)]

    total_mb = sum(f.stat().st_size for f in inputs) / (1024 * 1024)
    print(f"Training BPE: vocab_size={args.vocab_size:,} over {total_mb:,.0f} MB")
    print("  pre-tokenizer: Digits(individual_digits=True) -> ByteLevel")
    started = time.time()
    tok = train_tokenizer(inputs, args.vocab_size, args.out)
    print(f"Saved {args.out}  ({time.time() - started:,.0f}s, "
          f"{tok.n_vocab:,} tokens, {END_OF_TEXT} = id {tok.eot_id})")

    if tok.n_vocab != VOCAB_SIZE:
        print(
            f"\n!! config.VOCAB_SIZE is {VOCAB_SIZE:,} but this tokenizer holds "
            f"{tok.n_vocab:,}.\n"
            f"   Update config.VOCAB_SIZE — the embedding table is sized from it and "
            f"training refuses to start on a mismatch."
        )

    # A vocabulary is only as good as its compression on real text; show it rather
    # than make the caller go looking.
    sample = ("The bakery sold 48 cupcakes in the morning and 27 in the afternoon, "
              "so 48 + 27 = 75 cupcakes were sold in total.")
    ids = tok.encode(sample)
    print(f"\n  sample      : {sample}")
    print(f"  tokens ({len(ids)}) : {[tok.decode([i]) for i in ids][:28]}")
    print(f"  chars/token : {len(sample) / len(ids):.2f}")


if __name__ == "__main__":
    main()
