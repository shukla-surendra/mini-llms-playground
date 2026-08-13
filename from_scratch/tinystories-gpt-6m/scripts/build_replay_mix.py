"""
Build a "replay" training set: a new dataset's tokens, with a sample of an OLDER
dataset's already-tokenized tokens mixed in, to reduce catastrophic forgetting during
continual training. See docs/CONTINUAL_TRAINING_LOW_RESOURCE.md for the full reasoning.

Works purely at the TOKEN level (numpy uint16 arrays already on disk) — never re-tokenizes
or re-touches the original text, and never re-downloads anything. This only works
correctly if every dataset involved was tokenized with the SAME tokenizer (see
--reuse-tokenizer in prepare_dataset.py); mixing token arrays from different vocabularies
would silently produce garbage.

Usage:
    python scripts/build_replay_mix.py \\
      --new data_v2/train.bin \\
      --replay-from data/train.bin \\
      --replay-fraction 0.3 \\
      --out data_v2/train_with_replay.bin
"""
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Mix a sample of old tokenized data into a new dataset")
    p.add_argument("--new", required=True, help="Path to the new dataset's train.bin")
    p.add_argument("--replay-from", action="append", required=True,
                    help="Path to an OLDER dataset's train.bin to sample replay tokens from. "
                         "Pass multiple times to replay from several prior datasets at once.")
    p.add_argument("--replay-fraction", type=float, default=0.3,
                    help="Target fraction of the OUTPUT that should be replay tokens "
                         "(0.3 = 30%% replay, 70%% new data)")
    p.add_argument("--out", required=True, help="Output path for the mixed train.bin")
    p.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible mixes")
    return p.parse_args()


def sample_contiguous_chunk(tokens, n_tokens, rng):
    """Take one random contiguous slice, rather than fully shuffling — cheaper, and still
    gives varied replay content since the slice's start position is random each run."""
    if n_tokens >= len(tokens):
        return tokens.copy()
    start = rng.integers(0, len(tokens) - n_tokens)
    return tokens[start:start + n_tokens].copy()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    new_tokens = np.fromfile(args.new, dtype=np.uint16)
    print(f"[new] {args.new}: {len(new_tokens):,} tokens")

    # Solve for how many replay tokens are needed so that
    # replay_tokens / (replay_tokens + len(new_tokens)) == replay_fraction
    f = args.replay_fraction
    if not (0.0 <= f < 1.0):
        raise ValueError("--replay-fraction must be in [0, 1)")
    total_replay_needed = int(len(new_tokens) * f / (1 - f)) if f > 0 else 0

    replay_chunks = []
    if total_replay_needed > 0:
        per_source = total_replay_needed // len(args.replay_from)
        for src_path in args.replay_from:
            src_tokens = np.fromfile(src_path, dtype=np.uint16)
            chunk = sample_contiguous_chunk(src_tokens, per_source, rng)
            print(f"[replay] {src_path}: sampled {len(chunk):,} tokens "
                  f"(source has {len(src_tokens):,} total)")
            replay_chunks.append(chunk)

    # Replay and new tokens are kept as contiguous blocks, not shuffled token-by-token —
    # shuffling individual tokens would destroy sentence structure entirely. train.py's
    # get_batch already samples random windows across the whole file, so contiguous
    # blocks vs. fully interleaved content doesn't meaningfully change training dynamics.
    combined = np.concatenate(replay_chunks + [new_tokens]) if replay_chunks else new_tokens

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.tofile(out_path)

    actual_replay_frac = (len(combined) - len(new_tokens)) / len(combined) if len(combined) else 0
    print(f"[done] wrote {len(combined):,} tokens -> {out_path} "
          f"(actual replay fraction: {actual_replay_frac:.1%})")


if __name__ == "__main__":
    main()
