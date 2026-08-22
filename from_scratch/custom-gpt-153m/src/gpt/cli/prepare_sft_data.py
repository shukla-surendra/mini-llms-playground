"""`gpt-sft-data` — download the registered chat/instruction datasets and build the
SFT corpus (data/sft/{train,test}.jsonl), turn-structure preserved for masked loss.

Same registry, same per-source download/parse/filter pipeline as `gpt-data`
(cli/prepare_data.py) — see data/sft_prepare.py's docstring for why the output differs.
"""

import argparse
import os

from ..config import load_settings
from ..data.sft_prepare import build_sft_corpus
from ..data.sources import DATASETS


def main():
    parser = argparse.ArgumentParser(
        description="Download and build the SFT corpus from the registered chat/instruction datasets.",
    )
    parser.add_argument("--no-gated", action="store_true",
                        help="Skip gated datasets (LMSYS-Chat-1M) — no HF token needed")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse already-downloaded files under data/raw/")
    parser.add_argument("--max-per-dataset", type=int, default=100_000,
                        help="Cap on conversations kept per dataset")
    parser.add_argument("--min-turns", type=int, default=2,
                        help="Minimum turns to keep a conversation (2 = one exchange).")
    parser.add_argument("--min-turn-chars", type=int, default=24)
    parser.add_argument("--min-ascii-ratio", type=float, default=0.995)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true",
                        help="List the registered datasets and exit")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes for downloading+parsing sources in "
                             "parallel. Default: os.cpu_count(), capped at the number "
                             "of sources selected. --workers 1 forces sequential.")
    args = parser.parse_args()

    if args.list:
        for source in DATASETS:
            flag = "gated " if source.gated else "public"
            print(f"[{flag}] {source.hf_id:<40} {source.name}")
            print(f"           {source.summary}")
        return

    _, _, paths, _ = load_settings()
    token = os.getenv("HF_TOKEN")

    result = build_sft_corpus(
        data_dir=paths.data_dir,
        include_gated=not args.no_gated,
        token=token,
        max_per_dataset=args.max_per_dataset,
        min_turns=args.min_turns,
        min_turn_chars=args.min_turn_chars,
        min_ascii_ratio=args.min_ascii_ratio,
        train_ratio=args.train_ratio,
        seed=args.seed,
        skip_download=args.skip_download,
        max_workers=args.workers,
    )

    print()
    print("Per-source conversations kept:")
    for hf_id, count in result["per_source"].items():
        print(f"  {hf_id:<40} {count:,}")
    print(
        f"\nTotal: {result['total_conversations']:,} conversations "
        f"({result['train_conversations']:,} train / {result['test_conversations']:,} test)"
    )
    print(f"Wrote: {paths.data_dir / 'sft' / 'train.jsonl'}")
    print(f"Wrote: {paths.data_dir / 'sft' / 'test.jsonl'}")


if __name__ == "__main__":
    main()
