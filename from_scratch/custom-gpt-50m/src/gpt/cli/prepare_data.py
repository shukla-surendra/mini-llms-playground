"""`gpt-data` — download the registered datasets and build the training corpus."""

import argparse
import os

from ..config import load_settings
from ..data.prepare import build_corpus, load_book_chunks
from ..data.sources import DATASETS


def main():
    parser = argparse.ArgumentParser(
        description="Download and build the training corpus from the registered datasets.",
    )
    parser.add_argument("--no-gated", action="store_true",
                        help="Skip gated datasets (LMSYS-Chat-1M) — no HF token needed")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse already-downloaded files under data/raw/")
    parser.add_argument("--max-per-dataset", type=int, default=100_000,
                        help="Cap on conversations kept per dataset")
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--min-turn-chars", type=int, default=24)
    parser.add_argument("--min-ascii-ratio", type=float, default=0.995)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true",
                        help="List the registered datasets and exit")
    parser.add_argument("--books-jsonl", default=None,
                        help="Path to tools/corpus-extractor's JSONL output "
                             "(e.g. data/books_staging/dataset.jsonl) — pooled and "
                             "shuffled in alongside the chat conversations. "
                             "See docs/BOOKS_CORPUS_INTEGRATION.md.")
    args = parser.parse_args()

    if args.list:
        for source in DATASETS:
            flag = "gated " if source.gated else "public"
            print(f"[{flag}] {source.hf_id:<40} {source.name}")
            print(f"           {source.summary}")
        return

    extra_documents = ()
    if args.books_jsonl:
        extra_documents = load_book_chunks(args.books_jsonl)
        print(f"[info] loaded {len(extra_documents):,} extra document(s) from {args.books_jsonl}")

    _, _, paths, _ = load_settings()
    stats = build_corpus(
        paths=paths,
        include_gated=not args.no_gated,
        token=os.getenv("HF_TOKEN"),
        max_per_dataset=args.max_per_dataset,
        min_turns=args.min_turns,
        min_turn_chars=args.min_turn_chars,
        min_ascii_ratio=args.min_ascii_ratio,
        num_prompts=args.num_prompts,
        train_ratio=args.train_ratio,
        seed=args.seed,
        skip_download=args.skip_download,
        extra_documents=extra_documents,
    )

    print("\n=== Corpus built ===")
    for hf_id, count in stats["per_source"].items():
        print(f"  {hf_id:<40} {count:>8,} conversations")
    print(f"  {'TOTAL':<40} {stats['total_conversations']:>8,}")
    if stats["extra_documents"]:
        print(f"  {'extra_documents (books)':<40} {stats['extra_documents']:>8,}")
    print(f"\n  train: {paths.train_data} ({stats['train_conversations']:,} conversations"
          f" + {stats['extra_train_documents']:,} book docs)")
    print(f"  test:  {paths.test_data} ({stats['test_conversations']:,} conversations"
          f" + {stats['extra_test_documents']:,} book docs)")
    print(f"  prompts: {paths.test_prompts} ({stats['prompts']} prompts)")


if __name__ == "__main__":
    main()
