"""`gpt-train-mlm` — train the masked-LM (BERT-style) objective."""

import argparse

from ..training.trainer_mlm import run


def main():
    parser = argparse.ArgumentParser(description="Train the masked-LM model.")
    parser.add_argument("--preset", default=None,
                        help="Model size preset (default: $GPT_PRESET or '6m')")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoints and start fresh")
    args = parser.parse_args()

    run(preset_name=args.preset, resume=not args.no_resume)


if __name__ == "__main__":
    main()
