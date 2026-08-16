"""`gpt-train` — train the configured model from scratch (or resume)."""

import argparse
import os

from ..config import load_settings
from ..training import train as run_training


def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--preset", default=None,
                        help="Model size preset (default: $GPT_PRESET or '153m')")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoints and start fresh")
    args = parser.parse_args()

    model_cfg, train_cfg, paths, label = load_settings(args.preset)
    resume = not args.no_resume and os.getenv("RESUME_TRAINING", "1") == "1"
    run_training(model_cfg, train_cfg, paths, label, resume=resume)


if __name__ == "__main__":
    main()
