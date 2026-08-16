"""`gpt-train-fsdp` — FullyShardedDataParallel mechanism demo (multi-process CPU, gloo).

See src/gpt/training/trainer_fsdp.py's module docstring for the full mechanism and
hardware-honesty notes.
"""

import argparse

from ..training.trainer_fsdp import run


def main():
    parser = argparse.ArgumentParser(description="FSDP training demo.")
    parser.add_argument("--preset", default=None,
                        help="Model size preset (default: $GPT_PRESET or '6m')")
    args = parser.parse_args()

    run(preset_name=args.preset)


if __name__ == "__main__":
    main()
