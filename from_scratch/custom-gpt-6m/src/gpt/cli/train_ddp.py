"""`gpt-train-ddp` — DistributedDataParallel mechanism demo (multi-process CPU, gloo).

See src/gpt/training/trainer_ddp.py's module docstring for the full mechanism and
hardware-honesty notes. WORLD_SIZE/MASTER_PORT are read as env vars (see
config.resolve_distributed_config); STEPS/BATCH_SIZE/... are this demo's own small-scale
env vars, not the shared TrainConfig.
"""

import argparse

from ..training.trainer_ddp import run


def main():
    parser = argparse.ArgumentParser(description="DDP training demo.")
    parser.add_argument("--preset", default=None,
                        help="Model size preset (default: $GPT_PRESET or '6m')")
    args = parser.parse_args()

    run(preset_name=args.preset)


if __name__ == "__main__":
    main()
