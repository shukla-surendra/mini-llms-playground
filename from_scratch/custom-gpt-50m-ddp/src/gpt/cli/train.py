"""`gpt-train` — train the configured model from scratch (or resume).

Multi-GPU (DDP): launch with `torchrun --nproc_per_node=N -m gpt.cli.train` instead of
`gpt-train` directly. torchrun sets RANK/WORLD_SIZE/LOCAL_RANK env vars per spawned
process before this module even imports — this file reads them, initializes the
process group, and picks the per-rank CUDA device (`cuda:{LOCAL_RANK}`), before handing
off to the same `train()` every single-process run already uses. With no torchrun
(WORLD_SIZE unset/1), everything here is a no-op and behavior is identical to before
DDP support existed.

Backend: `nccl` when CUDA is actually available (the real multi-GPU path this exists
for), `gloo` otherwise — lets `WORLD_SIZE=2 torchrun --nproc_per_node=2 -m gpt.cli.train`
run as a local CPU smoke test (multiple processes on one machine, no real GPUs) to
verify the DDP mechanism/checkpoint compatibility before ever renting real hardware,
same spirit as the sibling custom-gpt-6m project's CPU-only DDP demo.
"""

import argparse
import os

import torch
import torch.distributed as dist

from ..config import load_settings
from ..runtime import get_device
from ..training import train as run_training


def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--preset", default=None,
                        help="Model size preset (default: $GPT_PRESET or '50m')")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoints and start fresh")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = None
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            # CPU smoke-test path (see module docstring) — gloo supports CPU tensors,
            # nccl does not, so there is no per-rank "device index" to select here.
            device = torch.device("cpu")
        if rank == 0:
            print(f"[ddp] world_size={world_size} backend={backend} device={device}")
    else:
        device = get_device()

    model_cfg, train_cfg, paths, label = load_settings(args.preset, world_size=world_size)
    resume = not args.no_resume and os.getenv("RESUME_TRAINING", "1") == "1"
    try:
        run_training(model_cfg, train_cfg, paths, label, resume=resume, device=device,
                      rank=rank, world_size=world_size, local_rank=local_rank)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
