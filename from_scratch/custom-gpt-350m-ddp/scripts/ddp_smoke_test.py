#!/usr/bin/env python3
"""Local DDP smoke test — same production train() as `gpt-train`/torchrun, launched via
`torch.multiprocessing.spawn` instead of torchrun.

Why not just use torchrun locally: it can hang for minutes on some sandboxed/dev
machines doing a rendezvous DNS lookup — a documented gotcha in the sibling
custom-gpt-6m and custom-gpt-50m-ddp projects. mp.spawn with an explicit
tcp://127.0.0.1:PORT init_method sets up the exact same process group without that
lookup. This script is ONLY the local-verification path — the real multi-node run
still uses `torchrun --nnodes=2 --nproc_per_node=1 -m gpt.cli.train` across two actual
EC2 instances (see docs/GPU_TRAINING.md's "Multi-Node DDP on a Single AZ" section),
since torchrun is the standard production launcher and the DNS hang is specific to some
dev sandboxes, not real hardware.

Backend is gloo (CPU) here regardless of CUDA availability — this verifies the DDP
*mechanism* (gradients actually sync, loss decreases, checkpoints save/load with clean
non-`module.`-prefixed keys, no rank races on file writes) for near-$0 and in minutes,
before trusting any of that on rented L4 hardware. It is not a speed benchmark — see
`gpt-benchmark` (single-GPU only) plus a short real multi-node run for that.

    uv run python scripts/ddp_smoke_test.py
    GPT_STEPS=40 uv run python scripts/ddp_smoke_test.py   # more steps
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpt.config import load_settings
from gpt.training import train as run_training

WORLD_SIZE = int(os.getenv("SMOKE_WORLD_SIZE", 2))
MASTER_PORT = int(os.getenv("SMOKE_MASTER_PORT", 29501))


def _worker(rank, world_size):
    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{MASTER_PORT}",
        rank=rank, world_size=world_size,
    )
    try:
        # Tiny, fast-finishing overrides — this is a mechanism check, not a real run.
        # GPT_PRESET=tiny keeps this seconds long regardless of the 350m default's
        # batch_size=16/ctx=2048 (which would make even 20 CPU steps slow).
        os.environ.setdefault("GPT_PRESET", "tiny")
        os.environ.setdefault("GPT_BATCH_SIZE", "2")
        os.environ.setdefault("GPT_GRAD_ACCUM", "2")
        os.environ.setdefault("GPT_STEPS", "20")
        os.environ.setdefault("GPT_SAVE_EVERY", "10")
        os.environ.setdefault("GPT_EVAL_INTERVAL", "10")
        os.environ.setdefault("GPT_EVAL_BATCHES", "2")

        model_cfg, train_cfg, paths, label = load_settings(world_size=world_size)
        result = run_training(
            model_cfg, train_cfg, paths, label,
            resume=False,  # fresh start every smoke-test run, not resuming real training
            device=torch.device("cpu"),
            rank=rank, world_size=world_size, local_rank=rank,
        )
        if rank == 0:
            print(f"\n[smoke test] result: {result}")
    finally:
        dist.destroy_process_group()


def main():
    print(f"[smoke test] spawning {WORLD_SIZE} CPU/gloo workers on 127.0.0.1:{MASTER_PORT}")
    mp.spawn(_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
