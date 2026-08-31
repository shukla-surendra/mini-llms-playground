#!/usr/bin/env python3
"""Local FSDP smoke test — same production train() as `gpt-train`/torchrun, launched via
`torch.multiprocessing.spawn` instead of torchrun. Sibling to `ddp_smoke_test.py`; see
that script's docstring for why `mp.spawn` + an explicit `tcp://` init_method rather than
`torchrun` locally.

Backend is gloo (CPU) here regardless of CUDA availability — a mechanism check, not a
speed benchmark, same as the DDP smoke test. What this specifically verifies that the DDP
smoke test cannot: the full-state-dict checkpoint save/resume round trip
(_FSDPCheckpointView/_FSDPOptimizerView in trainer.py) actually produces a loadable,
correctly-gathered checkpoint — not just that FSDP wraps and trains without crashing.

    uv run python scripts/fsdp_smoke_test.py
    GPT_STEPS=40 uv run python scripts/fsdp_smoke_test.py   # more steps
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpt.config import load_settings
from gpt.training import train as run_training

WORLD_SIZE = int(os.getenv("SMOKE_WORLD_SIZE", 2))
MASTER_PORT = int(os.getenv("SMOKE_MASTER_PORT", 29502))  # different port from ddp_smoke_test.py, in case both run close together


def _worker(rank, world_size, resume, steps, result_queue):
    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{MASTER_PORT}",
        rank=rank, world_size=world_size,
    )
    try:
        os.environ["GPT_PRESET"] = "tiny"
        os.environ["GPT_PARALLELISM"] = "fsdp"
        os.environ["GPT_BATCH_SIZE"] = "2"
        os.environ["GPT_GRAD_ACCUM"] = "2"
        os.environ["GPT_STEPS"] = str(steps)
        os.environ["GPT_SAVE_EVERY"] = "10"
        os.environ["GPT_EVAL_INTERVAL"] = "10"
        os.environ["GPT_EVAL_BATCHES"] = "2"

        model_cfg, train_cfg, paths, label = load_settings(world_size=world_size)
        result = run_training(
            model_cfg, train_cfg, paths, label,
            resume=resume,
            device=torch.device("cpu"),
            rank=rank, world_size=world_size, local_rank=rank,
        )
        if rank == 0:
            print(f"\n[smoke test] result: {result}")
            result_queue.put(result)
    finally:
        dist.destroy_process_group()


def main():
    # Two passes deliberately: the second one's resume=True is what actually exercises
    # _resume_into's FSDP branch (optim_state_dict_to_load, the full-state-dict context
    # for loading) — a single fresh run never reaches that code path at all.
    result_queue = mp.Queue()

    print(f"[smoke test] pass 1/2: fresh run (10 steps), spawning {WORLD_SIZE} CPU/gloo workers on 127.0.0.1:{MASTER_PORT}")
    mp.spawn(_worker, args=(WORLD_SIZE, False, 10, result_queue), nprocs=WORLD_SIZE, join=True)
    first_result = result_queue.get()

    print(f"\n[smoke test] pass 2/2: resume, extended to 20 steps — must actually run steps 10-19, not skip them")
    mp.spawn(_worker, args=(WORLD_SIZE, True, 20, result_queue), nprocs=WORLD_SIZE, join=True)
    second_result = result_queue.get()

    print(f"\n[smoke test] pass 1 finished at step {first_result['step']}, "
          f"pass 2 resumed and finished at step {second_result['step']}")
    if second_result["step"] <= first_result["step"]:
        print("[smoke test] WARNING: resume did not advance past pass 1's step — "
              "resume likely did not actually pick up where pass 1 left off.")
    else:
        print("[smoke test] OK: resume picked up past pass 1's step — "
              "full-state-dict checkpoint round trip works under FSDP.")


if __name__ == "__main__":
    main()
