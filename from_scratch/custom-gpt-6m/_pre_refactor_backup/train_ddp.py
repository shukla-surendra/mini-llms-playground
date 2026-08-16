"""
DistributedDataParallel (DDP) training — wraps the exact same causal-LM model and data
this project's train.py already trains, adding only the distributed-training machinery on
top. See docs/DISTRIBUTED_TRAINING.md for the mechanism and what "distributed" actually
means on this specific machine (multi-process CPU, not multi-GPU — see below).

Launch directly — `python train_ddp.py` — no external launcher needed. This uses
`torch.multiprocessing.spawn` internally rather than `torchrun`, a deliberate choice
documented in docs/DISTRIBUTED_TRAINING.md's gotcha section: `torchrun`'s elastic-agent
rendezvous does a reverse-DNS lookup that hung indefinitely in this project's actual
development environment (a real, reproduced issue, not a hypothetical) — `mp.spawn` with
an explicit `tcp://127.0.0.1:PORT` init method sets up the exact same process group
without that lookup. `torchrun --nproc_per_node=N train_ddp.py` is still expected to work
as the standard production launch method on a normal (non-sandboxed) machine; see the docs
for both paths.

Hardware honesty, upfront: this machine is Apple Silicon (MPS), no CUDA, no multiple real
GPUs. DDP here runs multiple CPU processes on one machine using the `gloo` backend — this
proves the DDP API and gradient-sync mechanism work correctly, genuinely, but it is NOT a
speed benchmark and won't show the near-linear scaling DDP gets across real GPUs on
separate (or NVLink-connected) devices. The production path is `nccl` + one process per
real GPU; `gloo` + CPU is this project's honest substitute for a mechanism demo without
GPU hardware. See docs/DISTRIBUTED_TRAINING.md's benchmark section for what was and wasn't
actually measured here.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange

from model import build_model

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 2))
MASTER_PORT = int(os.getenv("MASTER_PORT", 29500))

# -------- CONFIG --------
data_dir = Path(os.getenv("DATA_DIR", "data"))
context_length = int(os.getenv("CONTEXT_LENGTH", 256))
embed_size = int(os.getenv("EMBED_SIZE", 256))
num_heads = int(os.getenv("NUM_HEADS", 8))
num_layers = int(os.getenv("NUM_LAYERS", 6))
dropout = float(os.getenv("DROPOUT", 0.1))

batch_size = int(os.getenv("BATCH_SIZE", 16))  # per-rank batch size, not global
lr = float(os.getenv("LR", 3e-4))
steps = int(os.getenv("STEPS", 200))
eval_interval = int(os.getenv("EVAL_INTERVAL", 50))
eval_batches = int(os.getenv("EVAL_BATCHES", 10))


def load_meta():
    with open(data_dir / "meta.json") as f:
        return json.load(f)


def load_tokens(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def get_batch(tokens, ctx_len, bsz):
    max_start = len(tokens) - ctx_len - 1
    ix = torch.randint(0, max_start, (bsz,))
    x = torch.stack([tokens[i:i + ctx_len] for i in ix])
    y = torch.stack([tokens[i + 1:i + ctx_len + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, tokens, ctx_len, bsz, n_batches):
    model.eval()
    losses = []
    for _ in range(n_batches):
        xb, yb = get_batch(tokens, ctx_len, bsz)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main(rank, world_size):
    is_main = rank == 0

    # gloo, not nccl: nccl requires CUDA GPUs, unavailable here. gloo supports CPU
    # tensors, which is what makes a CPU-only distributed demo possible at all. See the
    # module docstring and docs/DISTRIBUTED_TRAINING.md for why this is a mechanism proof,
    # not a GPU-cluster substitute.
    #
    # init_method is an explicit tcp:// IP literal, not the default env:// (which is what
    # torchrun sets RANK/WORLD_SIZE/MASTER_ADDR for) — see the module docstring for why:
    # torchrun's own rendezvous handshake hung in this project's actual development
    # environment, and this explicit form sidesteps it entirely while setting up the
    # identical process group.
    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{MASTER_PORT}",
        rank=rank, world_size=world_size,
    )
    device = torch.device("cpu")
    torch.manual_seed(1234 + rank)  # deliberately different per rank -> different batches

    if is_main:
        print(f"[ddp] world_size={world_size} backend=gloo device={device}")

    meta = load_meta()
    train_tokens = load_tokens(data_dir / "train.bin")
    val_tokens = load_tokens(data_dir / "val.bin")

    model = build_model(
        vocab_size=meta["vocab_size"],
        context_length=context_length,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # DDP's actual mechanism: broadcast rank 0's initial weights to every other rank at
    # construction time (so every replica starts identical), then, during every
    # loss.backward() call, all-reduce (average) gradients across all ranks before any
    # optimizer.step() runs — each rank ends up with the *same* averaged gradient, applies
    # the *same* update, and the replicas stay in sync for the entire run without ever
    # explicitly synchronizing parameters again.
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=lr, weight_decay=0.1)

    if is_main:
        print(f"[model] {model.num_parameters():,} parameters (replicated across {world_size} ranks)")
        progress = trange(steps, desc="training", unit="step")

    for step in range(steps):
        if is_main and (step % eval_interval == 0 or step == steps - 1):
            val_loss = estimate_loss(ddp_model, val_tokens, context_length, batch_size, eval_batches)
            progress.set_postfix(val_loss=f"{val_loss:.3f}")

        xb, yb = get_batch(train_tokens, context_length, batch_size)
        logits = ddp_model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # <- the all-reduce across ranks happens inside this call
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
        optimizer.step()

        if is_main:
            progress.set_postfix(loss=f"{loss.item():.3f}")
            progress.update(1)

    if is_main:
        progress.close()
        torch.save(model.state_dict(), "tinystories_gpt_ddp_checkpoint.pt")
        print("[done] saved tinystories_gpt_ddp_checkpoint.pt (rank 0 only)")

    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" in os.environ:
        # Launched via torchrun (env:// rendezvous) — the standard production path.
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
    else:
        # Launched directly (`python train_ddp.py`) — spawn WORLD_SIZE worker processes
        # ourselves, each calling main(rank, world_size) directly. See the module
        # docstring for why this project uses this as its primary, tested launch path.
        mp.spawn(main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
