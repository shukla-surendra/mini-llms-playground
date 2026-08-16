"""
FullyShardedDataParallel (FSDP) training — same model, same data, same launch mechanism as
train_ddp.py, but a genuinely different distribution strategy: instead of every rank
holding a full replica of the model (DDP) and only synchronizing gradients, FSDP *shards*
the model's parameters, gradients, and optimizer state themselves across ranks, and
temporarily reassembles a given layer's full parameters via an all-gather right before
that layer needs them. See docs/DISTRIBUTED_TRAINING.md for the mechanism in depth and why
this trade (more communication, less memory) is the whole point of FSDP.

Launch directly — `python train_fsdp.py` — same `mp.spawn` + explicit `tcp://` rationale
as train_ddp.py (see that file's docstring): `torchrun`'s elastic rendezvous hung on a
reverse-DNS lookup in this project's actual development environment, a real reproduced
issue documented in docs/DISTRIBUTED_TRAINING.md. `torchrun --nproc_per_node=N
train_fsdp.py` is still the expected production launch path elsewhere.

Same hardware-honesty note as train_ddp.py: this runs as CPU multi-process via `gloo`, not
real multi-GPU — a mechanism proof, not a scaling benchmark. FSDP's actual purpose (fit a
model too large for any single GPU's memory) can't be meaningfully demonstrated on a
6M-parameter model that already fits trivially anywhere — see
docs/DISTRIBUTED_TRAINING.md for what this script does and doesn't prove as a result.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import trange

from model import build_model

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 2))
MASTER_PORT = int(os.getenv("MASTER_PORT", 29501))

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

    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{MASTER_PORT}",
        rank=rank, world_size=world_size,
    )
    device = torch.device("cpu")
    torch.manual_seed(1234 + rank)

    if is_main:
        print(f"[fsdp] world_size={world_size} backend=gloo device={device}")

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

    # FSDP's actual mechanism, unlike DDP's: parameters, gradients, AND optimizer state
    # are sharded across ranks from the start — each rank only ever permanently holds
    # 1/world_size of each. Right before a given submodule's forward/backward needs its
    # full (unsharded) parameters, FSDP all-gathers the missing shards from every other
    # rank, uses them, then immediately frees everything except this rank's own shard
    # again. More communication than DDP (which only all-reduces gradients once per step)
    # in exchange for a peak memory footprint that doesn't scale with the model's full
    # size on any single rank — the entire reason FSDP exists.
    #
    # device_id=torch.device("cpu") is required here, explicitly, on this machine — a
    # real issue hit during development, documented in docs/DISTRIBUTED_TRAINING.md:
    # without it, FSDP auto-detects a compute device via torch._C._get_accelerator(),
    # which returns "mps" on any Apple Silicon Mac regardless of what device the model
    # tensors are actually on, then crashes because torch.mps doesn't implement the full
    # CUDA-like device-handle interface (`current_device()`) FSDP expects from an
    # accelerator backend. Passing device_id explicitly skips that auto-detection.
    fsdp_model = FSDP(model, device_id=torch.device("cpu"))
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=lr, weight_decay=0.1)

    if is_main:
        print(f"[model] {model.num_parameters():,} parameters (sharded across {world_size} ranks)")
        progress = trange(steps, desc="training", unit="step")

    for step in range(steps):
        if is_main and (step % eval_interval == 0 or step == steps - 1):
            val_loss = estimate_loss(fsdp_model, val_tokens, context_length, batch_size, eval_batches)
            progress.set_postfix(val_loss=f"{val_loss:.3f}")

        xb, yb = get_batch(train_tokens, context_length, batch_size)
        logits = fsdp_model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), max_norm=1.0)
        optimizer.step()

        if is_main:
            progress.set_postfix(loss=f"{loss.item():.3f}")
            progress.update(1)

    if is_main:
        progress.close()
        print("[done] FSDP run complete (checkpointing a sharded model needs "
              "FSDP.state_dict_type/full-state-dict gathering, deliberately out of scope "
              "here — see docs/DISTRIBUTED_TRAINING.md)")

    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" in os.environ:
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
    else:
        mp.spawn(main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
