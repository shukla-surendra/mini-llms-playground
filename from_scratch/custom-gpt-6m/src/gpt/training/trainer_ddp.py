"""
DistributedDataParallel (DDP) training — wraps the exact same causal-LM model and data
this project's trainer.py already trains, adding only the distributed-training machinery
on top. See docs/DISTRIBUTED_TRAINING.md for the mechanism and what "distributed" actually
means on this specific machine (multi-process CPU, not multi-GPU — see below).

Launch directly — `gpt-train-ddp` — no external launcher needed. This uses
`torch.multiprocessing.spawn` internally rather than `torchrun`, a deliberate choice
documented in docs/DISTRIBUTED_TRAINING.md's gotcha section: `torchrun`'s elastic-agent
rendezvous does a reverse-DNS lookup that hung indefinitely in this project's actual
development environment (a real, reproduced issue, not a hypothetical) — `mp.spawn` with
an explicit `tcp://127.0.0.1:PORT` init method sets up the exact same process group
without that lookup. `torchrun --nproc_per_node=N -m gpt.training.trainer_ddp` is still
expected to work as the standard production launch method on a normal (non-sandboxed)
machine; see the docs for both paths.

Hardware honesty, upfront: this machine is Apple Silicon (MPS), no CUDA, no multiple real
GPUs. DDP here runs multiple CPU processes on one machine using the `gloo` backend — this
proves the DDP API and gradient-sync mechanism work correctly, genuinely, but it is NOT a
speed benchmark and won't show the near-linear scaling DDP gets across real GPUs on
separate (or NVLink-connected) devices. The production path is `nccl` + one process per
real GPU; `gloo` + CPU is this project's honest substitute for a mechanism demo without
GPU hardware. See docs/DISTRIBUTED_TRAINING.md's benchmark section for what was and wasn't
actually measured here.

Architecture (embed_size/num_layers/...) comes from the shared config.py, so
`GPT_PRESET`/`GPT_EMBED_SIZE`/... apply here exactly like every other trainer. Training
hyperparameters (steps/batch_size/lr/eval_*) deliberately do NOT come from the shared
TrainConfig — this is a mechanism demo meant to finish in seconds, not a real training
run, so it keeps its own small dedicated defaults (STEPS=200 vs TrainConfig's 5,000)
rather than silently inheriting production-scale settings.
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import trange

from ..config import Paths, resolve_distributed_config, resolve_model_config, resolve_vocab_size
from ..model import build_model
from .distributed import estimate_loss, get_batch, load_meta, load_tokens

# -------- CONFIG (this demo's own scale, not shared TrainConfig — see module docstring) --------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))  # per-rank batch size, not global
LR = float(os.getenv("LR", 3e-4))
STEPS = int(os.getenv("STEPS", 200))
EVAL_INTERVAL = int(os.getenv("EVAL_INTERVAL", 50))
EVAL_BATCHES = int(os.getenv("EVAL_BATCHES", 10))


def _worker(rank, world_size, master_port, model_cfg, label):
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
        backend="gloo", init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank, world_size=world_size,
    )
    device = torch.device("cpu")
    torch.manual_seed(1234 + rank)  # deliberately different per rank -> different batches

    if is_main:
        print(f"[ddp] world_size={world_size} backend=gloo device={device}")

    paths = Paths(label=label, objective="ddp")
    meta = load_meta(paths)
    model_cfg = resolve_vocab_size(model_cfg, meta)
    train_tokens = load_tokens(paths.train_bin)
    val_tokens = load_tokens(paths.val_bin)

    model = build_model(
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        embed_size=model_cfg.embed_size,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
    ).to(device)

    # DDP's actual mechanism: broadcast rank 0's initial weights to every other rank at
    # construction time (so every replica starts identical), then, during every
    # loss.backward() call, all-reduce (average) gradients across all ranks before any
    # optimizer.step() runs — each rank ends up with the *same* averaged gradient, applies
    # the *same* update, and the replicas stay in sync for the entire run without ever
    # explicitly synchronizing parameters again.
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=LR, weight_decay=0.1)

    if is_main:
        print(f"[model] {model.num_parameters():,} parameters (replicated across {world_size} ranks)")
        progress = trange(STEPS, desc="training", unit="step")

    for step in range(STEPS):
        if is_main and (step % EVAL_INTERVAL == 0 or step == STEPS - 1):
            val_loss = estimate_loss(ddp_model, val_tokens, model_cfg.context_length, BATCH_SIZE, EVAL_BATCHES)
            progress.set_postfix(val_loss=f"{val_loss:.3f}")

        xb, yb = get_batch(train_tokens, model_cfg.context_length, BATCH_SIZE)
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
        paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), paths.final_checkpoint)
        print(f"[done] saved {paths.final_checkpoint} (rank 0 only)")

    dist.destroy_process_group()


def run(preset_name=None):
    model_cfg, label = resolve_model_config(preset_name)
    dist_cfg = resolve_distributed_config(default_master_port=29500)

    if "RANK" in os.environ:
        # Launched via torchrun (env:// rendezvous) — the standard production path.
        _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), dist_cfg.master_port, model_cfg, label)
    else:
        # Launched directly (`gpt-train-ddp`) — spawn WORLD_SIZE worker processes
        # ourselves, each calling _worker(rank, world_size, ...) directly. See the module
        # docstring for why this project uses this as its primary, tested launch path.
        mp.spawn(_worker, args=(dist_cfg.world_size, dist_cfg.master_port, model_cfg, label),
                  nprocs=dist_cfg.world_size, join=True)
