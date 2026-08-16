"""
FullyShardedDataParallel (FSDP) training — same model, same data, same launch mechanism as
trainer_ddp.py, but a genuinely different distribution strategy: instead of every rank
holding a full replica of the model (DDP) and only synchronizing gradients, FSDP *shards*
the model's parameters, gradients, and optimizer state themselves across ranks, and
temporarily reassembles a given layer's full parameters via an all-gather right before
that layer needs them. See docs/DISTRIBUTED_TRAINING.md for the mechanism in depth and why
this trade (more communication, less memory) is the whole point of FSDP.

Launch directly — `gpt-train-fsdp` — same `mp.spawn` + explicit `tcp://` rationale as
trainer_ddp.py (see that module's docstring): `torchrun`'s elastic rendezvous hung on a
reverse-DNS lookup in this project's actual development environment, a real reproduced
issue documented in docs/DISTRIBUTED_TRAINING.md. `torchrun --nproc_per_node=N
-m gpt.training.trainer_fsdp` is still the expected production launch path elsewhere.

Same hardware-honesty note as trainer_ddp.py: this runs as CPU multi-process via `gloo`,
not real multi-GPU — a mechanism proof, not a scaling benchmark. FSDP's actual purpose (fit
a model too large for any single GPU's memory) can't be meaningfully demonstrated on a
6M-parameter model that already fits trivially anywhere — see docs/DISTRIBUTED_TRAINING.md
for what this module does and doesn't prove as a result.

Same architecture-vs-hyperparameters split as trainer_ddp.py: ModelConfig comes from the
shared config.py (GPT_PRESET/GPT_EMBED_SIZE/... apply), training hyperparameters keep
their own small demo-scale defaults rather than inheriting production-scale TrainConfig.
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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

    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank, world_size=world_size,
    )
    device = torch.device("cpu")
    torch.manual_seed(1234 + rank)

    if is_main:
        print(f"[fsdp] world_size={world_size} backend=gloo device={device}")

    paths = Paths(label=label, objective="fsdp")
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
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=LR, weight_decay=0.1)

    if is_main:
        print(f"[model] {model.num_parameters():,} parameters (sharded across {world_size} ranks)")
        progress = trange(STEPS, desc="training", unit="step")

    for step in range(STEPS):
        if is_main and (step % EVAL_INTERVAL == 0 or step == STEPS - 1):
            val_loss = estimate_loss(fsdp_model, val_tokens, model_cfg.context_length, BATCH_SIZE, EVAL_BATCHES)
            progress.set_postfix(val_loss=f"{val_loss:.3f}")

        xb, yb = get_batch(train_tokens, model_cfg.context_length, BATCH_SIZE)
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


def run(preset_name=None):
    model_cfg, label = resolve_model_config(preset_name)
    dist_cfg = resolve_distributed_config(default_master_port=29501)

    if "RANK" in os.environ:
        _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), dist_cfg.master_port, model_cfg, label)
    else:
        mp.spawn(_worker, args=(dist_cfg.world_size, dist_cfg.master_port, model_cfg, label),
                  nprocs=dist_cfg.world_size, join=True)
