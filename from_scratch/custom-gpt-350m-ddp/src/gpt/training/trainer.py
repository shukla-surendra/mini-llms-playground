"""The training loop.

Five steps per iteration, the same cycle that trains any transformer at any scale:
    forward -> loss -> backward -> optimizer step -> zero grads
with gradient accumulation so a batch_size of 1 still yields a large effective batch.
"""

import csv
from datetime import datetime, timedelta, timezone
import math
import os
import time

import numpy as np
import torch
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange

from ..checkpoint import atomic_save, is_compatible, make_payload
from ..config import TOKENIZER_NAME, TOKENIZER_PATH
from ..data import (
    effective_context_length,
    get_batch,
    load_token_array,
    next_token_loss,
)
from ..inference.generate import generate_text
from ..model import TinyGPT
from ..runtime import get_device
from ..tokenizer import load_tokenizer

EVAL_HISTORY_FIELDS = [
    "timestamp_utc",
    "step",
    "est_epoch",
    "lr",
    "train_loss",
    "test_loss",
    "test_perplexity",
    "best_test_loss",
    "improved",
    "processed_tokens",
    "total_training_hours",
]


def _fsdp_full_state_dict_ctx(training_model, rank0_only, device):
    """Context manager making `training_model.state_dict()`/`.load_state_dict()`
    operate on the FULL (unsharded, reassembled) model, not this rank's local shard.

    This is the one piece of FSDP machinery that has no DDP equivalent, because DDP
    never touches parameter storage in the first place — `raw_model.state_dict()`
    already returns the full model under DDP. FSDP, by contrast, replaces the wrapped
    module's own parameter storage with just this rank's shard *in place* the moment
    it wraps it (see trainer_fsdp.py's reference implementation and
    docs/DISTRIBUTED_TRAINING.md) — so `raw_model.state_dict()` after an FSDP wrap
    would silently return only a fraction of the model, not an error, which is exactly
    the kind of bug that looks like a working checkpoint until someone tries to resume
    from it. `rank0_only=True` (saving) means every non-main rank gets an empty dict
    back — harmless, since only rank 0 ever calls `atomic_save`.

    `offload_to_cpu` is only enabled on CUDA, deliberately — a real bug hit building
    this: on a CPU-only (gloo) run, `offload_to_cpu=True` crashes the unshard/gather
    with `RuntimeError: setStorage: ... storage of size 0` (confirmed with a minimal
    repro, isolated to this one flag). It makes sense once you know what the flag is
    for: it exists to move the gathered full model off GPU memory during a
    checkpoint, so there is nothing sensible for it to do when the model was never on
    a GPU in the first place. Real CUDA runs still get the memory benefit it's meant
    to provide; the CPU/gloo smoke test just doesn't ask for it.
    """
    return FSDP.state_dict_type(
        training_model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=rank0_only, offload_to_cpu=(device.type == "cuda")),
    )


class _FSDPCheckpointView:
    """Adapts an FSDP-wrapped model to the plain `model.state_dict()` / `.param_count()`
    / `.attn_impl` interface `checkpoint.make_payload` and the DDP/single-process path
    already use — so `_run_loop`'s `payload()` closure needs no `if parallelism ==
    "fsdp"` branch of its own; the branch lives here, once, instead of scattered across
    every call site that touches `model_state_dict`/`param_count`/`attn_impl`.

    `total_param_count` must be captured BEFORE FSDP wraps `raw_model` — afterward,
    `raw_model.parameters()` only iterates this rank's shard (see module docstring
    above), so `raw_model.param_count()` computed post-wrap would report roughly
    `1/world_size` of the true total, exactly the "DDP reports full, FSDP reports half"
    distinction `docs/DISTRIBUTED_TRAINING.md` calls out as the direct, observable proof
    the two mechanisms differ.
    """

    def __init__(self, training_model, raw_model, total_param_count, device):
        self._training_model = training_model
        self._raw_model = raw_model
        self._total_param_count = total_param_count
        self._device = device

    def state_dict(self):
        with _fsdp_full_state_dict_ctx(self._training_model, rank0_only=True, device=self._device):
            return self._training_model.state_dict()

    def param_count(self):
        return self._total_param_count

    @property
    def attn_impl(self):
        # A plain string attribute, not a tensor FSDP shards — always safe to read
        # straight off raw_model regardless of wrapping.
        return getattr(self._raw_model, "attn_impl", "naive")


class _FSDPOptimizerView:
    """Same adaptation as `_FSDPCheckpointView`, for the optimizer half of a checkpoint.

    `make_payload` calls `optimizer.state_dict()` directly — under FSDP, that returns
    only this rank's local shard of AdamW's moment/variance state, not the full
    picture, unless routed through `FSDP.optim_state_dict()` (a genuinely different,
    collective call, not just a config toggle on the plain method). This wraps that up
    behind the same `.state_dict()` interface `make_payload` already expects, so it
    needs no FSDP-awareness of its own.
    """

    def __init__(self, training_model, optimizer, device):
        self._training_model = training_model
        self._optimizer = optimizer
        self._device = device

    def state_dict(self):
        with _fsdp_full_state_dict_ctx(self._training_model, rank0_only=True, device=self._device):
            return FSDP.optim_state_dict(self._training_model, self._optimizer)


def format_eta(remaining_steps, steps_per_hour):
    """(hours, 'Mon 17 Aug 11:28') for a remaining-step count, or None if unknown."""
    if not steps_per_hour or steps_per_hour <= 0 or remaining_steps <= 0:
        return None
    hours = remaining_steps / steps_per_hour
    finish = datetime.now() + timedelta(hours=hours)
    return hours, finish.strftime("%a %d %b %H:%M")


def lr_for_step(step_idx, train_cfg):
    """Linear warmup, then cosine decay to min_lr."""
    warmup_steps = max(200, int(train_cfg.steps * 0.02))
    if step_idx < warmup_steps:
        return train_cfg.lr * float(step_idx + 1) / float(warmup_steps)
    decay_steps = max(1, train_cfg.steps - warmup_steps)
    progress = min(1.0, max(0.0, (step_idx - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    return train_cfg.min_lr + (train_cfg.lr - train_cfg.min_lr) * cosine


def safe_perplexity(loss_value):
    # Bound the exponent so early-training logs show a number rather than inf.
    return float(math.exp(min(float(loss_value), 20.0)))


def format_duration(total_seconds):
    total_seconds = int(max(0, total_seconds))
    return (
        f"{total_seconds // 3600:02d}:"
        f"{(total_seconds % 3600) // 60:02d}:"
        f"{total_seconds % 60:02d}"
    )


def append_eval_history(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_HISTORY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def resolve_amp(precision, device):
    """Return (device_type, dtype_or_None) for `torch.autocast`.

    "auto" means bfloat16 on CUDA and full fp32 everywhere else. bf16 rather than fp16
    because it has fp32's exponent range, so training needs no GradScaler and cannot
    silently underflow gradients — but it requires Ampere or newer (an L4 or A10G has
    it; a T4 does not). MPS/CPU stay fp32: MPS autocast is not dependable, and without
    tensor cores there is nothing to win.

    Returning dtype=None means "no autocast", which callers pass straight through as
    `enabled=False` so there is only one code path.
    """
    # Normalise first: `device` may be "cuda:0", but autocast wants a bare device type.
    device_type = "cuda" if str(device).startswith("cuda") else str(device)
    if precision == "fp32":
        return device_type, None
    if precision in ("bf16", "bfloat16"):
        return device_type, torch.bfloat16
    if precision in ("fp16", "float16"):
        return device_type, torch.float16
    if precision != "auto":
        raise ValueError(
            f"Unknown precision {precision!r}. Use auto, bf16, fp16 or fp32."
        )
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        return device_type, torch.bfloat16
    return device_type, None


@torch.no_grad()
def estimate_loss(model, train_tokens, test_tokens, ctx_len, vocab_size, train_cfg,
                  device, amp=None):
    model.eval()
    device_type, amp_dtype = amp if amp else resolve_amp(train_cfg.precision, device)
    out = {}
    for name, tokens in (("train", train_tokens), ("test", test_tokens)):
        losses = []
        for _ in range(train_cfg.eval_batches):
            xb, yb = get_batch(tokens, ctx_len, train_cfg.batch_size, device)
            with torch.autocast(device_type=device_type,
                                dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                loss = next_token_loss(model(xb), yb, vocab_size)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train(model_cfg, train_cfg, paths, label, resume=True, device=None,
          rank=0, world_size=1, local_rank=0):
    """`rank`/`world_size`/`local_rank` are DDP identity — defaults (0/1/0) reproduce
    single-process training exactly, unchanged from before DDP support existed. When
    `world_size > 1`, `device` must already be the caller's own `cuda:{local_rank}`
    (see cli/train.py) — this function does not construct it.

    Per-rank RNG seeding matters here specifically because `get_batch` (data/dataset.py)
    draws from numpy's *global* legacy RNG, not a per-call generator — without an
    explicit per-rank offset, every rank starts from whatever numpy's unseeded
    OS-entropy default happens to be per process, which is uncontrolled, not merely
    "the same everywhere" (the more familiar DDP footgun). Offsetting by `rank` makes
    it both controlled AND different per rank, so ranks draw different random windows
    instead of accidentally-random or accidentally-identical ones.
    """
    device = device or get_device()
    torch.manual_seed(train_cfg.seed + rank)
    np.random.seed(train_cfg.seed + rank)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    if tokenizer.n_vocab != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer '{TOKENIZER_NAME}' has {tokenizer.n_vocab} tokens but config "
            f"declares vocab_size={model_cfg.vocab_size}. Update config.VOCAB_SIZE."
        )

    # Disk-backed uint16 memmaps, built once by `gpt-tokenize` (and built on demand
    # here if missing/stale). Never materialises the corpus in RAM or VRAM — see
    # data/dataset.py's module docstring for why that matters at this scale.
    train_tokens = load_token_array(paths.train_data, tokenizer)
    test_tokens = load_token_array(paths.test_data, tokenizer)
    if len(train_tokens) < 2 or len(test_tokens) < 2:
        raise ValueError("Train/test corpora must each contain at least 2 tokens.")

    ctx_len = effective_context_length(model_cfg.context_length, train_tokens, test_tokens)
    if ctx_len < model_cfg.context_length:
        print(
            f"Info: reducing context_length {model_cfg.context_length} -> {ctx_len} "
            f"to fit the available corpus."
        )

    attn_impl = os.getenv("ATTN_IMPL", "sdpa")
    # `raw_model` is always the plain, unwrapped TinyGPT. Under DDP (and single-process)
    # it stays the complete, correct model throughout — DDP never touches parameter
    # storage, only adds gradient-sync hooks — so checkpoint save/load and the
    # end-of-run demo generation can go through it directly. Under FSDP that is NOT
    # true: FSDP replaces raw_model's own parameter storage with just this rank's shard
    # the moment it wraps it, so anything needing the FULL model (checkpointing, demo
    # generation) has to go through `training_model` inside `_fsdp_full_state_dict_ctx`
    # instead — see `_FSDPCheckpointView` above. `total_param_count` is captured here,
    # before any wrapping, for exactly that reason: after an FSDP wrap,
    # `raw_model.param_count()` would report roughly 1/world_size of the truth.
    raw_model = TinyGPT.from_config(model_cfg, context_length=ctx_len, attn_impl=attn_impl).to(device)
    total_param_count = raw_model.param_count()

    use_fsdp = world_size > 1 and train_cfg.parallelism == "fsdp"

    if use_fsdp:
        # FSDP's mechanism, unlike DDP's: parameters, gradients, AND optimizer state are
        # sharded across ranks from the start — each rank only ever permanently holds
        # 1/world_size of each. Right before a submodule's forward/backward needs its
        # full parameters, FSDP all-gathers the missing shards from every other rank,
        # uses them, then frees everything except this rank's own shard again. Strictly
        # more communication than DDP's one all-reduce per accumulation window, in
        # exchange for a peak per-device memory footprint that doesn't scale with the
        # model's full size — see docs/DISTRIBUTED_TRAINING.md for the full trade-off,
        # and Chapter 26 for the first-principles version.
        #
        # device_id is passed explicitly (not left to FSDP's auto-detection) for the
        # same reason the sibling custom-gpt-6m project's trainer_fsdp.py does: on
        # Apple Silicon, FSDP's device auto-detection falls back to
        # torch._C._get_accelerator(), which returns "mps" regardless of what device
        # this run actually uses, and torch.mps doesn't implement the full CUDA-like
        # interface FSDP expects from an accelerator backend — a real, reproduced crash
        # (AttributeError: Custom backend 'mps' not implement 'torch.mps.current_device')
        # on that project's own dev machine. Passing device_id explicitly skips that
        # auto-detection entirely, on every platform.
        device_id = device if device.type == "cuda" else torch.device("cpu")
        training_model = FSDP(raw_model, device_id=device_id)
    elif world_size > 1:
        # DDP's actual mechanism: broadcast rank 0's initial weights to every other rank
        # at construction time (every replica starts identical), then every backward()
        # call all-reduces (averages) gradients across ranks before optimizer.step() —
        # each rank ends up applying the same update, staying in sync without any
        # further explicit parameter synchronization.
        #
        # device_ids is CUDA-only — DDP raises if you pass it for a CPU module (the
        # local gloo/CPU smoke test hits this). None here means "infer from the
        # module's own device," which is correct for both: on CPU there's nothing to
        # pick between; on CUDA each rank's module already lives on its own
        # cuda:{local_rank} (see cli/train.py's device construction), so DDP infers
        # the right one from that alone.
        device_ids = [local_rank] if device.type == "cuda" else None
        training_model = DDP(raw_model, device_ids=device_ids)
    else:
        training_model = raw_model

    # GPT_OPTIMIZER=adamw8bit: quantizes AdamW's momentum/variance state to int8
    # (bitsandbytes), cutting that portion of the 16-bytes/param static memory total
    # (see docs/llm-engineering/26_distributed_training_ddp_and_fsdp.md's static-memory
    # table) roughly 4x. Opt-in, not default — bitsandbytes is an optional dependency
    # (`uv sync --extra 8bit`), and its quantized-optimizer kernel is CUDA/CPU only:
    # NOT implemented for MPS (verified directly — raises
    # `bitsandbytes::optimizer_update_8bit_blockwise ... not currently implemented for
    # the MPS device`; tracked upstream at pytorch/pytorch#141287). Fail loud here
    # rather than silently falling back to plain AdamW on MPS, same "don't silently
    # downgrade" convention as ATTN_IMPL/GPT_COMPILE elsewhere in this file.
    optimizer_choice = os.getenv("GPT_OPTIMIZER", "adamw")
    if optimizer_choice == "adamw8bit":
        if str(device) == "mps":
            raise RuntimeError(
                "GPT_OPTIMIZER=adamw8bit requires bitsandbytes' quantized-optimizer "
                "kernel, which has no MPS implementation (pytorch/pytorch#141287). "
                "Use device=cpu (for a mechanism check) or a CUDA GPU, or unset "
                "GPT_OPTIMIZER to train with plain AdamW on MPS."
            )
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            training_model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
    elif optimizer_choice == "adamw":
        optimizer = torch.optim.AdamW(
            training_model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
    else:
        raise ValueError(f"Unknown GPT_OPTIMIZER {optimizer_choice!r}. Use 'adamw' or 'adamw8bit'.")

    amp_device_type, amp_dtype = resolve_amp(train_cfg.precision, device)

    is_main = rank == 0
    if is_main:
        # total_param_count (captured before any wrap) is always the true total.
        # Also printing raw_model's own post-wrap count under FSDP is deliberate, not
        # a bug worth hiding — it's the same "DDP reports full, FSDP reports a shard"
        # distinction docs/DISTRIBUTED_TRAINING.md uses as the concrete, observable
        # proof the two mechanisms actually differ, not just a claim.
        mode = f"fsdp (this rank's shard: {raw_model.param_count():,})" if use_fsdp else "ddp" if world_size > 1 else "single-process"
        print(f"Model: {label}  |  {total_param_count:,} parameters total  |  device={device}  |  "
              f"attn_impl={attn_impl}  |  world_size={world_size}  |  parallelism={mode}")
        print(f"Train tokens: {len(train_tokens):,}  Test tokens: {len(test_tokens):,}")
        budget = train_cfg.steps * train_cfg.batch_size * ctx_len * world_size
        print(
            f"Precision: {amp_dtype if amp_dtype else 'fp32'}  |  "
            f"batch {train_cfg.batch_size} x accum {train_cfg.grad_accum_steps} x "
            f"world_size {world_size} = "
            f"{train_cfg.batch_size * train_cfg.grad_accum_steps * world_size} seqs/update"
        )
        # The token budget is implied by steps*batch_size*ctx_len*world_size — printed
        # explicitly (not silently rescaled) because `TrainConfig.steps` is NOT
        # world-size-aware here: running the same `steps` under a bigger world_size
        # consumes proportionally more total tokens, since every rank processes its
        # own full batch in parallel. See cli/train.py's module docstring.
        print(
            f"Budget: {train_cfg.steps:,} steps x {train_cfg.batch_size * ctx_len * world_size:,} tok "
            f"(world_size {world_size}) = "
            f"{budget / 1e9:.2f}B tokens ({budget / total_param_count:.1f} tok/param, "
            f"{budget / len(train_tokens):.2f} epochs)"
        )
        print(f"Checkpoints: {paths.checkpoint_dir}/")


    state = {
        "best_test_loss": float("inf"),
        "start_step": 0,
        "processed_tokens": 0,
        "total_training_seconds": 0.0,
    }

    if resume and paths.latest_checkpoint.exists():
        # Every rank loads the same checkpoint file independently from shared local
        # disk (single-node multi-GPU) — no coordination needed. This runs after
        # construction's rank-0-broadcast (DDP) or shard-init (FSDP) above, so it
        # simply overwrites that fresh-init state with the real resumed weights on
        # every rank identically; a little redundant work, not a correctness issue.
        _resume_into(state, raw_model, training_model, optimizer, paths, model_cfg,
                     ctx_len, device, use_fsdp=use_fsdp, is_main=is_main)

    # ETA from this run's own history. Only meaningful after a resume, where
    # `start_step` steps have demonstrably taken `total_training_seconds` — at step 0
    # there is no rate to extrapolate from yet, so it is simply omitted rather than
    # guessed. Note this is *training* time: a machine that sleeps or gets stopped
    # finishes later in wall-clock terms than this says.
    done_steps = state["start_step"]
    done_hours = state["total_training_seconds"] / 3600.0
    if is_main and done_steps > 0 and done_hours > 0:
        rate = done_steps / done_hours
        eta = format_eta(train_cfg.steps - done_steps, rate)
        if eta:
            hours, finish = eta
            print(
                f"Progress: step {done_steps:,}/{train_cfg.steps:,} "
                f"({100.0 * done_steps / train_cfg.steps:.1f}%)  |  "
                f"{rate:,.0f} steps/hr so far"
            )
            print(
                f"ETA: {hours:,.1f} more training-hours "
                f"({hours / 24:.1f} days) -> ~{finish} if run continuously"
            )

    return _run_loop(
        raw_model=raw_model,
        training_model=training_model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        train_tokens=train_tokens,
        test_tokens=test_tokens,
        ctx_len=ctx_len,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        paths=paths,
        label=label,
        device=device,
        state=state,
        rank=rank,
        world_size=world_size,
        amp_device_type=amp_device_type,
        amp_dtype=amp_dtype,
        use_fsdp=use_fsdp,
        total_param_count=total_param_count,
        attn_impl=attn_impl,
    )


def _resume_into(state, raw_model, training_model, optimizer, paths, model_cfg, ctx_len,
                 device, use_fsdp=False, is_main=True):
    """Restore weights/optimizer/progress from the latest checkpoint, if compatible.

    Every rank calls this independently against the same shared-disk checkpoint (see
    call site's comment), so `is_main` only gates printing, not the load itself — this
    matters even more under FSDP than DDP, since `optim_state_dict_to_load` below is a
    genuine collective operation: every rank must call it, with the identical full
    dict in hand, for FSDP to correctly work out which slice belongs to this rank.
    """
    checkpoint = None
    for candidate in (paths.latest_checkpoint, paths.best_checkpoint):
        if not candidate.exists():
            continue
        try:
            if is_main:
                print(f"Resuming from {candidate}...")
            checkpoint = torch.load(candidate, map_location=device)
            break
        except (RuntimeError, EOFError) as exc:
            if is_main:
                print(f"Warning: could not read {candidate}: {exc}")

    if checkpoint is None:
        if is_main:
            print("No readable checkpoint found — starting a fresh run.")
        return

    if not is_compatible(checkpoint, model_cfg, ctx_len):
        if is_main:
            print(
                "Warning: checkpoint architecture does not match the current config "
                f"(checkpoint embed={checkpoint.get('embed_size')} "
                f"layers={checkpoint.get('num_layers')} ctx={checkpoint.get('context_length')} "
                f"vs current embed={model_cfg.embed_size} layers={model_cfg.num_layers} "
                f"ctx={ctx_len}). Starting a fresh run."
            )
        return

    # No attn_impl remap here — unlike the sibling GPT-2-style projects, this model has
    # no attn_impl switch (SDPA is the only path; see model.py), so there is nothing to
    # remap between. `checkpoint["attn_impl"]` is a vestigial field from before the RoPE
    # rewrite (still written by checkpoint.make_payload's getattr(..., "naive") default);
    # this used to read the now-nonexistent `model.attn_impl` and crashed every resume.
    if use_fsdp:
        # Both calls are collective — every rank must reach them, with the same
        # checkpoint dict in hand (true here since every rank read the identical file
        # off shared disk above), for FSDP to correctly work out this rank's slice of
        # both the model weights and the optimizer state. rank0_only=False here
        # (unlike the save side's True) because loading, unlike gathering-to-save,
        # needs every rank actively participating with the full data, not just rank 0.
        with _fsdp_full_state_dict_ctx(training_model, rank0_only=False, device=device):
            training_model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                sharded_optim_state = FSDP.optim_state_dict_to_load(
                    training_model, optimizer, checkpoint["optimizer_state_dict"]
                )
                optimizer.load_state_dict(sharded_optim_state)
    else:
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    state["start_step"] = int(checkpoint.get("step", -1)) + 1
    state["best_test_loss"] = float(checkpoint.get("best_test_loss", float("inf")))
    state["processed_tokens"] = int(checkpoint.get("processed_tokens", 0))
    state["total_training_seconds"] = float(checkpoint.get("total_training_seconds", 0.0))
    if is_main:
        print(f"Resumed at step {state['start_step']} "
              f"(cumulative {format_duration(state['total_training_seconds'])})")


def _run_loop(raw_model, training_model, optimizer, tokenizer, train_tokens, test_tokens,
              ctx_len, model_cfg, train_cfg, paths, label, device, state,
              rank=0, world_size=1, amp_device_type=None, amp_dtype=None,
              use_fsdp=False, total_param_count=None, attn_impl="sdpa"):
    """Checkpoint save (`payload()` below) and the demo generation at the end use
    `raw_model` directly under DDP and single-process — DDP never touches parameter
    storage, so `raw_model` stays the complete, correct model throughout, and a saved
    checkpoint's `model_state_dict` never picks up DDP's `module.` prefix, keeping it
    loadable by the plain (non-DDP) inference code unchanged. Under FSDP, `raw_model`
    holds only this rank's shard once wrapped — `payload()` uses `_FSDPCheckpointView`
    instead, and the demo generation reloads a fresh, fully-materialized model from the
    just-written checkpoint (see below) rather than trying to run inference through
    `raw_model` directly. `training_model` (== `raw_model` itself when world_size == 1)
    is what forward/backward actually call, since that's the object DDP's/FSDP's hooks
    are registered on.

    Only rank 0 ("is_main") prints, shows the progress bar, runs eval, and writes
    checkpoints/eval-history — every rank running these redundantly would race to
    write the same files and spam duplicate output. The one thing every rank *does*
    do independently is the stop-file check (see below) and the training step itself
    (forward/backward/optimizer.step()) — DDP's gradient all-reduce is a collective
    operation and needs every rank present at every synced backward call, so nothing
    that could make ranks diverge in *which* step they're on is allowed to be
    rank-gated.
    """
    is_main = rank == 0
    run_start = time.time()

    def elapsed():
        return state["total_training_seconds"] + (time.time() - run_start)

    # Bug fix note: `state["total_training_seconds"]` must stay fixed at its
    # resume-time value for as long as elapsed() may still be called — mutating it
    # mid-function and then calling elapsed() again double-counts the current
    # session's duration (elapsed() would add (time.time() - run_start) on top of a
    # value that already includes that same delta). The final-save code below
    # therefore computes elapsed() exactly once into a local variable and passes it
    # explicitly to every payload() call from that point on, rather than mutating
    # `state` first and letting payload()'s default elapsed() recompute it.

    # Constructed once, outside the hot loop — see _FSDPCheckpointView/_FSDPOptimizerView
    # docstrings for why raw_model/optimizer alone aren't enough to checkpoint an
    # FSDP-wrapped run correctly. No-ops (unused) when use_fsdp is False.
    checkpoint_model = _FSDPCheckpointView(training_model, raw_model, total_param_count, device) if use_fsdp else raw_model
    checkpoint_optimizer = _FSDPOptimizerView(training_model, optimizer, device) if use_fsdp else optimizer

    def payload(step, total_training_seconds=None):
        return make_payload(
            model=checkpoint_model,
            optimizer=checkpoint_optimizer,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            context_length=ctx_len,
            step=step,
            best_test_loss=state["best_test_loss"],
            processed_tokens=state["processed_tokens"],
            # Mid-loop periodic saves want the live value (state["total_training_seconds"]
            # is still the fixed value loaded at resume, so elapsed() == correct-so-far).
            # The final save(s) after the loop pass an already-computed, frozen value
            # instead — see the fix note below `elapsed()`'s definition for why.
            total_training_seconds=(
                elapsed() if total_training_seconds is None else total_training_seconds
            ),
            label=label,
        )

    start_step = state["start_step"]
    progress = trange(train_cfg.steps, desc="training", unit="step", initial=start_step) if is_main else None
    optimizer.zero_grad(set_to_none=True)
    last_step = start_step - 1
    interrupted = False
    latest_metrics = None

    try:
        for step in range(start_step, train_cfg.steps):
            if paths.stop_file.exists():
                # Every rank checks (and unlinks) this independently, not just rank 0:
                # single-node multi-GPU means all ranks share the same local disk, so
                # the check is naturally consistent without coordination, and
                # `unlink(missing_ok=True)` makes the redundant deletes across ranks
                # harmless. This *cannot* be rank-0-only — if only rank 0 broke out of
                # the loop, other ranks would still call backward() on the next step
                # and hang forever waiting for rank 0's all-reduce participation that
                # never comes (see the module-level docstring above).
                paths.stop_file.unlink(missing_ok=True)
                if is_main:
                    print(f"\n{paths.stop_file} found — stopping gracefully...")
                interrupted = True
                break

            eval_now = step % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1
            if eval_now and (is_main or use_fsdp):
                # Under DDP/single-process, eval is genuinely rank-0-only: a plain
                # forward pass with no collective op, so other ranks safely skip
                # straight to their own training step below — the original reasoning
                # here, still correct for DDP. It is NOT correct for FSDP: every layer's
                # forward pass triggers an all-gather to reconstruct that layer's full
                # parameters, which — like any collective — needs every rank to call it
                # together, or the ranks that do call it hang (or, observed directly
                # while building this: corrupt with a storage-size RuntimeError)
                # waiting on peers that took the other branch entirely. So `eval_now`
                # is evaluated identically on every rank (it depends only on `step`,
                # which is already in lockstep across ranks), and the forward pass
                # itself runs on every rank under FSDP — only the bookkeeping that
                # follows (printing, CSV history, deciding *whether* to save) is still
                # rank-0-only, since state["best_test_loss"] is only meaningfully
                # tracked there.
                losses = estimate_loss(
                    training_model, train_tokens, test_tokens, ctx_len,
                    model_cfg.vocab_size, train_cfg, device,
                    amp=(amp_device_type, amp_dtype),
                )
                improved = is_main and losses["test"] < state["best_test_loss"]
                if improved:
                    state["best_test_loss"] = losses["test"]

                if use_fsdp:
                    # The state-dict gather itself is ALSO collective — every rank
                    # must call payload() together, same reasoning as the forward pass
                    # above. Gathering unconditionally here (rather than only when
                    # `improved`) avoids needing a separate broadcast of rank 0's
                    # improved/not-improved decision to every other rank before they'd
                    # know whether to enter the collective too.
                    gathered = payload(step)
                    if improved:
                        atomic_save(gathered, paths.best_checkpoint)
                        atomic_save(gathered, paths.serving_checkpoint)
                elif improved:
                    best = payload(step)
                    atomic_save(best, paths.best_checkpoint)
                    atomic_save(best, paths.serving_checkpoint)

                if is_main:
                    append_eval_history(paths.eval_history, {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "step": step,
                        "est_epoch": f"{state['processed_tokens'] / len(train_tokens):.6f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.8e}",
                        "train_loss": f"{losses['train']:.6f}",
                        "test_loss": f"{losses['test']:.6f}",
                        "test_perplexity": f"{safe_perplexity(losses['test']):.6f}",
                        "best_test_loss": f"{state['best_test_loss']:.6f}",
                        "improved": int(improved),
                        "processed_tokens": state["processed_tokens"],
                        "total_training_hours": f"{elapsed() / 3600.0:.4f}",
                    })
                    latest_metrics = {
                        "train_loss": f"{losses['train']:.4f}",
                        "test_loss": f"{losses['test']:.4f}",
                        "test_ppl": f"{safe_perplexity(losses['test']):.1f}",
                    }

            xb, yb = get_batch(train_tokens, ctx_len, train_cfg.batch_size, device)
            with torch.autocast(device_type=amp_device_type,
                                dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                loss = next_token_loss(training_model(xb), yb, model_cfg.vocab_size)

            # Computed BEFORE backward (unlike the pre-DDP version) because DDP's
            # no_sync() decision below needs to know, going in, whether this backward
            # call is the accumulation boundary — the all-reduce should fire on
            # exactly the boundary step's backward, not every micro-step's.
            is_accum_boundary = (step - start_step + 1) % train_cfg.grad_accum_steps == 0
            sync_now = is_accum_boundary or step == train_cfg.steps - 1

            # No GradScaler: bf16 keeps fp32's exponent range, so gradients cannot
            # underflow the way fp16's would. Weights/grads stay fp32 regardless.
            if world_size > 1 and not sync_now:
                # Non-boundary micro-step: accumulate gradients locally on this rank
                # only, skip the (expensive, and semantically premature) all-reduce
                # DDP would otherwise trigger on every single backward() call.
                with training_model.no_sync():
                    (loss / train_cfg.grad_accum_steps).backward()
            else:
                # Boundary step (or single-process, where no_sync() doesn't apply):
                # let DDP's hook fire normally — this is the one backward call per
                # accumulation window that actually averages gradients across ranks.
                (loss / train_cfg.grad_accum_steps).backward()

            if sync_now:
                for group in optimizer.param_groups:
                    group["lr"] = lr_for_step(step, train_cfg)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if is_main:
                postfix = {
                    "batch_loss": f"{loss.item():.4f}",
                    "est_epoch": f"{state['processed_tokens'] / len(train_tokens):.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "total_h": f"{elapsed() / 3600.0:.2f}",
                    "eta_h": (f"{(train_cfg.steps - step) / max(step / max(elapsed() / 3600.0, 1e-9), 1e-9):.1f}"
                              if step > 0 and elapsed() > 0 else "?"),
                }
                if latest_metrics:
                    postfix.update(latest_metrics)
                progress.set_postfix(**postfix)
                progress.update(1)

            # Global tokens processed this step, across all ranks — each rank
            # independently processes batch_size*ctx_len tokens, world_size of them
            # in parallel, so the *global* count (what est_epoch/Budget mean by
            # "tokens") scales by world_size even though train_cfg.steps doesn't
            # change. See train()'s Budget print for the same reasoning.
            state["processed_tokens"] += train_cfg.batch_size * ctx_len * world_size
            last_step = step

            if (step + 1) % train_cfg.save_every_steps == 0 and (is_main or use_fsdp):
                # Same collective-vs-local split as the eval block above: every rank
                # must reach payload()'s state-dict gather together under FSDP; only
                # rank 0 writes the result to disk.
                periodic_payload = payload(step)
                if is_main:
                    atomic_save(periodic_payload, paths.latest_checkpoint)
    except KeyboardInterrupt:
        interrupted = True
        if is_main:
            print("\nInterrupted — saving a resumable checkpoint...")
    finally:
        if progress is not None:
            progress.close()

    final_total_seconds = elapsed()  # single source of truth from here on — see note above
    state["total_training_seconds"] = final_total_seconds
    final_step = max(last_step, start_step - 1)

    # Under DDP/single-process, non-main ranks can simply return here — payload()'s
    # state_dict() call is purely local, so rank 0 alone calling it below is fine.
    # Under FSDP that would hang or corrupt: payload()'s gather is collective, so
    # every rank must reach BOTH calls below together, even though only rank 0 goes
    # on to actually write files, print, or generate the demo completion. This is the
    # same collective-vs-local split as the eval and periodic-save blocks above,
    # just applied to the two end-of-run saves instead.
    final_payload = payload(final_step, total_training_seconds=final_total_seconds)
    if is_main:
        atomic_save(final_payload, paths.latest_checkpoint)

    if interrupted:
        if is_main:
            print(f"Saved: {paths.latest_checkpoint}")
            print(f"Cumulative training time: {format_duration(state['total_training_seconds'])}")
            print("Resume with: make train")
        return {"interrupted": True, "step": final_step,
                "best_test_loss": state["best_test_loss"]}

    completed = payload(max(final_step, train_cfg.steps - 1), total_training_seconds=final_total_seconds)

    if not is_main:
        # Non-main ranks never ran eval, so their local state["best_test_loss"] was
        # never updated off its float("inf") initial value — only rank 0's return
        # value/checkpoints are authoritative; callers should not inspect a non-main
        # rank's returned dict for anything but step/interrupted. They've now done
        # their part of both collective gathers above, so it's safe to stop here.
        return {"interrupted": False, "step": final_step,
                "best_test_loss": state["best_test_loss"]}

    atomic_save(completed, paths.final_checkpoint)
    atomic_save(completed, paths.latest_checkpoint)
    if not paths.serving_checkpoint.exists():
        atomic_save(completed, paths.serving_checkpoint)

    print(f"Saved final checkpoint: {paths.final_checkpoint}")
    print(f"Serving checkpoint: {paths.serving_checkpoint}")
    print(f"Eval history: {paths.eval_history}")
    print(f"Cumulative training time: {format_duration(state['total_training_seconds'])}")

    # Under FSDP, raw_model's own parameter storage holds only this rank's shard once
    # wrapped (same reason checkpointing needs _FSDPCheckpointView above) — it cannot
    # run a normal forward pass on its own. `completed["model_state_dict"]` is already
    # the full, gathered state dict (that's what _FSDPCheckpointView.state_dict()
    # returned into payload() above), so building one fresh, plain TinyGPT and loading
    # it there is simpler and cheaper than trying to run generation through the FSDP
    # wrapper itself for a handful of demo tokens.
    demo_model = raw_model
    if use_fsdp:
        demo_model = TinyGPT.from_config(model_cfg, context_length=ctx_len, attn_impl=attn_impl).to(device)
        demo_model.load_state_dict(completed["model_state_dict"])

    _, completion = generate_text(
        model=demo_model,
        tokenizer=tokenizer,
        prompt=train_cfg.demo_prompt,
        context_length=ctx_len,
        max_new_tokens=train_cfg.max_new_tokens,
        device=device,
        do_sample=True,
        temperature=0.9,
        top_k=40,
        top_p=0.95,
        postprocess=False,
    )
    print(f"\nSample continuation of {train_cfg.demo_prompt!r}:")
    print(completion if completion.strip() else "[empty completion]")

    return {"interrupted": False, "step": final_step,
            "best_test_loss": state["best_test_loss"]}
