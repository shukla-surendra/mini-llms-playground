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

import torch
from tqdm import trange

from ..checkpoint import atomic_save, is_compatible, make_payload, remap_attn_impl
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


def train(model_cfg, train_cfg, paths, label, resume=True, device=None):
    device = device or get_device()
    torch.manual_seed(train_cfg.seed)

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
    model = TinyGPT.from_config(model_cfg, context_length=ctx_len, attn_impl=attn_impl).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    amp_device_type, amp_dtype = resolve_amp(train_cfg.precision, device)

    tokens_per_step = train_cfg.batch_size * ctx_len
    print(f"Model: {label}  |  {model.param_count():,} parameters  |  device={device}  |  attn_impl={attn_impl}")
    print(f"Train tokens: {len(train_tokens):,}  Test tokens: {len(test_tokens):,}")
    print(
        f"Precision: {amp_dtype if amp_dtype else 'fp32'}  |  "
        f"batch {train_cfg.batch_size} x accum {train_cfg.grad_accum_steps} = "
        f"{train_cfg.batch_size * train_cfg.grad_accum_steps} seqs/update"
    )
    # The token budget is implied by steps*batch_size*ctx_len, so print it rather than
    # let a changed batch_size silently rescale the run (see TrainConfig's docstring).
    budget = train_cfg.steps * tokens_per_step
    print(
        f"Budget: {train_cfg.steps:,} steps x {tokens_per_step:,} tok = {budget / 1e9:.2f}B tokens "
        f"({budget / model.param_count():.1f} tok/param, {budget / len(train_tokens):.2f} epochs)"
    )
    print(f"Checkpoints: {paths.checkpoint_dir}/")


    state = {
        "best_test_loss": float("inf"),
        "start_step": 0,
        "processed_tokens": 0,
        "total_training_seconds": 0.0,
    }

    if resume and paths.latest_checkpoint.exists():
        _resume_into(state, model, optimizer, paths, model_cfg, ctx_len, device)

    # ETA from this run's own history. Only meaningful after a resume, where
    # `start_step` steps have demonstrably taken `total_training_seconds` — at step 0
    # there is no rate to extrapolate from yet, so it is simply omitted rather than
    # guessed. Note this is *training* time: a machine that sleeps or gets stopped
    # finishes later in wall-clock terms than this says.
    done_steps = state["start_step"]
    done_hours = state["total_training_seconds"] / 3600.0
    if done_steps > 0 and done_hours > 0:
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
        model=model,
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
        amp_device_type=amp_device_type,
        amp_dtype=amp_dtype,
    )


def _resume_into(state, model, optimizer, paths, model_cfg, ctx_len, device):
    """Restore weights/optimizer/progress from the latest checkpoint, if compatible."""
    checkpoint = None
    for candidate in (paths.latest_checkpoint, paths.best_checkpoint):
        if not candidate.exists():
            continue
        try:
            print(f"Resuming from {candidate}...")
            checkpoint = torch.load(candidate, map_location=device)
            break
        except (RuntimeError, EOFError) as exc:
            print(f"Warning: could not read {candidate}: {exc}")

    if checkpoint is None:
        print("No readable checkpoint found — starting a fresh run.")
        return

    if not is_compatible(checkpoint, model_cfg, ctx_len):
        print(
            "Warning: checkpoint architecture does not match the current config "
            f"(checkpoint embed={checkpoint.get('embed_size')} "
            f"layers={checkpoint.get('num_layers')} ctx={checkpoint.get('context_length')} "
            f"vs current embed={model_cfg.embed_size} layers={model_cfg.num_layers} "
            f"ctx={ctx_len}). Starting a fresh run."
        )
        return

    checkpoint_attn_impl = checkpoint.get("attn_impl", "naive")
    model_state_dict = checkpoint["model_state_dict"]
    if checkpoint_attn_impl != model.attn_impl:
        print(
            f"Checkpoint was trained with attn_impl={checkpoint_attn_impl!r}, current "
            f"run uses attn_impl={model.attn_impl!r} — remapping attention weights "
            f"(same values, different parameter names; see checkpoint.remap_attn_impl)."
        )
        model_state_dict = remap_attn_impl(
            model_state_dict, num_layers=model_cfg.num_layers,
            from_impl=checkpoint_attn_impl, to_impl=model.attn_impl,
        )
    model.load_state_dict(model_state_dict)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    state["start_step"] = int(checkpoint.get("step", -1)) + 1
    state["best_test_loss"] = float(checkpoint.get("best_test_loss", float("inf")))
    state["processed_tokens"] = int(checkpoint.get("processed_tokens", 0))
    state["total_training_seconds"] = float(checkpoint.get("total_training_seconds", 0.0))
    print(f"Resumed at step {state['start_step']} "
          f"(cumulative {format_duration(state['total_training_seconds'])})")


def _run_loop(model, optimizer, tokenizer, train_tokens, test_tokens, ctx_len,
              model_cfg, train_cfg, paths, label, device, state,
              amp_device_type=None, amp_dtype=None):
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

    def payload(step, total_training_seconds=None):
        return make_payload(
            model=model,
            optimizer=optimizer,
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
    progress = trange(train_cfg.steps, desc="training", unit="step", initial=start_step)
    optimizer.zero_grad(set_to_none=True)
    last_step = start_step - 1
    interrupted = False
    latest_metrics = None

    try:
        for step in range(start_step, train_cfg.steps):
            if paths.stop_file.exists():
                # Fallback for when SIGINT doesn't get through (see Paths.stop_file's
                # docstring) — checked every step, so honored within one step's time
                # rather than hanging indefinitely like an unreceived signal would.
                paths.stop_file.unlink(missing_ok=True)
                print(f"\n{paths.stop_file} found — stopping gracefully...")
                interrupted = True
                break

            if step % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1:
                losses = estimate_loss(
                    model, train_tokens, test_tokens, ctx_len,
                    model_cfg.vocab_size, train_cfg, device,
                    amp=(amp_device_type, amp_dtype),
                )
                improved = losses["test"] < state["best_test_loss"]
                if improved:
                    state["best_test_loss"] = losses["test"]
                    best = payload(step)
                    atomic_save(best, paths.best_checkpoint)
                    atomic_save(best, paths.serving_checkpoint)

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
                loss = next_token_loss(model(xb), yb, model_cfg.vocab_size)
            # No GradScaler: bf16 keeps fp32's exponent range, so gradients cannot
            # underflow the way fp16's would. Weights/grads stay fp32 regardless —
            # autocast only changes the dtype of the ops inside the block.
            (loss / train_cfg.grad_accum_steps).backward()

            is_accum_boundary = (step - start_step + 1) % train_cfg.grad_accum_steps == 0
            if is_accum_boundary or step == train_cfg.steps - 1:
                for group in optimizer.param_groups:
                    group["lr"] = lr_for_step(step, train_cfg)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

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

            state["processed_tokens"] += train_cfg.batch_size * ctx_len
            last_step = step

            if (step + 1) % train_cfg.save_every_steps == 0:
                atomic_save(payload(step), paths.latest_checkpoint)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — saving a resumable checkpoint...")
    finally:
        progress.close()

    final_total_seconds = elapsed()  # single source of truth from here on — see note above
    state["total_training_seconds"] = final_total_seconds
    final_step = max(last_step, start_step - 1)
    atomic_save(payload(final_step, total_training_seconds=final_total_seconds), paths.latest_checkpoint)

    if interrupted:
        print(f"Saved: {paths.latest_checkpoint}")
        print(f"Cumulative training time: {format_duration(state['total_training_seconds'])}")
        print("Resume with: make train")
        return {"interrupted": True, "step": final_step,
                "best_test_loss": state["best_test_loss"]}

    completed = payload(max(final_step, train_cfg.steps - 1), total_training_seconds=final_total_seconds)
    atomic_save(completed, paths.final_checkpoint)
    atomic_save(completed, paths.latest_checkpoint)
    if not paths.serving_checkpoint.exists():
        atomic_save(completed, paths.serving_checkpoint)

    print(f"Saved final checkpoint: {paths.final_checkpoint}")
    print(f"Serving checkpoint: {paths.serving_checkpoint}")
    print(f"Eval history: {paths.eval_history}")
    print(f"Cumulative training time: {format_duration(state['total_training_seconds'])}")

    _, completion = generate_text(
        model=model,
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
