"""
Training loop for the causal-LM TinyStories GPT. See docs/TRAINING.md for the full
explanation of every hyperparameter and mechanism here — this module assumes that doc as
context and keeps inline comments minimal.

Run `gpt-data` first to build data/{meta.json,tokenizer.json,train.bin,val.bin}.
"""
import csv
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from ..checkpoint import is_compatible, load_checkpoint, make_payload
from ..config import load_settings, resolve_vocab_size
from ..model import build_model, detect_device

ATTN_IMPL = os.getenv("ATTN_IMPL", "naive")
USE_AMP = os.getenv("AMP", "0") == "1"
USE_GRAD_CHECKPOINT = os.getenv("GRAD_CHECKPOINT", "0") == "1"


def load_meta(paths):
    import json
    with open(paths.meta_json) as f:
        return json.load(f)


def load_tokens(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def get_batch(tokens, ctx_len, bsz, device):
    max_start = len(tokens) - ctx_len - 1
    ix = torch.randint(0, max_start, (bsz,))
    x = torch.stack([tokens[i:i + ctx_len] for i in ix]).to(device)
    y = torch.stack([tokens[i + 1:i + ctx_len + 1] for i in ix]).to(device)
    return x, y


def lr_for_step(step_idx, steps, lr, min_lr):
    warmup_steps = max(100, int(steps * 0.02))
    if step_idx < warmup_steps:
        return lr * float(step_idx + 1) / float(warmup_steps)
    decay_steps = max(1, steps - warmup_steps)
    progress = min(1.0, max(0.0, (step_idx - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())
    return min_lr + (lr - min_lr) * cosine


def append_eval_history(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def estimate_loss(model, train_tokens, val_tokens, ctx_len, bsz, device, amp_dtype, amp_enabled, eval_batches):
    model.eval()
    out = {}
    for name, tokens in (("train", train_tokens), ("val", val_tokens)):
        losses = []
        for _ in range(eval_batches):
            xb, yb = get_batch(tokens, ctx_len, bsz, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def run(preset_name=None, resume=True):
    model_cfg, train_cfg, paths, label = load_settings(preset_name, objective="causal")

    device = detect_device()
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"[device] using {device}")

    # Mixed precision: real fp16 autocast + GradScaler on CUDA, bf16 autocast (no scaler
    # needed — bf16 has fp32's exponent range, just less mantissa, so it doesn't underflow
    # the way fp16 does) on MPS, and a documented no-op on CPU. See docs/EFFICIENT_TRAINING.md
    # for why the mechanism differs per device instead of being one code path.
    amp_dtype = {"cuda": torch.float16, "mps": torch.bfloat16, "cpu": torch.bfloat16}[device]
    amp_enabled = USE_AMP and device != "cpu"
    if USE_AMP and device == "cpu":
        print("[amp] AMP=1 requested but device=cpu — autocast has no meaningful effect "
              "without a GPU/MPS accelerator; running in full fp32 instead.")
    use_scaler = amp_enabled and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    print(f"[amp] enabled={amp_enabled} dtype={amp_dtype if amp_enabled else 'fp32'} "
          f"grad_scaler={use_scaler}")
    print(f"[attn] impl={ATTN_IMPL}")
    print(f"[grad_checkpoint] enabled={USE_GRAD_CHECKPOINT}")

    meta = load_meta(paths)
    print(f"[data] meta: {meta}")
    model_cfg = resolve_vocab_size(model_cfg, meta)

    train_tokens = load_tokens(paths.train_bin)
    val_tokens = load_tokens(paths.val_bin)
    print(f"[data] train_tokens={len(train_tokens):,} val_tokens={len(val_tokens):,}")

    model = build_model(
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        embed_size=model_cfg.embed_size,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        attn_impl=ATTN_IMPL,
        grad_checkpoint=USE_GRAD_CHECKPOINT,
    ).to(device)
    print(f"[model] {model.num_parameters():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    best_val_loss = float("inf")
    start_step = 0
    processed_tokens = 0

    if resume and paths.latest_checkpoint.exists():
        ckpt = load_checkpoint(paths.latest_checkpoint, device)
        if is_compatible(ckpt, model_cfg):
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", -1) + 1
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            processed_tokens = ckpt.get("processed_tokens", 0)
            print(f"[resume] resumed from step {start_step}")
        else:
            print("[resume] checkpoint config mismatch, starting fresh")

    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress = trange(start_step, train_cfg.steps, desc="training", unit="step")
    optimizer.zero_grad(set_to_none=True)
    last_step = start_step - 1
    latest_eval = None

    def payload(step, best):
        extra = {"processed_tokens": processed_tokens, "tokenizer_path": str(paths.tokenizer_json)}
        return make_payload(model, optimizer, step, best, model_cfg, extra_fields=extra)

    try:
        for step in range(start_step, train_cfg.steps):
            if step % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1:
                losses = estimate_loss(model, train_tokens, val_tokens, model_cfg.context_length,
                                        train_cfg.batch_size, device, amp_dtype, amp_enabled,
                                        train_cfg.eval_batches)
                improved = losses["val"] < best_val_loss
                if improved:
                    best_val_loss = losses["val"]
                    torch.save(payload(step, best_val_loss), paths.best_checkpoint)
                append_eval_history(paths.eval_history, {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": step,
                    "train_loss": f"{losses['train']:.4f}",
                    "val_loss": f"{losses['val']:.4f}",
                    "best_val_loss": f"{best_val_loss:.4f}",
                    "improved": int(improved),
                })
                latest_eval = {"train": f"{losses['train']:.3f}", "val": f"{losses['val']:.3f}"}

            xb, yb = get_batch(train_tokens, model_cfg.context_length, train_cfg.batch_size, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            scaler.scale(loss / train_cfg.grad_accum_steps).backward()

            if (step - start_step + 1) % train_cfg.grad_accum_steps == 0 or step == train_cfg.steps - 1:
                current_lr = lr_for_step(step, train_cfg.steps, train_cfg.lr, train_cfg.min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            postfix = {"loss": f"{loss.item():.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
            if latest_eval:
                postfix.update(latest_eval)
            progress.set_postfix(**postfix)
            progress.update(1)
            processed_tokens += train_cfg.batch_size * model_cfg.context_length
            last_step = step

            if (step + 1) % train_cfg.save_every_steps == 0:
                torch.save(payload(step, best_val_loss), paths.latest_checkpoint)
    except KeyboardInterrupt:
        print("\n[interrupt] saving latest checkpoint...")
    finally:
        progress.close()

    final_payload = payload(last_step, best_val_loss)
    torch.save(final_payload, paths.latest_checkpoint)
    if not paths.best_checkpoint.exists():
        torch.save(final_payload, paths.best_checkpoint)
    torch.save(final_payload, paths.final_checkpoint)
    print(f"[done] latest checkpoint: {paths.latest_checkpoint}")
    print(f"[done] best checkpoint: {paths.best_checkpoint} (best_val_loss={best_val_loss:.4f})")
