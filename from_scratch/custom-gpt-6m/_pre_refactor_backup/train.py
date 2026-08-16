"""
Training loop for the TinyStories GPT.

Run `python prepare_dataset.py` first. See docs/TRAINING.md for the full explanation of
every hyperparameter and mechanism here — this file assumes that doc as context and keeps
inline comments minimal.
"""
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from model import build_model, detect_device

# -------- CONFIG --------
data_dir = Path(os.getenv("DATA_DIR", "data"))
context_length = int(os.getenv("CONTEXT_LENGTH", 256))
embed_size = int(os.getenv("EMBED_SIZE", 256))
num_heads = int(os.getenv("NUM_HEADS", 8))
num_layers = int(os.getenv("NUM_LAYERS", 6))
dropout = float(os.getenv("DROPOUT", 0.1))

batch_size = int(os.getenv("BATCH_SIZE", 32))
grad_accum_steps = int(os.getenv("GRAD_ACCUM_STEPS", 1))
attn_impl = os.getenv("ATTN_IMPL", "naive")
use_amp = os.getenv("AMP", "0") == "1"
use_grad_checkpoint = os.getenv("GRAD_CHECKPOINT", "0") == "1"
lr = float(os.getenv("LR", 3e-4))
min_lr = float(os.getenv("MIN_LR", 3e-5))
steps = int(os.getenv("STEPS", 5000))
eval_interval = int(os.getenv("EVAL_INTERVAL", 250))
eval_batches = int(os.getenv("EVAL_BATCHES", 20))
save_every_steps = int(os.getenv("SAVE_EVERY_STEPS", 500))
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 120))
resume_training = os.getenv("RESUME_TRAINING", "1") == "1"

artifact_root = Path(".")
checkpoint_path = artifact_root / "tinystories_gpt_checkpoint.pt"
latest_checkpoint_path = artifact_root / "tinystories_gpt_checkpoint_latest.pt"
best_checkpoint_path = artifact_root / "tinystories_gpt_checkpoint_best.pt"
eval_history_path = artifact_root / "logs" / "train_eval_history.csv"

device = detect_device()
if device == "cuda":
    torch.set_float32_matmul_precision("high")
print(f"[device] using {device}")

# Mixed precision: real fp16 autocast + GradScaler on CUDA, bf16 autocast (no scaler
# needed — bf16 has fp32's exponent range, just less mantissa, so it doesn't underflow
# the way fp16 does) on MPS, and a documented no-op on CPU. See docs/EFFICIENT_TRAINING.md
# for why the mechanism differs per device instead of being one code path.
amp_dtype = {"cuda": torch.float16, "mps": torch.bfloat16, "cpu": torch.bfloat16}[device]
amp_enabled = use_amp and device != "cpu"
if use_amp and device == "cpu":
    print("[amp] AMP=1 requested but device=cpu — autocast has no meaningful effect "
          "without a GPU/MPS accelerator; running in full fp32 instead.")
use_scaler = amp_enabled and device == "cuda"
scaler = torch.amp.GradScaler(enabled=use_scaler)
print(f"[amp] enabled={amp_enabled} dtype={amp_dtype if amp_enabled else 'fp32'} "
      f"grad_scaler={use_scaler}")
print(f"[attn] impl={attn_impl}")
print(f"[grad_checkpoint] enabled={use_grad_checkpoint}")


def load_meta():
    with open(data_dir / "meta.json") as f:
        return json.load(f)


def load_tokens(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def get_batch(tokens, ctx_len, bsz, device):
    max_start = len(tokens) - ctx_len - 1
    ix = torch.randint(0, max_start, (bsz,))
    x = torch.stack([tokens[i:i + ctx_len] for i in ix]).to(device)
    y = torch.stack([tokens[i + 1:i + ctx_len + 1] for i in ix]).to(device)
    return x, y


def lr_for_step(step_idx):
    warmup_steps = max(100, int(steps * 0.02))
    if step_idx < warmup_steps:
        return lr * float(step_idx + 1) / float(warmup_steps)
    decay_steps = max(1, steps - warmup_steps)
    progress = min(1.0, max(0.0, (step_idx - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())
    return min_lr + (lr - min_lr) * cosine


def safe_perplexity(loss_value):
    return float(torch.exp(torch.tensor(min(float(loss_value), 20.0))).item())


def append_eval_history(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def make_checkpoint_payload(model, optimizer, step, best_val_loss, meta, processed_tokens):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "processed_tokens": processed_tokens,
        "vocab_size": meta["vocab_size"],
        "context_length": context_length,
        "embed_size": embed_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "architecture": "tinystories_gpt_decoder_pre_norm_weight_tied",
        "tokenizer_path": str(data_dir / "tokenizer.json"),
    }


@torch.no_grad()
def estimate_loss(model, train_tokens, val_tokens, ctx_len, bsz, device):
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


def main():
    meta = load_meta()
    print(f"[data] meta: {meta}")

    train_tokens = load_tokens(data_dir / "train.bin")
    val_tokens = load_tokens(data_dir / "val.bin")
    print(f"[data] train_tokens={len(train_tokens):,} val_tokens={len(val_tokens):,}")

    model = build_model(
        vocab_size=meta["vocab_size"],
        context_length=context_length,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        attn_impl=attn_impl,
        grad_checkpoint=use_grad_checkpoint,
    ).to(device)
    print(f"[model] {model.num_parameters():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    best_val_loss = float("inf")
    start_step = 0
    processed_tokens = 0

    if resume_training and latest_checkpoint_path.exists():
        ckpt = torch.load(latest_checkpoint_path, map_location=device)
        compatible = (
            ckpt.get("embed_size") == embed_size
            and ckpt.get("num_heads") == num_heads
            and ckpt.get("num_layers") == num_layers
            and ckpt.get("context_length") == context_length
            and ckpt.get("vocab_size") == meta["vocab_size"]
        )
        if compatible:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", -1) + 1
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            processed_tokens = ckpt.get("processed_tokens", 0)
            print(f"[resume] resumed from step {start_step}")
        else:
            print("[resume] checkpoint config mismatch, starting fresh")

    progress = trange(start_step, steps, desc="training", unit="step")
    optimizer.zero_grad(set_to_none=True)
    last_step = start_step - 1
    latest_eval = None

    try:
        for step in range(start_step, steps):
            if step % eval_interval == 0 or step == steps - 1:
                losses = estimate_loss(model, train_tokens, val_tokens, context_length, batch_size, device)
                improved = losses["val"] < best_val_loss
                if improved:
                    best_val_loss = losses["val"]
                    payload = make_checkpoint_payload(model, optimizer, step, best_val_loss, meta, processed_tokens)
                    torch.save(payload, best_checkpoint_path)
                    torch.save(payload, checkpoint_path)
                append_eval_history(eval_history_path, {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": step,
                    "train_loss": f"{losses['train']:.4f}",
                    "val_loss": f"{losses['val']:.4f}",
                    "val_perplexity": f"{safe_perplexity(losses['val']):.2f}",
                    "best_val_loss": f"{best_val_loss:.4f}",
                    "improved": int(improved),
                })
                latest_eval = {"train": f"{losses['train']:.3f}", "val": f"{losses['val']:.3f}"}

            xb, yb = get_batch(train_tokens, context_length, batch_size, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            scaler.scale(loss / grad_accum_steps).backward()

            if (step - start_step + 1) % grad_accum_steps == 0 or step == steps - 1:
                current_lr = lr_for_step(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            postfix = {"loss": f"{loss.item():.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
            if latest_eval:
                postfix.update(latest_eval)
            progress.set_postfix(**postfix)
            progress.update(1)
            processed_tokens += batch_size * context_length
            last_step = step

            if (step + 1) % save_every_steps == 0:
                payload = make_checkpoint_payload(model, optimizer, step, best_val_loss, meta, processed_tokens)
                torch.save(payload, latest_checkpoint_path)
    except KeyboardInterrupt:
        print("\n[interrupt] saving latest checkpoint...")
    finally:
        progress.close()

    payload = make_checkpoint_payload(model, optimizer, last_step, best_val_loss, meta, processed_tokens)
    torch.save(payload, latest_checkpoint_path)
    if not checkpoint_path.exists():
        torch.save(payload, checkpoint_path)
    print(f"[done] latest checkpoint: {latest_checkpoint_path}")
    print(f"[done] best checkpoint: {checkpoint_path} (best_val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
