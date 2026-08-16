"""
Contrastive self-supervised training loop — same conventions as train.py/train_mlm.py
(env-var config, checkpoint save/resume, eval-history CSV, cosine LR with warmup) applied
to a third pretraining objective. See docs/CONTRASTIVE_LEARNING.md for the SimCSE
positive-pair mechanism and the InfoNCE loss.

Run `python prepare_dataset.py` first (same tokenized data/tokenizer as the other two
training scripts in this project).
"""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from model import detect_device
from model_contrastive import build_contrastive_model, info_nce_loss

# -------- CONFIG --------
data_dir = Path(os.getenv("DATA_DIR", "data"))
context_length = int(os.getenv("CONTEXT_LENGTH", 256))
embed_size = int(os.getenv("EMBED_SIZE", 256))
num_heads = int(os.getenv("NUM_HEADS", 8))
num_layers = int(os.getenv("NUM_LAYERS", 6))
dropout = float(os.getenv("DROPOUT", 0.1))
proj_dim = int(os.getenv("PROJ_DIM", 128))
temperature = float(os.getenv("TEMPERATURE", 0.05))
attn_impl = os.getenv("ATTN_IMPL", "naive")

batch_size = int(os.getenv("BATCH_SIZE", 32))
lr = float(os.getenv("LR", 3e-4))
min_lr = float(os.getenv("MIN_LR", 3e-5))
steps = int(os.getenv("STEPS", 5000))
eval_interval = int(os.getenv("EVAL_INTERVAL", 250))
eval_batches = int(os.getenv("EVAL_BATCHES", 20))
save_every_steps = int(os.getenv("SAVE_EVERY_STEPS", 500))
resume_training = os.getenv("RESUME_TRAINING", "1") == "1"

artifact_root = Path(".")
checkpoint_path = artifact_root / "tinystories_contrastive_checkpoint.pt"
latest_checkpoint_path = artifact_root / "tinystories_contrastive_checkpoint_latest.pt"
best_checkpoint_path = artifact_root / "tinystories_contrastive_checkpoint_best.pt"
eval_history_path = artifact_root / "logs" / "train_contrastive_eval_history.csv"

device = detect_device()
if device == "cuda":
    torch.set_float32_matmul_precision("high")
print(f"[device] using {device}")
print(f"[contrastive] proj_dim={proj_dim} temperature={temperature}")

# batch_size directly determines the number of in-batch negatives (batch_size - 1 per
# anchor) — worth noting since, unlike the causal-LM/MLM scripts, batch_size here isn't
# just a memory/throughput knob, it's part of the objective's difficulty.


def load_meta():
    with open(data_dir / "meta.json") as f:
        return json.load(f)


def load_tokens(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def get_window(tokens, ctx_len, bsz, device):
    max_start = len(tokens) - ctx_len
    ix = torch.randint(0, max_start, (bsz,))
    return torch.stack([tokens[i:i + ctx_len] for i in ix]).to(device)


def lr_for_step(step_idx):
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


def make_checkpoint_payload(model, optimizer, step, best_val_loss, meta):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "vocab_size": meta["vocab_size"],
        "context_length": context_length,
        "embed_size": embed_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "proj_dim": proj_dim,
        "temperature": temperature,
        "architecture": "tinystories_contrastive_simcse_infonce",
        "tokenizer_path": str(data_dir / "tokenizer.json"),
    }


def contrastive_step(model, tokens, ctx_len, bsz, device):
    window = get_window(tokens, ctx_len, bsz, device)
    # Same input, two independent forward passes — model.training controls whether
    # dropout is actually stochastic (see estimate_loss, which deliberately keeps
    # training-mode dropout on despite otherwise looking like an eval function — see
    # docs/CONTRASTIVE_LEARNING.md's gotcha section for why).
    z1 = model(window)
    z2 = model(window)
    return info_nce_loss(z1, z2, temperature)


def estimate_loss(model, train_tokens, val_tokens, ctx_len, bsz, device):
    # Deliberately does NOT call model.eval() — the contrastive objective's positive pairs
    # only exist because dropout is stochastic; switching to eval mode would make z1 and
    # z2 identical (accuracy trivially 1.0, an uninformative eval signal). This is the one
    # meaningful difference from train.py/train_mlm.py's estimate_loss.
    #
    # Also deliberately NOT wrapped in @torch.no_grad() (unlike every other estimate_loss
    # in this project) — on MPS specifically, nn.MultiheadAttention's dropout_p>0 routes
    # into a fast inference path built on F.scaled_dot_product_attention as soon as grad
    # tracking is off, and that MPS kernel raises `NotImplementedError:
    # scaled_dot_product_attention for MPS does not support dropout` — a real error hit
    # running this exact function during development, not a hypothetical. Since nothing
    # here calls .backward(), the unused autograd graph this leaves behind is immediately
    # discarded — a small, acceptable memory cost at this project's scale, not a
    # correctness issue.
    with torch.no_grad() if device != "mps" else torch.enable_grad():
        out_loss, out_acc = {}, {}
        for name, tokens in (("train", train_tokens), ("val", val_tokens)):
            losses, accs = [], []
            for _ in range(eval_batches):
                loss, acc = contrastive_step(model, tokens, ctx_len, bsz, device)
                losses.append(loss.item())
                accs.append(acc)
            out_loss[name] = sum(losses) / len(losses)
            out_acc[name] = sum(accs) / len(accs)
        return out_loss, out_acc


def main():
    meta = load_meta()
    print(f"[data] meta: {meta}")

    train_tokens = load_tokens(data_dir / "train.bin")
    val_tokens = load_tokens(data_dir / "val.bin")
    print(f"[data] train_tokens={len(train_tokens):,} val_tokens={len(val_tokens):,}")

    model = build_contrastive_model(
        vocab_size=meta["vocab_size"],
        context_length=context_length,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        proj_dim=proj_dim,
        attn_impl=attn_impl,
    ).to(device)
    model.train()  # stays in train mode throughout — see estimate_loss's note above
    print(f"[model] {model.num_parameters():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    best_val_loss = float("inf")
    start_step = 0

    if resume_training and latest_checkpoint_path.exists():
        ckpt = torch.load(latest_checkpoint_path, map_location=device)
        compatible = (
            ckpt.get("embed_size") == embed_size
            and ckpt.get("num_heads") == num_heads
            and ckpt.get("num_layers") == num_layers
            and ckpt.get("context_length") == context_length
            and ckpt.get("vocab_size") == meta["vocab_size"]
            and ckpt.get("proj_dim") == proj_dim
        )
        if compatible:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", -1) + 1
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            print(f"[resume] resumed from step {start_step}")
        else:
            print("[resume] checkpoint config mismatch, starting fresh")

    progress = trange(start_step, steps, desc="training", unit="step")
    last_step = start_step - 1
    latest_eval = None

    try:
        for step in range(start_step, steps):
            if step % eval_interval == 0 or step == steps - 1:
                losses, accs = estimate_loss(model, train_tokens, val_tokens, context_length, batch_size, device)
                improved = losses["val"] < best_val_loss
                if improved:
                    best_val_loss = losses["val"]
                    payload = make_checkpoint_payload(model, optimizer, step, best_val_loss, meta)
                    torch.save(payload, best_checkpoint_path)
                    torch.save(payload, checkpoint_path)
                append_eval_history(eval_history_path, {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": step,
                    "train_loss": f"{losses['train']:.4f}",
                    "val_loss": f"{losses['val']:.4f}",
                    "train_acc": f"{accs['train']:.4f}",
                    "val_acc": f"{accs['val']:.4f}",
                    "best_val_loss": f"{best_val_loss:.4f}",
                    "improved": int(improved),
                })
                latest_eval = {"train": f"{losses['train']:.3f}", "val": f"{losses['val']:.3f}", "val_acc": f"{accs['val']:.3f}"}

            optimizer.zero_grad(set_to_none=True)
            loss, acc = contrastive_step(model, train_tokens, context_length, batch_size, device)
            loss.backward()
            current_lr = lr_for_step(step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            postfix = {"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
            if latest_eval:
                postfix.update(latest_eval)
            progress.set_postfix(**postfix)
            progress.update(1)
            last_step = step

            if (step + 1) % save_every_steps == 0:
                payload = make_checkpoint_payload(model, optimizer, step, best_val_loss, meta)
                torch.save(payload, latest_checkpoint_path)
    except KeyboardInterrupt:
        print("\n[interrupt] saving latest checkpoint...")
    finally:
        progress.close()

    payload = make_checkpoint_payload(model, optimizer, last_step, best_val_loss, meta)
    torch.save(payload, latest_checkpoint_path)
    if not checkpoint_path.exists():
        torch.save(payload, checkpoint_path)
    print(f"[done] latest checkpoint: {latest_checkpoint_path}")
    print(f"[done] best checkpoint: {checkpoint_path} (best_val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
