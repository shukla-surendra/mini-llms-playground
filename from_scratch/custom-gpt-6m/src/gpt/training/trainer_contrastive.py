"""
Contrastive self-supervised training loop — same conventions as trainer.py/trainer_mlm.py
(config-driven, checkpoint save/resume, eval-history CSV, cosine LR with warmup) applied
to a third pretraining objective. See docs/CONTRASTIVE_LEARNING.md for the SimCSE
positive-pair mechanism and the InfoNCE loss.

Run `gpt-data` first (same tokenized data/tokenizer as the other two training modules).
"""
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm import trange

from ..checkpoint import is_compatible, load_checkpoint, make_payload
from ..config import load_settings, resolve_contrastive_config, resolve_vocab_size
from ..model import detect_device
from ..model_contrastive import build_contrastive_model, info_nce_loss

ATTN_IMPL = os.getenv("ATTN_IMPL", "naive")

# batch_size directly determines the number of in-batch negatives (batch_size - 1 per
# anchor) — worth noting since, unlike the causal-LM/MLM trainers, batch_size here isn't
# just a memory/throughput knob, it's part of the objective's difficulty.


def load_meta(paths):
    with open(paths.meta_json) as f:
        return json.load(f)


def load_tokens(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def get_window(tokens, ctx_len, bsz, device):
    max_start = len(tokens) - ctx_len
    ix = torch.randint(0, max_start, (bsz,))
    return torch.stack([tokens[i:i + ctx_len] for i in ix]).to(device)


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


def contrastive_step(model, tokens, ctx_len, bsz, device, temperature):
    window = get_window(tokens, ctx_len, bsz, device)
    # Same input, two independent forward passes — model.training controls whether
    # dropout is actually stochastic (see estimate_loss, which deliberately keeps
    # training-mode dropout on despite otherwise looking like an eval function — see
    # docs/CONTRASTIVE_LEARNING.md's gotcha section for why).
    z1 = model(window)
    z2 = model(window)
    return info_nce_loss(z1, z2, temperature)


def estimate_loss(model, train_tokens, val_tokens, ctx_len, bsz, device, temperature, eval_batches):
    # Deliberately does NOT call model.eval() — the contrastive objective's positive pairs
    # only exist because dropout is stochastic; switching to eval mode would make z1 and
    # z2 identical (accuracy trivially 1.0, an uninformative eval signal). This is the one
    # meaningful difference from trainer.py/trainer_mlm.py's estimate_loss.
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
                loss, acc = contrastive_step(model, tokens, ctx_len, bsz, device, temperature)
                losses.append(loss.item())
                accs.append(acc)
            out_loss[name] = sum(losses) / len(losses)
            out_acc[name] = sum(accs) / len(accs)
        return out_loss, out_acc


def run(preset_name=None, resume=True):
    model_cfg, train_cfg, paths, label = load_settings(preset_name, objective="contrastive")
    contrastive_cfg = resolve_contrastive_config()

    device = detect_device()
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"[device] using {device}")
    print(f"[contrastive] proj_dim={contrastive_cfg.proj_dim} temperature={contrastive_cfg.temperature}")

    meta = load_meta(paths)
    print(f"[data] meta: {meta}")
    model_cfg = resolve_vocab_size(model_cfg, meta)

    train_tokens = load_tokens(paths.train_bin)
    val_tokens = load_tokens(paths.val_bin)
    print(f"[data] train_tokens={len(train_tokens):,} val_tokens={len(val_tokens):,}")

    model = build_contrastive_model(
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        embed_size=model_cfg.embed_size,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        proj_dim=contrastive_cfg.proj_dim,
        attn_impl=ATTN_IMPL,
    ).to(device)
    model.train()  # stays in train mode throughout — see estimate_loss's note above
    print(f"[model] {model.num_parameters():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    best_val_loss = float("inf")
    start_step = 0

    if resume and paths.latest_checkpoint.exists():
        ckpt = load_checkpoint(paths.latest_checkpoint, device)
        if is_compatible(ckpt, model_cfg, extra_check_fields={"proj_dim": contrastive_cfg.proj_dim}):
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", -1) + 1
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            print(f"[resume] resumed from step {start_step}")
        else:
            print("[resume] checkpoint config mismatch, starting fresh")

    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress = trange(start_step, train_cfg.steps, desc="training", unit="step")
    last_step = start_step - 1
    latest_eval = None

    def payload(step, best):
        extra = {
            "proj_dim": contrastive_cfg.proj_dim,
            "temperature": contrastive_cfg.temperature,
            "tokenizer_path": str(paths.tokenizer_json),
        }
        return make_payload(model, optimizer, step, best, model_cfg, extra_fields=extra)

    try:
        for step in range(start_step, train_cfg.steps):
            if step % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1:
                losses, accs = estimate_loss(model, train_tokens, val_tokens, model_cfg.context_length,
                                              train_cfg.batch_size, device, contrastive_cfg.temperature,
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
                    "train_acc": f"{accs['train']:.4f}",
                    "val_acc": f"{accs['val']:.4f}",
                    "best_val_loss": f"{best_val_loss:.4f}",
                    "improved": int(improved),
                })
                latest_eval = {"train": f"{losses['train']:.3f}", "val": f"{losses['val']:.3f}", "val_acc": f"{accs['val']:.3f}"}

            optimizer.zero_grad(set_to_none=True)
            loss, acc = contrastive_step(model, train_tokens, model_cfg.context_length, train_cfg.batch_size,
                                          device, contrastive_cfg.temperature)
            loss.backward()
            current_lr = lr_for_step(step, train_cfg.steps, train_cfg.lr, train_cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip_norm)
            optimizer.step()

            postfix = {"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
            if latest_eval:
                postfix.update(latest_eval)
            progress.set_postfix(**postfix)
            progress.update(1)
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
