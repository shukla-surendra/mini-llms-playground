"""
Masked-LM training loop — same conventions as trainer.py (config-driven, checkpoint
save/resume, eval-history CSV, cosine LR with warmup) applied to a different pretraining
objective. See docs/MASKED_LM.md for the masking policy and why loss/perplexity here are
computed only over masked positions, not the whole sequence (unlike trainer.py's causal-LM
loss, which is computed over every position since every position has a real next-token
target).

Run `gpt-data` first (same tokenized data as trainer.py — the masked-LM objective reuses
this project's existing tokenizer and train/val splits unchanged).
"""
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from ..checkpoint import is_compatible, load_checkpoint, make_payload
from ..config import load_settings, resolve_mlm_config, resolve_vocab_size
from ..model import detect_device
from ..model_mlm import apply_bert_masking, build_mlm_model

ATTN_IMPL = os.getenv("ATTN_IMPL", "naive")


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


@torch.no_grad()
def estimate_loss(model, train_tokens, val_tokens, vocab_size, ctx_len, bsz, device, mask_prob, eval_batches):
    model.eval()
    out = {}
    for name, tokens in (("train", train_tokens), ("val", val_tokens)):
        losses = []
        for _ in range(eval_batches):
            window = get_window(tokens, ctx_len, bsz, device)
            masked_input, labels = apply_bert_masking(window, vocab_size, model.mask_token_id, mask_prob)
            logits = model(masked_input)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def run(preset_name=None, resume=True):
    model_cfg, train_cfg, paths, label = load_settings(preset_name, objective="mlm")
    mlm_cfg = resolve_mlm_config()

    device = detect_device()
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"[device] using {device}")
    print(f"[mask] prob={mlm_cfg.mask_prob}")

    meta = load_meta(paths)
    print(f"[data] meta: {meta}")
    model_cfg = resolve_vocab_size(model_cfg, meta)

    train_tokens = load_tokens(paths.train_bin)
    val_tokens = load_tokens(paths.val_bin)
    print(f"[data] train_tokens={len(train_tokens):,} val_tokens={len(val_tokens):,}")

    model = build_mlm_model(
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        embed_size=model_cfg.embed_size,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        attn_impl=ATTN_IMPL,
    ).to(device)
    print(f"[model] {model.num_parameters():,} parameters (mask_token_id={model.mask_token_id})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    best_val_loss = float("inf")
    start_step = 0

    if resume and paths.latest_checkpoint.exists():
        ckpt = load_checkpoint(paths.latest_checkpoint, device)
        if is_compatible(ckpt, model_cfg, extra_check_fields={"mask_prob": mlm_cfg.mask_prob}):
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", -1) + 1
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            print(f"[resume] resumed from step {start_step}")
        else:
            print("[resume] checkpoint config mismatch, starting fresh")

    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress = trange(start_step, train_cfg.steps, desc="training", unit="step")
    optimizer.zero_grad(set_to_none=True)
    last_step = start_step - 1
    latest_eval = None

    def payload(step, best):
        extra = {"mask_prob": mlm_cfg.mask_prob, "tokenizer_path": str(paths.tokenizer_json)}
        return make_payload(model, optimizer, step, best, model_cfg, extra_fields=extra)

    try:
        for step in range(start_step, train_cfg.steps):
            if step % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1:
                losses = estimate_loss(model, train_tokens, val_tokens, model_cfg.vocab_size,
                                        model_cfg.context_length, train_cfg.batch_size, device,
                                        mlm_cfg.mask_prob, train_cfg.eval_batches)
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

            window = get_window(train_tokens, model_cfg.context_length, train_cfg.batch_size, device)
            masked_input, labels = apply_bert_masking(window, model_cfg.vocab_size, model.mask_token_id, mlm_cfg.mask_prob)
            logits = model(masked_input)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            (loss / train_cfg.grad_accum_steps).backward()

            if (step - start_step + 1) % train_cfg.grad_accum_steps == 0 or step == train_cfg.steps - 1:
                current_lr = lr_for_step(step, train_cfg.steps, train_cfg.lr, train_cfg.min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            postfix = {"loss": f"{loss.item():.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
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
