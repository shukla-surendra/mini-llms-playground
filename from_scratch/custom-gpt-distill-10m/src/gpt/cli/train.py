"""The training loop: forward -> loss -> backward -> optimizer step -> zero grads,
with gradient accumulation so a small batch_size still yields a larger effective batch.
Resumes automatically from checkpoints/latest.pt if present.
"""

import argparse
import csv
import math
import time
from dataclasses import replace
from datetime import datetime, timezone

import torch

from ..checkpoint import atomic_save, is_compatible, make_payload
from ..config import load_settings
from ..data.dataset import TextData
from ..model import TinyGPT
from ..runtime import get_device

EVAL_HISTORY_FIELDS = [
    "timestamp_utc", "step", "lr", "train_loss", "test_loss", "test_perplexity",
    "best_test_loss", "improved", "processed_tokens", "total_training_hours",
]


def lr_for_step(step_idx, train_cfg):
    """Linear warmup, then cosine decay to min_lr."""
    warmup_steps = max(50, int(train_cfg.steps * 0.02))
    if step_idx < warmup_steps:
        return train_cfg.lr * float(step_idx + 1) / float(warmup_steps)
    decay_steps = max(1, train_cfg.steps - warmup_steps)
    progress = min(1.0, max(0.0, (step_idx - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    return train_cfg.min_lr + (train_cfg.lr - train_cfg.min_lr) * cosine


def safe_perplexity(loss_value):
    return float(math.exp(min(float(loss_value), 20.0)))


def append_eval_history(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_HISTORY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def estimate_loss(model, data, train_cfg, ctx_len, device):
    model.eval()
    out = {}
    for split in ("train", "test"):
        losses = []
        for _ in range(train_cfg.eval_batches):
            x, y = data.get_batch(split, train_cfg.batch_size, ctx_len, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the model (resumes automatically).")
    parser.add_argument("--steps", type=int, default=None, help="Override TrainConfig.steps (useful for a quick smoke test).")
    parser.add_argument("--eval-interval", type=int, default=None)
    args = parser.parse_args()

    model_cfg, train_cfg, paths = load_settings()
    overrides = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.eval_interval is not None:
        overrides["eval_interval"] = args.eval_interval
    if overrides:
        train_cfg = replace(train_cfg, **overrides)

    device = get_device()
    torch.manual_seed(train_cfg.seed)

    data = TextData(paths.train_data, paths.test_data)
    print(f"Train tokens: {len(data.train_ids):,}  Test tokens: {len(data.test_ids):,}")
    tokens_per_step = train_cfg.batch_size * train_cfg.grad_accum_steps * model_cfg.context_length
    epochs = (train_cfg.steps * tokens_per_step) / max(1, len(data.train_ids))
    print(f"tokens/step: {tokens_per_step:,}  ~epochs over train split at {train_cfg.steps:,} steps: {epochs:.1f}")

    model = TinyGPT.from_config(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    print(f"{model.param_count():,} parameters | device={device}")

    start_step, best_test_loss, processed_tokens, total_training_seconds = 0, float("inf"), 0, 0.0
    if paths.latest_checkpoint.exists():
        checkpoint = torch.load(paths.latest_checkpoint, map_location=device)
        if is_compatible(checkpoint, model_cfg):
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = int(checkpoint.get("step", -1)) + 1
            best_test_loss = float(checkpoint.get("best_test_loss", float("inf")))
            processed_tokens = int(checkpoint.get("processed_tokens", 0))
            total_training_seconds = float(checkpoint.get("total_training_seconds", 0.0))
            print(f"Resumed at step {start_step}")
        else:
            print("Warning: checkpoint architecture mismatch - starting fresh.")

    run_start = time.time()
    model.train()
    for step in range(start_step, train_cfg.steps):
        lr = lr_for_step(step, train_cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_cfg.grad_accum_steps):
            x, y = data.get_batch("train", train_cfg.batch_size, model_cfg.context_length, device)
            _, loss = model(x, y)
            (loss / train_cfg.grad_accum_steps).backward()
            processed_tokens += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1:
            losses = estimate_loss(model, data, train_cfg, model_cfg.context_length, device)
            improved = losses["test"] < best_test_loss
            best_test_loss = min(best_test_loss, losses["test"])
            elapsed = total_training_seconds + (time.time() - run_start)
            print(
                f"step {step:6d} | lr {lr:.2e} | train {losses['train']:.3f} | "
                f"test {losses['test']:.3f} | ppl {safe_perplexity(losses['test']):.1f} | "
                f"{'*' if improved else ' '}"
            )
            append_eval_history(paths.eval_history, {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "step": step, "lr": lr, "train_loss": losses["train"], "test_loss": losses["test"],
                "test_perplexity": safe_perplexity(losses["test"]), "best_test_loss": best_test_loss,
                "improved": improved, "processed_tokens": processed_tokens,
                "total_training_hours": elapsed / 3600,
            })
            payload = make_payload(model, optimizer, model_cfg, step, best_test_loss, processed_tokens, elapsed)
            atomic_save(payload, paths.latest_checkpoint)
            if improved:
                atomic_save(payload, paths.best_checkpoint)

    final_elapsed = total_training_seconds + (time.time() - run_start)
    atomic_save(
        make_payload(model, optimizer, model_cfg, train_cfg.steps - 1, best_test_loss, processed_tokens, final_elapsed),
        paths.final_checkpoint,
    )
    print(f"done - {train_cfg.steps:,} steps, {final_elapsed / 60:.1f} min, best test loss {best_test_loss:.3f}")


if __name__ == "__main__":
    main()
