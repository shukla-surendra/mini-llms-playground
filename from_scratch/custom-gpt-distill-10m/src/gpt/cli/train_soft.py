"""Soft-label (logit-level) distillation - the "pure" mechanism from Hinton et al. 2015,
as distinct from gpt-distill's sequence-level approach. See
../../../docs/llm-engineering/32_knowledge_distillation_mechanism_by_mechanism.md for the
full derivation of the loss below.

Teacher: real gpt2-medium (openai-community/gpt2-medium via transformers), Modified MIT
license, downloaded once and cached by huggingface_hub. It shares this project's exact
tiktoken "gpt2" vocabulary (verified: tiktoken and HF's gpt2 tokenizer produce identical
token ids for the same text), so no vocabulary-bridging is needed - teacher and student
logits are directly comparable index-for-index.

This is a ~35x student/teacher size ratio (10M student, 355M teacher) - aggressive
relative to named real distillations (DistilBERT ~1.7x, TinyBERT ~7.5x, MobileBERT ~14x;
see Chapter 32's capacity-gap section). Included and run anyway, as a real demonstration
of the mechanism with an honestly-flagged ratio, not a claim this is the ideal pairing.

Writes to checkpoints/soft/ - separate from gpt-train's checkpoints/, so a soft-label run
never overwrites a sequence-level run's weights (or vice versa). Both are comparable
afterward via `gpt-eval --checkpoint-dir checkpoints/soft`.
"""

import argparse
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from ..checkpoint import atomic_save, is_compatible, make_payload
from ..config import load_settings
from ..data.dataset import TextData
from ..model import TinyGPT
from ..runtime import get_device
from .train import append_eval_history, lr_for_step, safe_perplexity

TEACHER_MODEL_NAME = "gpt2-medium"


def distillation_loss(student_logits, teacher_logits, targets, vocab_size, alpha=0.5, T=2.0):
    """alpha=0 is pure soft-label, alpha=1 is equivalent to ordinary sequence-level
    training - 0.5 blends both, the standard starting point from the original paper.

    Both logits tensors are (B, T, V) and MUST be flattened to (B*T, V) before kl_div:
    reduction="batchmean" divides only by input.size(0), which for an unflattened 3D
    tensor is just B, not B*T - the KL sum still implicitly runs over the T dimension
    too, so skipping the flatten inflates the loss by a factor of ~context_length
    (caught empirically: this project's first smoke-test run showed soft loss ~5000
    against hard loss ~10.8, a ~460x gap at context_length=512 - not a training-dynamics
    difference, a units bug).
    """
    student_flat = student_logits.reshape(-1, vocab_size)
    teacher_flat = teacher_logits.reshape(-1, vocab_size)
    hard_loss = F.cross_entropy(student_flat, targets.reshape(-1))
    soft_loss = F.kl_div(
        F.log_softmax(student_flat / T, dim=-1),
        F.softmax(teacher_flat / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss, hard_loss, soft_loss


@torch.no_grad()
def estimate_loss(model, teacher, data, train_cfg, ctx_len, device, alpha, temperature):
    model.eval()
    out = {}
    for split in ("train", "test"):
        losses = []
        for _ in range(train_cfg.eval_batches):
            x, y = data.get_batch(split, train_cfg.batch_size, ctx_len, device)
            student_logits, _ = model(x)
            teacher_logits = teacher(x).logits
            loss, _, _ = distillation_loss(student_logits, teacher_logits, y, model.token_emb.num_embeddings, alpha, temperature)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft-label distillation from real gpt2-medium.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight on the hard (true-label) loss; (1-alpha) goes to the soft KL term.")
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    model_cfg, train_cfg, paths = load_settings()
    overrides = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.eval_interval is not None:
        overrides["eval_interval"] = args.eval_interval
    if overrides:
        train_cfg = replace(train_cfg, **overrides)

    soft_checkpoint_dir = paths.checkpoint_dir / "soft"
    soft_latest, soft_best, soft_final = (
        soft_checkpoint_dir / "latest.pt", soft_checkpoint_dir / "best.pt", soft_checkpoint_dir / "final.pt",
    )
    soft_eval_history = paths.log_dir / "train_eval_history_soft.csv"

    device = get_device()
    torch.manual_seed(train_cfg.seed)

    data = TextData(paths.train_data, paths.test_data)
    print(f"Train tokens: {len(data.train_ids):,}  Test tokens: {len(data.test_ids):,}")

    print(f"Loading teacher {TEACHER_MODEL_NAME}...")
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: {teacher_params:,} params  |  ratio: {teacher_params / model_cfg.param_count():.1f}x")

    model = TinyGPT.from_config(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    print(f"Student: {model.param_count():,} params | device={device} | alpha={args.alpha} T={args.temperature}")

    start_step, best_test_loss, processed_tokens, total_training_seconds = 0, float("inf"), 0, 0.0
    if soft_latest.exists():
        checkpoint = torch.load(soft_latest, map_location=device)
        if is_compatible(checkpoint, model_cfg):
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = int(checkpoint.get("step", -1)) + 1
            best_test_loss = float(checkpoint.get("best_test_loss", float("inf")))
            processed_tokens = int(checkpoint.get("processed_tokens", 0))
            total_training_seconds = float(checkpoint.get("total_training_seconds", 0.0))
            print(f"Resumed at step {start_step}")

    run_start = time.time()
    model.train()
    for step in range(start_step, train_cfg.steps):
        lr = lr_for_step(step, train_cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_cfg.grad_accum_steps):
            x, y = data.get_batch("train", train_cfg.batch_size, model_cfg.context_length, device)
            student_logits, _ = model(x)
            with torch.no_grad():
                teacher_logits = teacher(x).logits
            loss, hard, soft = distillation_loss(student_logits, teacher_logits, y, model_cfg.vocab_size, args.alpha, args.temperature)
            (loss / train_cfg.grad_accum_steps).backward()
            processed_tokens += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1:
            losses = estimate_loss(model, teacher, data, train_cfg, model_cfg.context_length, device, args.alpha, args.temperature)
            improved = losses["test"] < best_test_loss
            best_test_loss = min(best_test_loss, losses["test"])
            elapsed = total_training_seconds + (time.time() - run_start)
            print(
                f"step {step:6d} | lr {lr:.2e} | train {losses['train']:.3f} | test {losses['test']:.3f} | "
                f"hard {hard.item():.3f} | soft {soft.item():.3f} | {'*' if improved else ' '}"
            )
            append_eval_history(soft_eval_history, {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "step": step, "lr": lr, "train_loss": losses["train"], "test_loss": losses["test"],
                "test_perplexity": safe_perplexity(losses["test"]), "best_test_loss": best_test_loss,
                "improved": improved, "processed_tokens": processed_tokens,
                "total_training_hours": elapsed / 3600,
            })
            payload = make_payload(model, optimizer, model_cfg, step, best_test_loss, processed_tokens, elapsed)
            atomic_save(payload, soft_latest)
            if improved:
                atomic_save(payload, soft_best)

    final_elapsed = total_training_seconds + (time.time() - run_start)
    atomic_save(
        make_payload(model, optimizer, model_cfg, train_cfg.steps - 1, best_test_loss, processed_tokens, final_elapsed),
        soft_final,
    )
    print(f"done - {train_cfg.steps:,} steps, {final_elapsed / 60:.1f} min, best test loss {best_test_loss:.3f}")
    print(f"checkpoints -> {soft_checkpoint_dir}/  (compare against checkpoints/ via gpt-eval)")


if __name__ == "__main__":
    main()
