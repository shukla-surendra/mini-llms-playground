"""Checkpoint save/load.

A checkpoint carries the full architecture description alongside the weights, so any
consumer can rebuild the exact model that produced it without being told its size —
which is what makes switching presets safe.
"""

from pathlib import Path

import tiktoken
import torch

from .config import TOKENIZER_NAME
from .model import TinyGPT


def atomic_save(payload, path):
    """torch.save, but a Ctrl-C mid-write can't leave a truncated file behind.

    Writes to a sibling .tmp then atomically renames, so a reader sees either the
    complete old file or the complete new one. Plain torch.save has no such guarantee:
    Python can raise KeyboardInterrupt at nearly any bytecode boundary, including
    partway through serializing a multi-hundred-MB tensor.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def make_payload(model, optimizer, model_cfg, train_cfg, context_length, step,
                 best_test_loss, processed_tokens, total_training_seconds, label):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_test_loss": best_test_loss,
        "processed_tokens": processed_tokens,
        "total_training_seconds": float(total_training_seconds),
        # Architecture — everything needed to rebuild this exact model.
        "vocab_size": model_cfg.vocab_size,
        "context_length": context_length,
        "embed_size": model_cfg.embed_size,
        "num_heads": model_cfg.num_heads,
        "num_layers": model_cfg.num_layers,
        "dropout": model_cfg.dropout,
        "preset_label": label,
        "param_count": model.param_count(),
        # Provenance.
        "grad_accum_steps": train_cfg.grad_accum_steps,
        "batch_size": train_cfg.batch_size,
        "tokenizer": TOKENIZER_NAME,
        "architecture": "gpt_decoder_pre_norm_weight_tied",
        "training_objective": "raw_next_token_prediction",
    }


def load_model(checkpoint_path, device, eval_mode=True):
    """Rebuild the exact TinyGPT a checkpoint was saved from.

    Returns (checkpoint, tokenizer, model); callers commonly need
    checkpoint["context_length"] for generation.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run `make train` first "
            f"(checkpoints are namespaced per model size under checkpoints/)."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = tiktoken.get_encoding(checkpoint.get("tokenizer", TOKENIZER_NAME))

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embed_size=checkpoint["embed_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if eval_mode:
        model.eval()

    return checkpoint, tokenizer, model


def resolve_serving_checkpoint(paths):
    """Best-by-test-loss checkpoint, falling back to the latest resume checkpoint
    (which is all that exists early in a run), then to final."""
    for candidate in (paths.best_checkpoint, paths.latest_checkpoint, paths.final_checkpoint):
        if candidate.exists():
            return candidate
    return paths.best_checkpoint


def is_compatible(checkpoint, model_cfg, context_length):
    """Whether a checkpoint's architecture matches the currently configured one."""
    return (
        int(checkpoint.get("embed_size", -1)) == model_cfg.embed_size
        and int(checkpoint.get("num_heads", -1)) == model_cfg.num_heads
        and int(checkpoint.get("num_layers", -1)) == model_cfg.num_layers
        and int(checkpoint.get("context_length", -1)) == context_length
        and int(checkpoint.get("vocab_size", -1)) == model_cfg.vocab_size
    )
