"""Checkpoint save/load/compatibility-check, shared by all three training objectives
(causal, mlm, contrastive) — factored out of what used to be three near-identical copies
of this logic in train.py/train_mlm.py/train_contrastive.py.
"""

import torch


def make_payload(model, optimizer, step, best_val_loss, model_cfg, extra_fields=None):
    """Build a checkpoint dict. `extra_fields` holds objective-specific extras (e.g.
    `mask_prob` for MLM, `proj_dim`/`temperature` for contrastive) merged in on top of
    the shared architecture fields.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "vocab_size": model_cfg.vocab_size,
        "context_length": model_cfg.context_length,
        "embed_size": model_cfg.embed_size,
        "num_heads": model_cfg.num_heads,
        "num_layers": model_cfg.num_layers,
        "dropout": model_cfg.dropout,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def is_compatible(checkpoint, model_cfg, extra_check_fields=None) -> bool:
    """Whether a loaded checkpoint's architecture matches the currently-configured
    model — resuming across a mismatch (e.g. a different embed_size) would silently
    corrupt the run, so every field that determines tensor shapes is checked explicitly.
    `extra_check_fields` is a dict of {field: expected_value} for objective-specific
    shape-determining fields (e.g. contrastive's proj_dim).
    """
    compatible = (
        checkpoint.get("embed_size") == model_cfg.embed_size
        and checkpoint.get("num_heads") == model_cfg.num_heads
        and checkpoint.get("num_layers") == model_cfg.num_layers
        and checkpoint.get("context_length") == model_cfg.context_length
        and checkpoint.get("vocab_size") == model_cfg.vocab_size
    )
    if extra_check_fields:
        compatible = compatible and all(
            checkpoint.get(field) == expected for field, expected in extra_check_fields.items()
        )
    return compatible


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)
