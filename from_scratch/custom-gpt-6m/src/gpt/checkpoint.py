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


def remap_attn_impl(state_dict, num_layers, from_impl, to_impl):
    """Rename attention parameter keys between the "naive" and "sdpa" CausalSelfAttention
    implementations (model.py), so weights trained under one load correctly under the
    other. Same mechanism as the sibling custom-gpt-{10m,50m,153m} projects.

    The two implementations hold numerically-identical parameters (same shapes, same
    values) but organize them under different module paths:
      naive: blocks.<i>.attn.attn.in_proj_weight / .in_proj_bias
             blocks.<i>.attn.attn.out_proj.weight / .out_proj.bias
      sdpa:  blocks.<i>.attn.in_proj.weight / .in_proj.bias
             blocks.<i>.attn.out_proj.weight / .out_proj.bias
    This is a pure key rename (no reshape/transpose) — nn.MultiheadAttention's internal
    in_proj_weight is already a plain (3*embed_size, embed_size) matrix, identical in
    shape and meaning to a fused nn.Linear(embed_size, 3*embed_size).weight.

    Matches by SUFFIX rather than exact key, not just `blocks.<i>...` — the contrastive
    objective's model_contrastive.py wraps its TinyStoriesGPT as `self.backbone`, so its
    real keys are `backbone.blocks.<i>...`. Suffix matching handles that prefix (and any
    other future wrapper) without needing to know about it here.
    """
    if from_impl == to_impl:
        return state_dict
    if {from_impl, to_impl} != {"naive", "sdpa"}:
        raise ValueError(f"Unsupported remap: {from_impl!r} -> {to_impl!r}")

    if from_impl == "naive":
        rename = {
            "attn.attn.in_proj_weight": "attn.in_proj.weight",
            "attn.attn.in_proj_bias": "attn.in_proj.bias",
            "attn.attn.out_proj.weight": "attn.out_proj.weight",
            "attn.attn.out_proj.bias": "attn.out_proj.bias",
        }
    else:
        rename = {
            "attn.in_proj.weight": "attn.attn.in_proj_weight",
            "attn.in_proj.bias": "attn.attn.in_proj_bias",
            "attn.out_proj.weight": "attn.attn.out_proj.weight",
            "attn.out_proj.bias": "attn.attn.out_proj.bias",
        }

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for i in range(num_layers):
            for old_suffix, new_suffix in rename.items():
                old = f"blocks.{i}.{old_suffix}"
                if key == old or key.endswith("." + old):
                    new_key = key[: -len(old)] + f"blocks.{i}.{new_suffix}"
        remapped[new_key] = value
    return remapped
