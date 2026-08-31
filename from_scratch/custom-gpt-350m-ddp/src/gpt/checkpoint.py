"""Checkpoint save/load.

A checkpoint carries the full architecture description alongside the weights, so any
consumer can rebuild the exact model that produced it without being told its size —
which is what makes switching presets safe.
"""

from pathlib import Path

import torch

from .config import PRESETS, TOKENIZER_NAME, TOKENIZER_PATH
from .model import TinyGPT
from .tokenizer import load_tokenizer


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
        "ffn_hidden": model_cfg.ffn_hidden,
        "rope_theta": model_cfg.rope_theta,
        "preset_label": label,
        "param_count": model.param_count(),
        # Which attention kernel this state_dict's parameter names/shapes match —
        # "naive" (nn.MultiheadAttention) and "sdpa" (plain in_proj/out_proj Linear
        # layers) hold numerically-equivalent weights but under different key names,
        # so load_model/remap_attn_impl need this to rebuild the matching structure.
        "attn_impl": getattr(model, "attn_impl", "naive"),
        # Provenance.
        "grad_accum_steps": train_cfg.grad_accum_steps,
        "batch_size": train_cfg.batch_size,
        "tokenizer": TOKENIZER_NAME,
        "architecture": "gpt_decoder_pre_norm_weight_tied",
        "training_objective": "raw_next_token_prediction",
    }


def remap_attn_impl(state_dict, num_layers, from_impl, to_impl):
    """Rename attention parameter keys between the "naive" and "sdpa" CausalSelfAttention
    implementations, so weights trained under one load correctly under the other.

    The two implementations hold numerically-identical parameters (same shapes, same
    values) but organize them under different module paths:
      naive: blocks.<i>.attn.attn.in_proj_weight / .in_proj_bias
             blocks.<i>.attn.attn.out_proj.weight / .out_proj.bias
      sdpa:  blocks.<i>.attn.in_proj.weight / .in_proj.bias
             blocks.<i>.attn.out_proj.weight / .out_proj.bias
    This is a pure key rename (no reshape/transpose) — nn.MultiheadAttention's internal
    in_proj_weight is already a plain (3*embed_size, embed_size) matrix, identical in
    shape and meaning to a fused nn.Linear(embed_size, 3*embed_size).weight.
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
                if key == old:
                    new_key = f"blocks.{i}.{new_suffix}"
        remapped[new_key] = value
    return remapped


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
    # The tokenizer is a project artifact, not a named public encoding — a checkpoint
    # records which one it was trained with so a mismatch is visible, but the bytes
    # live on disk and must be the same file.
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # NOTE: no attn_impl kwarg here — this project's TinyGPT (RoPE + SDPA-only, see
    # model.py) doesn't accept one; that used to be passed anyway, a leftover from
    # copying this file from the sibling GPT-2-style projects before the RoPE rewrite
    # trimmed the attn_impl switch out of model.py, and it made every load_model() call
    # raise TypeError unconditionally. remap_attn_impl()/make_payload()'s "attn_impl"
    # field are dead code here for the same reason — left alone, out of scope for this
    # fix, since neither one crashes anything on its own.
    #
    # ffn_hidden/rope_theta: make_payload() didn't save these until just now, so any
    # checkpoint written before this fix has neither key (confirmed against real
    # checkpoints on disk) — without ffn_hidden specifically, TinyGPT can't even be
    # constructed (SwiGLU's hidden width has no other source of truth). Recover both
    # from the matching named preset in PRESETS via the checkpoint's own preset_label
    # (always saved) when they're missing; preset_label is only useful as a recovery
    # key when the run used a plain preset, not a custom GPT_FFN_HIDDEN/GPT_ROPE_THETA
    # override, so this is a best-effort fallback for pre-existing checkpoints, not a
    # substitute for the real fix above.
    preset = PRESETS.get(checkpoint.get("preset_label"))
    if "ffn_hidden" not in checkpoint and preset is None:
        raise KeyError(
            f"{checkpoint_path} has no 'ffn_hidden' (saved before that field existed) "
            f"and its preset_label {checkpoint.get('preset_label')!r} isn't a known "
            f"preset in PRESETS, so the SwiGLU hidden width can't be recovered. "
            f"Retrain, or edit this checkpoint's preset_label to the closest match."
        )
    ffn_hidden = checkpoint.get("ffn_hidden", preset.ffn_hidden if preset else None)
    rope_theta = checkpoint.get("rope_theta", preset.rope_theta if preset else 10000.0)

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embed_size=checkpoint["embed_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        ffn_hidden=ffn_hidden,
        dropout=checkpoint.get("dropout", 0.0),
        rope_theta=rope_theta,
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


def select_checkpoint(paths, which=None):
    """Pick a specific checkpoint by name, or fall back to resolve_serving_checkpoint()'s
    best->latest->final preference when `which` is None (unchanged default behavior).

    Exists because "best" isn't always the checkpoint worth looking at: best.pt only
    updates when a run's test_loss beats its prior best, so it can go stale for a long
    stretch after a real, non-regressive change to what "beating the old best" even
    means — e.g. a corpus rebuild that makes the test set genuinely harder. latest.pt
    always reflects current training state regardless of whether it's "won" yet. An
    explicit choice that doesn't exist on disk fails loudly rather than silently
    falling back to a different checkpoint than the one asked for.
    """
    if which is None:
        return resolve_serving_checkpoint(paths)
    mapping = {
        "best": paths.best_checkpoint,
        "latest": paths.latest_checkpoint,
        "final": paths.final_checkpoint,
    }
    if which not in mapping:
        raise ValueError(f"Unknown checkpoint choice {which!r}. Use one of: {', '.join(mapping)}")
    path = mapping[which]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (--checkpoint {which} was requested explicitly).")
    return path


def is_compatible(checkpoint, model_cfg, context_length):
    """Whether a checkpoint's architecture matches the currently configured one."""
    return (
        int(checkpoint.get("embed_size", -1)) == model_cfg.embed_size
        and int(checkpoint.get("num_heads", -1)) == model_cfg.num_heads
        and int(checkpoint.get("num_layers", -1)) == model_cfg.num_layers
        and int(checkpoint.get("context_length", -1)) == context_length
        and int(checkpoint.get("vocab_size", -1)) == model_cfg.vocab_size
    )
