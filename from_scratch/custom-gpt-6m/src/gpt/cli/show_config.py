"""`gpt-config` — inspect the active model config, or list the size preset.

This is how you answer "what do I get if I change this setting?" without training:
parameter counts are computed from the architecture formula, so they are exact.
"""

import argparse

from ..config import PRESETS, load_settings


def print_presets():
    print(f"{'preset':<10} {'params':>14}  {'ctx':>5} {'embed':>6} {'heads':>6} {'layers':>7}")
    print("-" * 56)
    for name, cfg in PRESETS.items():
        total = cfg.param_count()
        print(
            f"{name:<10} {total:>14,}  {cfg.context_length:>5} {cfg.embed_size:>6} "
            f"{cfg.num_heads:>6} {cfg.num_layers:>7}"
        )
    print()
    print("Override a field: GPT_EMBED_SIZE=384 make train")
    print("Note: vocab_size shown is a placeholder — the real value comes from "
          "data/meta.json (this project trains its own tokenizer per corpus).")


def print_active(model_cfg, train_cfg, paths, label):
    print(f"active model: {label}")
    print()
    print(model_cfg.describe())
    print()
    print("parameter breakdown:")
    breakdown = model_cfg.param_breakdown()
    width = max(len(k) for k in breakdown)
    for key, value in breakdown.items():
        share = value / breakdown["total"] * 100
        marker = "" if key == "total" else f"  ({share:5.1f}%)"
        print(f"  {key:<{width}}  {value:>13,}{marker}")
    print()
    print("training:")
    print(
        f"  batch_size={train_cfg.batch_size} x grad_accum_steps={train_cfg.grad_accum_steps} "
        f"= effective batch {train_cfg.batch_size * train_cfg.grad_accum_steps}"
    )
    print(f"  lr={train_cfg.lr} -> min_lr={train_cfg.min_lr} (warmup + cosine decay)")
    print(f"  steps={train_cfg.steps:,}  eval_interval={train_cfg.eval_interval}")
    print()
    print("paths (objective='causal' shown; mlm/contrastive/ddp/fsdp each get their own subdir):")
    print(f"  data:        {paths.train_bin} / {paths.val_bin}")
    print(f"  checkpoints: {paths.checkpoint_dir}/")
    print(f"  eval log:    {paths.eval_history}")


def main():
    parser = argparse.ArgumentParser(description="Show model/training configuration.")
    parser.add_argument("--preset", default=None, help="Inspect a specific preset")
    parser.add_argument("--list", action="store_true",
                        help="List every preset with its exact parameter count")
    args = parser.parse_args()

    if args.list:
        print_presets()
        return

    model_cfg, train_cfg, paths, label = load_settings(args.preset)
    print_active(model_cfg, train_cfg, paths, label)


if __name__ == "__main__":
    main()
