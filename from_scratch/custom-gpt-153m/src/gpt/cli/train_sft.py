"""`gpt-train-sft` — instruction/chat fine-tune a pretrained base checkpoint with
masked loss (gradient from Assistant-turn tokens only). Requires data/sft/*.jsonl —
run `gpt-sft-data` first.
"""

import argparse
import os

from ..checkpoint import select_checkpoint
from ..config import Paths, load_settings, resolve_sft_config
from ..runtime import get_device
from ..training import train_sft


def main():
    parser = argparse.ArgumentParser(description="Fine-tune (SFT) the model.")
    parser.add_argument("--base-preset", default=None,
                        help="Preset the BASE (pretrained) checkpoint was trained "
                             "under (default: $GPT_PRESET or '153m'). The SFT run "
                             "itself is namespaced under '<base-preset>-sft'.")
    parser.add_argument("--checkpoint", choices=["best", "latest", "final"], default="latest",
                        help="Which base checkpoint seeds a FRESH SFT run (default: "
                             "latest). Ignored if an SFT run is already underway — "
                             "that resumes from its own checkpoints/<label>-sft/ instead.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore an in-progress SFT checkpoint and start fresh "
                             "from --checkpoint again")
    args = parser.parse_args()

    base_preset = args.base_preset or os.getenv("GPT_PRESET", "153m")
    model_cfg, _, base_paths, _ = load_settings(base_preset)
    sft_label = f"{base_preset}-sft"
    sft_paths = Paths(label=sft_label, data_dir=base_paths.data_dir)
    sft_cfg = resolve_sft_config()

    base_checkpoint_path = select_checkpoint(base_paths, args.checkpoint)
    device = get_device()

    train_sft(
        base_checkpoint_path=base_checkpoint_path,
        model_cfg=model_cfg,
        sft_cfg=sft_cfg,
        paths=sft_paths,
        base_paths=base_paths,
        label=sft_label,
        device=device,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
