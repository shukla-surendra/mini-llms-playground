#!/usr/bin/env python3
"""Upload a trained checkpoint plus the code needed to load it to the HF Hub.

Run through the project env so `gpt` is importable:
    uv run python scripts/upload_to_hf.py --repo-id you/your-model
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gpt.checkpoint import resolve_serving_checkpoint  # noqa: E402
from gpt.config import load_settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Upload model artifacts to the HF Hub.")
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/custom-gpt-10m")
    parser.add_argument("--preset", default=None, help="Which trained size to upload")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path")
    parser.add_argument("--token", help="HF write token (or set HF_TOKEN)")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: no HF token. Set HF_TOKEN or pass --token.")
        return 1

    _, _, paths, label = load_settings(args.preset)
    checkpoint = Path(args.checkpoint) if args.checkpoint else resolve_serving_checkpoint(paths)
    if not checkpoint.exists():
        print(f"Error: checkpoint not found: {checkpoint}. Train first with `make train`.")
        return 1

    print(f"Uploading {label} from {checkpoint} -> https://huggingface.co/{args.repo_id}")
    create_repo(args.repo_id, repo_type="model", exist_ok=True, token=token)
    api = HfApi(token=token)

    # The whole package is uploaded, not just the entrypoints — loading a checkpoint
    # requires gpt.model/gpt.checkpoint/gpt.config, so a partial upload would be unusable.
    uploads = {checkpoint: f"{label}/{checkpoint.name}"}
    for path in sorted((REPO_ROOT / "src" / "gpt").rglob("*.py")):
        uploads[path] = str(path.relative_to(REPO_ROOT))
    for extra in ("README.md", "pyproject.toml", "uv.lock", "docs/DATASETS.md"):
        candidate = REPO_ROOT / extra
        if candidate.exists():
            uploads[candidate] = extra

    for local, remote in uploads.items():
        print(f"  {remote}")
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=args.repo_id,
                repo_type="model",
            )
        except Exception as exc:
            print(f"  failed: {exc}")

    print(f"\nDone: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
