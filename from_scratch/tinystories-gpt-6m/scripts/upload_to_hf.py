#!/usr/bin/env python3
"""
Upload the TinyStories GPT model to the Hugging Face Hub.

This is NOT a standard `transformers`-library model, so this uploads the raw checkpoint,
tokenizer, and the model code itself (model.py) — the same "raw code + weights" pattern
../custom-gpt-153m/scripts/upload_to_hf.py uses, extended here with the tokenizer.json
this project's custom-trained tokenizer requires (custom-gpt-153m reuses GPT-2's public
tokenizer, so it never needed to upload one).

Usage:
    make publish REPO_ID=your-username/tinystories-gpt-6m   # reads HF_TOKEN from .env
    # or directly:
    uv run scripts/upload_to_hf.py --repo-id your-username/tinystories-gpt-6m

HF_TOKEN is loaded from a .env file in the project root (see .env.example) via
python-dotenv, so you don't need to export it manually each session. An explicit
HF_TOKEN already set in your shell environment, or --token, both take priority over .env.

See docs/PUBLISHING_TO_HUGGING_FACE.md for the full walkthrough, including what gets
uploaded and why, and how to verify the upload actually works after it completes.
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

load_dotenv(PROJECT_DIR / ".env")  # populates os.environ from .env, if present


def parse_args():
    p = argparse.ArgumentParser(description="Upload TinyStories GPT to the Hugging Face Hub")
    p.add_argument("--repo-id", required=True,
                    help="Target repo id, e.g. your-username/tinystories-gpt-6m")
    p.add_argument("--checkpoint", default="tinystories_gpt_checkpoint.pt",
                    help="Which checkpoint file to upload (relative to the project root)")
    p.add_argument("--tokenizer", default="data/tokenizer.json",
                    help="Path to the tokenizer this checkpoint was trained with — MUST "
                         "be the exact tokenizer used, not one retrained on other data")
    p.add_argument("--token", help="Hugging Face write token (or set HF_TOKEN env var)")
    p.add_argument("--private", action="store_true", help="Create/keep the repo private")
    return p.parse_args()


def main():
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: no Hugging Face token found. Set HF_TOKEN env var or pass --token.")
        print("Get one at https://huggingface.co/settings/tokens (needs 'write' permission).")
        return

    checkpoint_path = PROJECT_DIR / args.checkpoint
    if not checkpoint_path.exists():
        fallback = PROJECT_DIR / "tinystories_gpt_checkpoint_latest.pt"
        if fallback.exists():
            print(f"Note: {checkpoint_path.name} not found, using {fallback.name} instead.")
            checkpoint_path = fallback
        else:
            print(f"Error: no checkpoint found at {checkpoint_path} (or the _latest fallback). "
                  f"Train a model first — see docs/TRAINING.md.")
            return

    tokenizer_path = PROJECT_DIR / args.tokenizer
    if not tokenizer_path.exists():
        print(f"Error: tokenizer not found at {tokenizer_path}. "
              f"Run prepare_dataset.py first, or point --tokenizer at the right file.")
        return

    print(f"Preparing to upload to: https://huggingface.co/{args.repo_id}")
    try:
        create_repo(args.repo_id, repo_type="model", exist_ok=True, token=token,
                    private=args.private)
    except Exception as e:
        print(f"Error creating/accessing repo: {e}")
        return

    api = HfApi(token=token)

    # local path (relative to project root) -> path in the HF repo
    files_map = {
        str(checkpoint_path.relative_to(PROJECT_DIR)): "tinystories_gpt_checkpoint.pt",
        str(tokenizer_path.relative_to(PROJECT_DIR)): "tokenizer.json",
        "model.py": "model.py",
        "inference.py": "inference.py",
        "api_server.py": "api_server.py",
        "pyproject.toml": "pyproject.toml",
        "model_card.md": "README.md",  # HF renders README.md as the model page
    }

    for local, remote in files_map.items():
        local_path = PROJECT_DIR / local
        if not local_path.exists():
            print(f"Skipping {local} (not found)")
            continue
        print(f"Uploading {local} -> {remote}...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote,
                repo_id=args.repo_id,
                repo_type="model",
                token=token,
            )
        except Exception as e:
            print(f"Failed to upload {local}: {e}")

    print(f"\nUpload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
