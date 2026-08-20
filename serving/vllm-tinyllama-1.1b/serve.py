#!/usr/bin/env python3
"""Launch TinyLlama with vLLM where accelerated hardware is available."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys

CUDA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
METAL_MODEL = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"


def accelerator() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return None
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8005")))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    detected = accelerator()
    backend = args.backend
    if backend == "auto":
        # mps still needs the `vllm` CLI (from vLLM-Metal's `[vllm]` extra). As of
        # 2026-08 that install fails on every real Mac — vLLM's PyPI package
        # unconditionally depends on nvidia-cudnn-frontend/nvidia-cutlass-dsl, neither
        # of which has a macOS wheel (see README's Known issues) — so check for the
        # CLI rather than assume the install worked, and fall back to CPU otherwise.
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        if detected == "cuda":
            backend = "cuda"
        elif is_apple_silicon and shutil.which("vllm") is not None:
            backend = "mps"
        else:
            backend = "cpu"
    if backend == "cuda":
        if detected != "cuda":
            parser.error("CUDA was requested but PyTorch cannot access an NVIDIA GPU.")
        model = CUDA_MODEL
        command = ["vllm", "serve", model, "--host", args.host, "--port", str(args.port), "--dtype", "auto", "--max-model-len", "2048", "--gpu-memory-utilization", "0.70"]
        hint = "uv sync --extra vllm"
    elif backend == "mps":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            parser.error("MPS/Metal is available only on Apple Silicon macOS.")
        model = METAL_MODEL
        command = ["vllm", "serve", model, "--host", args.host, "--port", str(args.port)]
        hint = "uv sync --extra metal"
    else:
        model = CUDA_MODEL
        command = [sys.executable, "-m", "cpu_server", "--model", model, "--host", args.host, "--port", str(args.port)]
        hint = "uv sync --extra cpu"

    if args.check:
        print(f"backend={backend}\nmodel={model}\ncommand={' '.join(command)}")
        return
    if command[0] != sys.executable and shutil.which(command[0]) is None:
        parser.error(f"'{command[0]}' is not installed. Run: {hint}")
    print(f"Starting {backend} server with {model} at http://{args.host}:{args.port}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
