#!/usr/bin/env python3
"""Select a local inference backend and start an OpenAI-compatible server.

CUDA uses vLLM. Apple Silicon uses mlx-lm's own native server (NOT vllm-metal —
see pyproject.toml's metal extra comment for why that wrapper doesn't currently
install on macOS). Systems without either accelerator use the small built-in
Transformers CPU fallback server.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

CUDA_MODEL = "Qwen/Qwen2.5-7B-Instruct"
METAL_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


@dataclass(frozen=True)
class Backend:
    name: str
    model: str
    command: list[str]
    install_hint: str


def torch_backend() -> str | None:
    """Return CUDA or MPS only when PyTorch confirms it can use that backend."""
    if importlib.util.find_spec("torch") is None:
        return None
    import torch  # Imported lazily so --check also works before installation.

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return None


def select_backend(requested: str, host: str, port: int) -> Backend:
    available = torch_backend()
    if requested == "auto":
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        if available == "cuda":
            requested = "cuda"
        elif is_apple_silicon:
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda":
        if available != "cuda":
            raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU.")
        return Backend(
            "cuda",
            CUDA_MODEL,
            [
                "vllm", "serve", CUDA_MODEL, "--host", host, "--port", str(port),
                "--dtype", "auto", "--max-model-len", "4096",
                "--gpu-memory-utilization", "0.85",
            ],
            "uv sync --extra vllm",
        )

    if requested == "mps":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("MPS is available only on Apple Silicon macOS.")
        return Backend(
            "mps",
            METAL_MODEL,
            [
                sys.executable, "-m", "mlx_lm.server",
                "--model", METAL_MODEL, "--host", host, "--port", str(port),
            ],
            "uv sync --extra metal",
        )

    if requested == "cpu":
        return Backend(
            "cpu",
            CUDA_MODEL,
            [sys.executable, "-m", "cpu_server", "--model", CUDA_MODEL, "--host", host, "--port", str(port)],
            "uv sync --extra cpu",
        )
    raise ValueError(f"Unknown backend: {requested}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8005")))
    parser.add_argument("--check", action="store_true", help="Print the chosen backend without starting it.")
    args = parser.parse_args()

    try:
        backend = select_backend(args.backend, args.host, args.port)
    except RuntimeError as exc:
        parser.error(str(exc))

    executable = backend.command[0]
    if args.check:
        print(f"backend={backend.name}")
        print(f"model={backend.model}")
        print("command=" + " ".join(backend.command))
        return
    if shutil.which(executable) is None and executable != sys.executable:
        parser.error(f"'{executable}' is not installed. Run: {backend.install_hint}")

    print(f"Starting {backend.name} server with {backend.model} at http://{args.host}:{args.port}", flush=True)
    subprocess.run(backend.command, check=True)


if __name__ == "__main__":
    main()
