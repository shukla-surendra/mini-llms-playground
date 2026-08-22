"""Device selection — previously copy-pasted (identically) into every entrypoint script."""

import os

import torch

_VALID_DEVICES = ("cuda", "mps", "cpu")


def get_device():
    """`GPT_DEVICE=cpu|mps|cuda` forces a specific device, bypassing autodetection —
    useful for running gpt-qa-report/gpt-infer on CPU deliberately while a gpt-train(-sft)
    run keeps the one local GPU (MPS is a single shared device on a Mac; there's no
    per-process isolation the way multiple CUDA devices would give you). Unset (the
    default) keeps the existing cuda -> mps -> cpu autodetect behavior unchanged.
    """
    forced = os.getenv("GPT_DEVICE")
    if forced is not None:
        if forced not in _VALID_DEVICES:
            raise ValueError(f"GPT_DEVICE={forced!r} — use one of {_VALID_DEVICES}")
        if forced == "cuda":
            torch.set_float32_matmul_precision("high")
        return forced

    if torch.cuda.is_available():
        # Enable TF32 on Ampere+ GPUs (e.g. A10G) for a real speedup, no accuracy cost
        # that matters here.
        torch.set_float32_matmul_precision("high")
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
