"""Device selection — previously copy-pasted (identically) into every entrypoint script."""

import torch


def get_device():
    if torch.cuda.is_available():
        # Enable TF32 on Ampere+ GPUs (e.g. A10G) for a real speedup, no accuracy cost
        # that matters here.
        torch.set_float32_matmul_precision("high")
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
