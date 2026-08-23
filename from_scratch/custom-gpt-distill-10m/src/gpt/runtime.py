"""Device selection."""

import torch


def get_device():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
