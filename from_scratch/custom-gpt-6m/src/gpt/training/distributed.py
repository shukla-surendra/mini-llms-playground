"""
Shared helpers for the DDP and FSDP demo trainers (trainer_ddp.py / trainer_fsdp.py) —
data loading and eval, identical between the two since only the distribution *strategy*
(DistributedDataParallel vs FullyShardedDataParallel) differs, not the data pipeline.

See docs/DISTRIBUTED_TRAINING.md for the full mechanism, and each trainer module's own
docstring for what is and isn't actually being demonstrated on this specific machine
(multi-process CPU via gloo, not real multi-GPU).
"""
import json

import numpy as np
import torch
import torch.nn.functional as F


def load_meta(paths):
    with open(paths.meta_json) as f:
        return json.load(f)


def load_tokens(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def get_batch(tokens, ctx_len, bsz):
    max_start = len(tokens) - ctx_len - 1
    ix = torch.randint(0, max_start, (bsz,))
    x = torch.stack([tokens[i:i + ctx_len] for i in ix])
    y = torch.stack([tokens[i + 1:i + ctx_len + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, tokens, ctx_len, bsz, n_batches):
    model.eval()
    losses = []
    for _ in range(n_batches):
        xb, yb = get_batch(tokens, ctx_len, bsz)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)
