"""Run one real optimization step and print the tensors; never save a checkpoint.

Use this before ``make train``. It proves that the corpus can be tokenized, a batch
has the expected shifted targets, the Transformer produces logits, autograd produces
non-zero gradients, and AdamW can update a weight. It is intentionally a *real* step,
not a mock, but its in-memory update disappears when the process exits.
"""

from __future__ import annotations

import torch

from .runtime import pick_device
from .train import build_training_objects


def main() -> None:
    torch.manual_seed(1337)
    device = pick_device()
    data, model, cfg, optimizer = build_training_objects(device)
    x, y = data.get_batch("train", cfg.batch_size, model.cfg.block_size, device)
    before = model.token_emb.weight[0, 0].item()
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = model.token_emb.weight.grad.norm().item()
    optimizer.step()
    after = model.token_emb.weight[0, 0].item()

    print(f"device: {device}")
    print(f"vocabulary ({data.tokenizer.vocab_size}): {' '.join(data.tokenizer.tokens)}")
    print(f"x shape: {tuple(x.shape)} | y shape: {tuple(y.shape)} | logits shape: {tuple(logits.shape)}")
    print(f"first input ids:  {x[0].tolist()}")
    print(f"first input text: {data.tokenizer.decode(x[0].tolist())}")
    print(f"first target text:{data.tokenizer.decode(y[0].tolist())}")
    print(f"loss before update: {loss.item():.4f}")
    print(f"embedding gradient L2 norm: {gradient_norm:.4f}")
    print(f"one weight changed: {before:.8f} -> {after:.8f}")
    print("dry run passed: no checkpoint was written.")


if __name__ == "__main__":
    main()
