"""
The training loop — the part of the codebase where the model actually learns anything.

WHAT "learning" is, mechanically, in five repeated steps:
  1. Show the model a batch of examples -> it produces a guess (forward pass).
  2. Measure how wrong the guess was, as a single number (the loss).
  3. Ask PyTorch's autograd engine: "for every one of this model's ~800,000 weights,
     if I nudged it slightly, would the loss go up or down, and by how much?" — this is
     `loss.backward()`, and the answer for each weight is that weight's ".grad".
  4. Nudge every weight a small step in the direction that *reduces* the loss —
     `optimizer.step()`.
  5. Reset the nudge-direction bookkeeping before the next batch — `zero_grad()`.
That's the entire mechanism. Every "smart" thing this model ever does is the accumulated
effect of this five-step loop repeated `max_steps` times. Deep dive on the calculus
behind step 3 and the update rule in step 4:
docs/llm-engineering/03_how_neural_networks_learn.md, grounded further in a bigger,
production training loop at docs/llm-engineering/13_the_training_loop_mechanism_by_mechanism.md.

Run with: `python -m nanogpt.train` (or `make train`).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import GPTConfig, TrainConfig
from .data import TextData
from .model import GPT

_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "nano.pt"


def pick_device() -> str:
    """Prefer Apple Silicon's GPU ("mps"), then any CUDA GPU, falling back to plain
    CPU. This model is small enough (~0.8M params) to train in well under a minute on
    CPU alone if neither is available — unlike its 6M-1B-parameter siblings in this
    workspace, no GPU is actually required here."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()  # no gradients needed for evaluation -> saves memory and compute;
# `@torch.no_grad()` tells autograd to skip building the graph it would otherwise use
# for backward() entirely, since we're never going to call backward() on this loss.
def estimate_loss(model: GPT, data: TextData, cfg: TrainConfig, device: str) -> dict[str, float]:
    """Average loss over `eval_iters` fresh batches, for both splits. A *single* batch's
    loss is noisy (few hundred tokens is a small sample) — averaging several gives a
    steadier read on how the model is actually doing, especially on the held-out `val`
    split, which is the only honest signal of whether the model is generalizing or just
    memorizing the exact training windows it's seen."""
    out = {}
    model.eval()  # see the note in the training loop below on why this matters
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = data.get_batch(split, cfg.batch_size, model.cfg.block_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # flip back before returning to the training loop
    return out


def main() -> None:
    torch.manual_seed(1337)  # fixed seed: reruns are reproducible, useful while learning
    # what one change (e.g. a different learning_rate) actually did, isolated from the
    # random noise of a different data shuffle/weight initialization each time.

    device = pick_device()
    print(f"device: {device}")

    data = TextData()
    gpt_cfg = GPTConfig(vocab_size=data.tokenizer.vocab_size)
    train_cfg = TrainConfig()

    model = GPT(gpt_cfg).to(device)
    print(f"vocab_size={gpt_cfg.vocab_size}  parameters={model.num_parameters():,}")

    # AdamW: gradient descent (see the module docstring) with two refinements almost
    # every modern network training run uses instead of plain SGD:
    #   - "Adam" part: keeps a running estimate of each weight's recent gradient
    #     *average* and *variance*, and uses both to size that weight's next step —
    #     weights with noisy/inconsistent gradients get smaller, more cautious steps;
    #     weights with a clear, consistent direction get to move faster.
    #   - "W" part (decoupled weight decay): on every step, in addition to the
    #     gradient-based update, also shrinks every weight slightly toward zero. This
    #     is a regularizer — it discourages any single weight from growing huge to
    #     memorize one specific training example, which tends to generalize better to
    #     text the model wasn't trained on.
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    for step in range(train_cfg.max_steps):
        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
            losses = estimate_loss(model, data, train_cfg, device)
            print(
                f"step {step:5d} | train loss {losses['train']:.4f} "
                f"| val loss {losses['val']:.4f}"
            )

        x, y = data.get_batch("train", train_cfg.batch_size, gpt_cfg.block_size, device)
        logits, loss = model(x, y)

        # Reset every parameter's .grad to None before this step's backward pass.
        # PyTorch *accumulates* gradients into .grad by default (calling backward()
        # twice without zeroing in between adds the two gradients together) — that
        # design choice exists to support techniques like gradient accumulation over
        # multiple mini-batches, but it means a plain training loop like this one must
        # explicitly clear .grad every step, or every step's update would be
        # (incorrectly) influenced by every previous step's gradient too.
        # `set_to_none=True` sets .grad to None instead of a zero-filled tensor — a
        # small speed/memory optimization; functionally equivalent for training.
        optimizer.zero_grad(set_to_none=True)

        # Walks the computation graph built during the forward pass above, backward,
        # computing d(loss)/d(weight) for every weight that contributed to `loss`, and
        # stores each result in that weight's `.grad`. This is the entire "backprop"
        # step — everything before it was forward computation.
        loss.backward()

        # Uses each weight's freshly-computed .grad (and AdamW's running averages) to
        # actually update every weight in place. This is the only line in the whole
        # file that changes the model's parameters.
        optimizer.step()

    final_losses = estimate_loss(model, data, train_cfg, device)
    print(
        f"final | train loss {final_losses['train']:.4f} "
        f"| val loss {final_losses['val']:.4f}"
    )

    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "gpt_config": gpt_cfg,
            "stoi": data.tokenizer.stoi,
            "itos": data.tokenizer.itos,
            # Metadata the report/serving code reads for display only — never used to
            # rebuild the model (gpt_config above is the only source of truth for that).
            "step": train_cfg.max_steps,
            "final_train_loss": final_losses["train"],
            "final_val_loss": final_losses["val"],
        },
        _CHECKPOINT_PATH,
    )
    print(f"saved checkpoint -> {_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
