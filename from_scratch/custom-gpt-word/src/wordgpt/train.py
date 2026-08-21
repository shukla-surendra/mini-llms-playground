"""Train the word-level GPT with the standard forward -> loss -> backward -> step loop."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import GPTConfig, TrainConfig
from .data import TextData
from .model import GPT
from .runtime import pick_device

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "word-gpt.pt"


@torch.no_grad()
def estimate_loss(model: GPT, data: TextData, cfg: TrainConfig, device: str) -> dict[str, float]:
    """Evaluate fresh batches without building gradient graphs or changing weights."""
    model.eval()
    losses: dict[str, float] = {}
    for split in ("train", "val"):
        values = []
        for _ in range(cfg.eval_iters):
            x, y = data.get_batch(split, cfg.batch_size, model.cfg.block_size, device)
            _, loss = model(x, y)
            values.append(loss.item())
        losses[split] = sum(values) / len(values)
    model.train()
    return losses


def build_training_objects(device: str) -> tuple[TextData, GPT, TrainConfig, torch.optim.Optimizer]:
    """Build all state in one place so dry-run exercises exactly the real setup."""
    data = TextData()
    model = GPT(GPTConfig(vocab_size=data.tokenizer.vocab_size)).to(device)
    cfg = TrainConfig()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    return data, model, cfg, optimizer


def main() -> None:
    torch.manual_seed(1337)  # Makes the learning demo repeatable.
    device = pick_device()
    data, model, cfg, optimizer = build_training_objects(device)
    print(f"device={device} vocab_size={data.tokenizer.vocab_size} parameters={model.num_parameters():,}")

    for step in range(cfg.max_steps):
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            losses = estimate_loss(model, data, cfg, device)
            print(f"step {step:4d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")
        x, y = data.get_batch("train", cfg.batch_size, model.cfg.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)  # Gradients accumulate unless explicitly cleared.
        loss.backward()  # Autograd computes d(loss)/d(each parameter).
        optimizer.step()  # AdamW changes the parameters using those gradients.

    losses = estimate_loss(model, data, cfg, device)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "gpt_config": model.cfg,
        "stoi": data.tokenizer.stoi, "itos": data.tokenizer.itos,
        "final_losses": losses,
    }, CHECKPOINT_PATH)
    print(f"saved checkpoint -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
