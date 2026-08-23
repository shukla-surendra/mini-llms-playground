"""Score a saved checkpoint on held-out validation data, without retraining. Walks every
held-out position exactly once (deterministic), unlike training's randomly-sampled
estimate_loss."""

import math

import torch

from ..checkpoint import load_model, resolve_serving_checkpoint
from ..config import load_settings
from ..data.dataset import TextData
from ..runtime import get_device


@torch.no_grad()
def evaluate(model, data, device, batch_size=32):
    context_length = model.context_length
    test = data.test_ids
    n_windows = (len(test) - 1) // context_length
    if n_windows == 0:
        raise ValueError("test split is shorter than one context_length; nothing to evaluate.")

    total_loss, total, correct1, correct5 = 0.0, 0, 0, 0
    starts = list(range(0, n_windows * context_length, context_length))
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        x = torch.stack([test[s : s + context_length] for s in batch_starts]).to(device)
        y = torch.stack([test[s + 1 : s + context_length + 1] for s in batch_starts]).to(device)
        logits, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total += x.numel()
        top1 = logits.argmax(dim=-1)
        top5 = logits.topk(5, dim=-1).indices
        correct1 += (top1 == y).sum().item()
        correct5 += (top5 == y.unsqueeze(-1)).any(dim=-1).sum().item()

    avg_loss = total_loss / total
    return {
        "n_positions": total, "loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20.0)),
        "top1_accuracy": correct1 / total, "top5_accuracy": correct5 / total,
    }


def main() -> None:
    model_cfg, _, paths = load_settings()
    device = get_device()
    checkpoint, tokenizer, model = load_model(resolve_serving_checkpoint(paths), device)
    data = TextData(paths.train_data, paths.test_data)

    m = evaluate(model, data, device)
    uniform_loss = math.log(model_cfg.vocab_size)
    print(f"device={device}  held-out positions scored={m['n_positions']:,}")
    print(f"loss:       {m['loss']:.4f}   (uniform-random baseline: {uniform_loss:.4f})")
    print(f"perplexity: {m['perplexity']:.1f}   (uniform-random baseline: {model_cfg.vocab_size})")
    print(f"top-1 accuracy: {m['top1_accuracy'] * 100:.2f}%")
    print(f"top-5 accuracy: {m['top5_accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
