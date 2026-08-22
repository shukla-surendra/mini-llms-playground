"""Score a saved checkpoint on held-out validation data, without retraining.

train.py's ``estimate_loss`` only samples random val batches mid-training and never
saves anything beyond a final loss number. This walks every held-out position exactly
once (deterministic, reproducible across runs) and reports loss/perplexity/accuracy
against two baselines, so "is this checkpoint any good" has a real answer.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

import torch

from .data import TextData
from .model import GPT
from .runtime import pick_device
from .train import CHECKPOINT_PATH


@torch.no_grad()
def evaluate(model: GPT, data: TextData, device: str, batch_size: int = 64) -> dict:
    block_size = model.cfg.block_size
    val = data.val_data
    n_windows = (len(val) - 1) // block_size
    if n_windows == 0:
        raise ValueError("val split is shorter than one block_size; nothing to evaluate.")
    unk_id = data.tokenizer.stoi["<unk>"]

    total_loss, total, correct1, correct5 = 0.0, 0, 0, 0
    known_total, known_correct1, known_correct5, unk_total = 0, 0, 0, 0
    starts = range(0, n_windows * block_size, block_size)
    for i in range(0, len(starts), batch_size):
        batch_starts = list(starts)[i : i + batch_size]
        x = torch.stack([val[s : s + block_size] for s in batch_starts]).to(device)
        y = torch.stack([val[s + 1 : s + block_size + 1] for s in batch_starts]).to(device)
        logits, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total += x.numel()

        top1 = logits.argmax(dim=-1)
        top5 = logits.topk(5, dim=-1).indices
        hit1 = top1 == y
        hit5 = (top5 == y.unsqueeze(-1)).any(dim=-1)
        correct1 += hit1.sum().item()
        correct5 += hit5.sum().item()

        known = y != unk_id
        known_total += known.sum().item()
        known_correct1 += (hit1 & known).sum().item()
        known_correct5 += (hit5 & known).sum().item()
        unk_total += (~known).sum().item()

    avg_loss = total_loss / total
    return {
        "n_positions": total,
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "top1_accuracy": correct1 / total,
        "top5_accuracy": correct5 / total,
        "n_known_positions": known_total,
        "top1_accuracy_known": known_correct1 / known_total if known_total else float("nan"),
        "top5_accuracy_known": known_correct5 / known_total if known_total else float("nan"),
        "unk_target_rate": unk_total / total,
    }


def baselines(data: TextData) -> dict:
    # Majority-class accuracy is the "did the model learn anything at all" floor: a
    # model that just always predicts the single most common token beats a model that
    # learned nothing wrong. <unk> is itself extremely frequent in a capped-vocab word
    # tokenizer, so it's reported separately - it's a real, honest baseline, but
    # "predicting <unk> correctly" doesn't mean predicting the actual next word right.
    counts = Counter(data.train_data.tolist())
    unk_id = data.tokenizer.stoi["<unk>"]
    n = len(data.train_data)
    top_id, top_n = counts.most_common(1)[0]
    known_counts = {tid: c for tid, c in counts.items() if tid != unk_id}
    n_known = sum(known_counts.values())
    top_known_id, top_known_n = max(known_counts.items(), key=lambda kv: kv[1])
    return {
        "uniform_loss": math.log(data.tokenizer.vocab_size),
        "uniform_perplexity": data.tokenizer.vocab_size,
        "majority_token": data.tokenizer.itos[top_id],
        "majority_accuracy": top_n / n,
        "majority_known_token": data.tokenizer.itos[top_known_id],
        "majority_known_accuracy": top_known_n / n_known,
    }


def main() -> None:
    argparse.ArgumentParser(description="Score checkpoints/word-gpt.pt on held-out validation data.").parse_args()
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("No checkpoint yet. Run `make train` before `make eval`.")

    device = pick_device()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    data = TextData()
    if data.tokenizer.stoi != checkpoint["stoi"]:
        print(
            "WARNING: data/corpus.txt (or MAX_VOCAB_SIZE) has changed since this checkpoint "
            "was trained - the val split and vocabulary no longer line up with what it saw. "
            "Re-run `make train` to refresh the checkpoint before trusting these numbers.\n"
        )

    model = GPT(checkpoint["gpt_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    m = evaluate(model, data, device)
    b = baselines(data)

    print(f"device={device}  held-out positions scored={m['n_positions']:,}")
    print(f"loss:            {m['loss']:.4f}   (uniform-random baseline: {b['uniform_loss']:.4f})")
    print(f"perplexity:      {m['perplexity']:.1f}   (uniform-random baseline: {b['uniform_perplexity']:.0f})")
    print(
        f"top-1 accuracy:  {m['top1_accuracy'] * 100:.2f}%   "
        f"(always-predict-{b['majority_token']!r} baseline: {b['majority_accuracy'] * 100:.2f}%)"
    )
    print(f"top-5 accuracy:  {m['top5_accuracy'] * 100:.2f}%")
    print(f"<unk> share of held-out targets: {m['unk_target_rate'] * 100:.2f}%")
    print(
        f"\nexcluding <unk> targets ({m['n_known_positions']:,} positions, "
        "the fairer 'did it predict a real word' number):"
    )
    print(
        f"top-1 accuracy:  {m['top1_accuracy_known'] * 100:.2f}%   "
        f"(always-predict-{b['majority_known_token']!r} baseline: {b['majority_known_accuracy'] * 100:.2f}%)"
    )
    print(f"top-5 accuracy:  {m['top5_accuracy_known'] * 100:.2f}%")


if __name__ == "__main__":
    main()
