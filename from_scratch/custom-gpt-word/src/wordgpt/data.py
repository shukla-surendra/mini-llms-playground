"""Turn the corpus into random next-token-prediction examples."""

from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import WordTokenizer

CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus.txt"

# Chinchilla-ish target for this model's fixed (non-embedding) 336,864 params, 20
# tokens/param, solved for the vocab size V that makes tokens/(96*V + 336,864) ~= 20
# against the bundled corpus's ~15.29M tokens: V = 4455.
MAX_VOCAB_SIZE = 4455


class TextData:
    def __init__(
        self, corpus_path: Path = CORPUS_PATH, val_fraction: float = 0.15,
        max_vocab_size: int | None = MAX_VOCAB_SIZE,
    ) -> None:
        text = corpus_path.read_text(encoding="utf-8")
        self.tokenizer = WordTokenizer(text, max_vocab_size=max_vocab_size)
        ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split = int(len(ids) * (1 - val_fraction))
        self.train_data, self.val_data = ids[:split], ids[split:]

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        data = self.train_data if split == "train" else self.val_data
        if len(data) <= block_size:
            raise ValueError("The selected split is shorter than block_size; add corpus text or lower it.")
        # x is ["the", "cat", ...], y is the same sequence moved one place left.
        starts = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
        return x.to(device), y.to(device)
