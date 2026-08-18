"""
Turns the corpus text file into (input, target) tensors the model can train on.

WHAT this model's training task actually is, concretely: given a chunk of `block_size`
consecutive tokens, predict — *at every single position simultaneously* — what the next
token is. This is called "next-token prediction" or "causal language modeling", and it's
the entire objective this model is trained on; nothing more sophisticated is happening
under the hood. Deep dive: docs/llm-engineering/12_the_pretraining_objective_and_why_data_dominates.md
and docs/llm-engineering/08_what_is_a_language_model.md (both in the repo root).

WHY random windows instead of walking through the corpus in fixed-size, non-overlapping
chunks (chunk 1 = chars 0-63, chunk 2 = chars 64-127, ...): fixed chunking means the
model only ever sees "chunk boundary" at the same handful of positions, every epoch. On
a corpus this tiny (a few thousand characters), that would sharply limit how many
distinct (context, target) examples the model ever sees. Sampling a fresh random start
index on every batch means, over enough steps, the model sees a next-token prediction
starting from nearly every possible position in the corpus — far more effective use of a
small dataset. This is also exactly why training here counts in "steps", not "epochs" —
see config.py's TrainConfig.max_steps docstring.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import CharTokenizer

_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus.txt"


class TextData:
    def __init__(self, corpus_path: Path = _CORPUS_PATH, val_fraction: float = 0.1) -> None:
        self.tokenizer = CharTokenizer.from_file(corpus_path)

        text = corpus_path.read_text(encoding="utf-8")
        # `torch.long` (64-bit int): required dtype for anything used as an index into
        # an nn.Embedding table or as a target for F.cross_entropy.
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # A held-out slice the model never trains on, used only to measure how well it
        # generalizes rather than just memorizes. Split by *position* (last 10% of the
        # text), not randomly shuffled per character — shuffling individual characters
        # would destroy the very sequential structure the model is trying to learn.
        split_idx = int(len(data) * (1 - val_fraction))
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        """One training step's worth of data: `batch_size` independent (x, y) pairs,
        each x a block_size-token window and y the *same* window shifted one token to
        the right — so y[t] is always "the correct next token after x[0..t]"."""
        data = self.train_data if split == "train" else self.val_data

        # Random starting index for each example in the batch. `len(data) - block_size`
        # is the last valid start position — starting any later wouldn't leave enough
        # tokens to fill a full block_size window.
        starts = torch.randint(len(data) - block_size, (batch_size,))

        x = torch.stack([data[i : i + block_size] for i in starts])
        # y is x shifted right by exactly one token: at every position t, y[t] is the
        # single token that came right after x[t] in the original text. This is what
        # lets the loss in model.py's forward() score the prediction at *every*
        # position in the window in one pass, not just the last one.
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts])

        return x.to(device), y.to(device)
