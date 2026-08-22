"""The architecture knobs (chosen before training) and training-loop knobs."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 12  # How many previous *words/punctuation tokens* are visible.
    n_embd: int = 96      # Length of every token representation.
    n_head: int = 4       # Parallel attention relationships; 96 / 4 = 24 per head.
    n_layer: int = 3      # Repeated attention + MLP refinements.


@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-3
    # One pass over the ~15.29M-token corpus: tokens / (batch_size * block_size)
    # = 15,288,804 / (16 * 12) ~= 79,629 steps. The old default (1_000) was sized for
    # the original hand-written "the cat sat" corpus and only covered 1.3% of the
    # current one.
    max_steps: int = 79_629
    eval_interval: int = 2_000
    eval_iters: int = 20
