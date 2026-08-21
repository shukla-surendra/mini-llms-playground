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
    max_steps: int = 1_000
    eval_interval: int = 100
    eval_iters: int = 20
