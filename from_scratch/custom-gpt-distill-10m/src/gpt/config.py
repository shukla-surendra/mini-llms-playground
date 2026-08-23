"""Every tunable knob in one place: model size, training hyperparameters, paths.

Architecture matches `custom-gpt-10m`'s real preset exactly (context_length=512,
embed_size=160, num_heads=8, num_layers=6, ~9.98M params) — the only project in this
repo with a *measured*, currently-in-progress local training run on this exact
MacBook, so it's the lowest-risk size to build a second local project around. The
difference from `custom-gpt-10m` is entirely in *where the training data comes from*:
this project's `data/corpus/train.txt`/`test.txt` are written by `cli/distill.py`
(sequence-level distillation from a local Ollama teacher — see
`../../../docs/LLM_AS_JUDGE_AND_DISTILLATION.md`), not fetched/assembled from
HuggingFace pretraining sources.
"""

from dataclasses import dataclass
from pathlib import Path

# GPT-2 BPE via tiktoken — a real, standard vocabulary the teacher's prose needs no
# truncation to fit into (unlike custom-gpt-word's capped 4,455-word vocabulary).
TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50257


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. `param_count()` mirrors model.py's actual layers exactly."""

    context_length: int = 512
    embed_size: int = 160
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE

    def __post_init__(self):
        if self.embed_size % self.num_heads != 0:
            raise ValueError(
                f"embed_size ({self.embed_size}) must be divisible by num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.embed_size // self.num_heads

    def param_count(self) -> int:
        """token_emb (V*E, lm_head is weight-tied so it adds 0) + pos_emb (C*E) +
        layers*(12E^2 + 13E) + final LayerNorm (2E) — see custom-gpt-10m/docs/
        MODEL_SIZING_GUIDE.md for the derivation of the per-block term."""
        e, c, v, layers = self.embed_size, self.context_length, self.vocab_size, self.num_layers
        return v * e + c * e + layers * (12 * e * e + 13 * e) + 2 * e

    def describe(self) -> str:
        total = self.param_count()
        return (
            f"context_length={self.context_length}  embed_size={self.embed_size}  "
            f"num_heads={self.num_heads} (head_dim={self.head_dim})  num_layers={self.num_layers}  "
            f"dropout={self.dropout}\nparameters: {total:,} ({total / 1e6:.2f}M)"
        )


@dataclass(frozen=True)
class TrainConfig:
    # batch_size * grad_accum_steps * context_length = tokens consumed per optimizer
    # step: 8 * 4 * 512 = 16,384. A distilled corpus is small (tens to low hundreds of
    # thousands of unique tokens, not hundreds of millions) - training runs many
    # epochs over it by design, unlike a pretraining-scale project. Once a real corpus
    # exists, sanity-check via: epochs = (steps * 16,384) / corpus_tokens - `gpt-config`
    # prints this once data/corpus/train.txt exists.
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    steps: int = 20_000
    eval_interval: int = 200
    eval_batches: int = 20
    save_every_steps: int = 1_000
    seed: int = 42
    max_new_tokens: int = 80
    demo_prompt: str = "User: Give me one practical tip for staying focused while studying.\nAssistant:"


@dataclass(frozen=True)
class Paths:
    data_dir: Path = Path("data")
    corpus_dir: Path = Path("data/corpus")
    distilled_dir: Path = Path("data/distilled")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

    @property
    def train_data(self) -> Path:
        return self.corpus_dir / "train.txt"

    @property
    def test_data(self) -> Path:
        return self.corpus_dir / "test.txt"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def final_checkpoint(self) -> Path:
        return self.checkpoint_dir / "final.pt"

    @property
    def eval_history(self) -> Path:
        return self.log_dir / "train_eval_history.csv"


def load_settings():
    """One call for everything an entrypoint needs: (model_cfg, train_cfg, paths)."""
    return ModelConfig(), TrainConfig(), Paths()
