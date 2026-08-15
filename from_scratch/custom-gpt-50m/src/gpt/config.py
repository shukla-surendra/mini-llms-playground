"""Every tunable knob in one place: model size, training hyperparameters, paths.

Model size is fully data-driven. The ~50M architecture below is this project's own
default — it lives directly on `ModelConfig`, so `gpt-train` with no env vars set trains
it without ever consulting `PRESETS`. `PRESETS`/`GPT_PRESET` remain available to
explicitly switch to another size (e.g. for comparison against the sibling projects),
and individual fields can be overridden on top of either. Nothing else in the package
hardcodes a dimension.

    GPT_PRESET=30m gpt-train          # train a ~30M model instead of the ~50M default
    GPT_EMBED_SIZE=192 gpt-train      # or override one field on top of the default/preset
"""

from dataclasses import dataclass, replace
import os
from pathlib import Path

# GPT-2 BPE via tiktoken. Declared here so parameter counts can be computed without
# loading the tokenizer; verified against the real tokenizer at training time.
TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50257


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. `param_count()` is exact — it mirrors model.py's actual layers.

    Field defaults are this project's own ~50M architecture — the config `gpt-train`
    uses when no `GPT_PRESET`/`--preset` is given, with no dict lookup involved.
    """

    context_length: int = 1024
    embed_size: int = 512
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE

    def __post_init__(self):
        if self.embed_size % self.num_heads != 0:
            raise ValueError(
                f"embed_size ({self.embed_size}) must be divisible by num_heads "
                f"({self.num_heads}) — each head gets embed_size/num_heads dimensions."
            )
        for field_name in ("context_length", "embed_size", "num_heads", "num_layers"):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be >= 1, got {getattr(self, field_name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        return self.embed_size // self.num_heads

    def param_count(self) -> int:
        """Exact trainable-parameter count for this config.

        Mirrors model.py exactly:
          token embedding   V*E        (lm_head is weight-tied to this, so it adds 0)
          position embedding C*E
          per block          12E^2 + 13E   (attention 4E^2+4E, MLP 8E^2+5E, 2 LayerNorms 4E)
          final LayerNorm    2E
        """
        e, c, v, layers = self.embed_size, self.context_length, self.vocab_size, self.num_layers
        token_emb = v * e
        pos_emb = c * e
        per_block = 12 * e * e + 13 * e
        final_ln = 2 * e
        return token_emb + pos_emb + layers * per_block + final_ln

    def param_breakdown(self) -> dict:
        e, c, v, layers = self.embed_size, self.context_length, self.vocab_size, self.num_layers
        per_block = 12 * e * e + 13 * e
        return {
            "token_embedding": v * e,
            "position_embedding": c * e,
            "transformer_blocks": layers * per_block,
            "final_layernorm": 2 * e,
            "total": self.param_count(),
        }

    def describe(self) -> str:
        total = self.param_count()
        return (
            f"context_length={self.context_length}  embed_size={self.embed_size}  "
            f"num_heads={self.num_heads} (head_dim={self.head_dim})  "
            f"num_layers={self.num_layers}  dropout={self.dropout}\n"
            f"parameters: {total:,} ({total / 1e6:.2f}M)"
        )


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 1          # keep at 1 for MPS/laptop VRAM; raise on a real GPU
    grad_accum_steps: int = 32   # effective batch = batch_size * grad_accum_steps
    lr: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    steps: int = 1_000_000
    eval_interval: int = 50
    eval_batches: int = 20
    save_every_steps: int = 200
    seed: int = 42
    max_new_tokens: int = 80     # demo completion printed at the end of a run
    demo_prompt: str = "The quick brown fox"


# Named sizes for explicitly switching away from this project's own default (e.g. to
# compare against the sibling custom-gpt-{10m,30m,153m} projects). "50m" is derived from
# ModelConfig()'s own field defaults, not re-hardcoded, so the two can never drift apart.
# Parameter counts are computed, never hardcoded, so they cannot drift out of sync either.
PRESETS = {
    "tiny": ModelConfig(context_length=256, embed_size=128, num_heads=4, num_layers=4),
    "10m": ModelConfig(context_length=512, embed_size=160, num_heads=8, num_layers=6),
    "30m": ModelConfig(context_length=512, embed_size=384, num_heads=6, num_layers=6),
    "50m": ModelConfig(),  # this project's own default architecture
    # Matches the sibling custom-gpt-153m project's architecture exactly.
    "153m": ModelConfig(context_length=1024, embed_size=768, num_heads=12, num_layers=16),
}

_ENV_OVERRIDES = {
    "context_length": ("GPT_CONTEXT_LENGTH", int),
    "embed_size": ("GPT_EMBED_SIZE", int),
    "num_heads": ("GPT_NUM_HEADS", int),
    "num_layers": ("GPT_NUM_LAYERS", int),
    "dropout": ("GPT_DROPOUT", float),
}


def resolve_model_config(preset_name=None):
    """Build the active ModelConfig: the ~50M default, or an explicit named preset, plus
    any per-field env overrides.

    With no `preset_name` and no `GPT_PRESET` set, this is `ModelConfig()` — the project's
    own default architecture, straight off the dataclass, no `PRESETS` lookup involved.

    Returns (config, label). `label` names the resulting size — `"50m"` for the untouched
    default, the preset name when one is picked, or a descriptive `custom-...` string when
    overrides changed it. Checkpoints are stored per-label so switching sizes never
    silently overwrites another model's weights.
    """
    preset_name = preset_name or os.getenv("GPT_PRESET")
    if preset_name is None:
        base, label = ModelConfig(), "50m"
    else:
        if preset_name not in PRESETS:
            available = ", ".join(PRESETS)
            raise ValueError(f"Unknown GPT_PRESET {preset_name!r}. Available: {available}")
        base, label = PRESETS[preset_name], preset_name

    overrides = {}
    for field_name, (env_var, cast) in _ENV_OVERRIDES.items():
        raw = os.getenv(env_var)
        if raw is not None:
            overrides[field_name] = cast(raw)

    if not overrides:
        return base, label

    cfg = replace(base, **overrides)
    label = (
        f"custom-e{cfg.embed_size}-l{cfg.num_layers}-h{cfg.num_heads}-c{cfg.context_length}"
    )
    return cfg, label


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. Checkpoints are namespaced per model size."""

    label: str
    data_dir: Path = Path("data")
    checkpoint_root: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

    @property
    def train_data(self) -> Path:
        return self.data_dir / "train.txt"

    @property
    def test_data(self) -> Path:
        return self.data_dir / "test.txt"

    @property
    def test_prompts(self) -> Path:
        return self.data_dir / "test_prompts.txt"

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_root / self.label

    @property
    def stop_file(self) -> Path:
        """A polled, file-based stop signal — see training/trainer.py's `_run_loop`.

        SIGINT (Ctrl-C / `kill -INT`) is normally caught as KeyboardInterrupt and handled
        gracefully, but this isn't guaranteed: a native library somewhere in the torch/MPS/
        multiprocessing stack can intercept or block signal delivery before Python's default
        handler ever sees it (observed in practice, not hypothetical — see
        docs/CODE_WALKTHROUGH.md). Checking for this file's existence every step is a
        signal-delivery-quirk-proof fallback: it only depends on the training loop's own
        Python code actually running, which — unlike signal delivery — is something we can
        already see happening (the step counter advancing) whenever a run is alive.
        Shared across labels/presets rather than namespaced per checkpoint_dir, since only
        one `gpt-train` process is ever expected to be running at a time (see the Makefile's
        `guard_not_running`).
        """
        return self.checkpoint_root / "STOP_TRAINING"

    @property
    def serving_checkpoint(self) -> Path:
        return self.checkpoint_dir / "serving.pt"

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
        return self.log_dir / f"train_eval_history_{self.label}.csv"

    @property
    def quality_history(self) -> Path:
        return self.log_dir / f"quality_history_{self.label}.jsonl"


def load_settings(preset_name=None):
    """One call for everything an entrypoint needs: (model_cfg, train_cfg, paths, label)."""
    model_cfg, label = resolve_model_config(preset_name)
    return model_cfg, TrainConfig(), Paths(label=label), label
