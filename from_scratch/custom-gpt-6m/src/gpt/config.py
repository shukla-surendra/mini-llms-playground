"""Every tunable knob in one place: model size, training hyperparameters, paths.

Model size is fully data-driven — pick a named preset (currently just `"6m"`, this
project's one real architecture) or override individual fields, and everything
downstream (parameter count, checkpoint location, the model itself) follows
automatically.

    GPT_EMBED_SIZE=384 gpt-train      # override one field on top of the default preset

Every training knob also takes an env override, matching the sibling `custom-gpt-*`
projects' convention:

    GPT_STEPS=200 GPT_EVAL_INTERVAL=50 gpt-train   # short smoke run
"""

from dataclasses import dataclass, replace
import os
from pathlib import Path

# This project trains its OWN BPE tokenizer fresh per corpus (see data/prepare.py,
# formerly prepare_dataset.py) rather than reusing a fixed external one like the
# GPT-2-tokenizer siblings do — so, unlike their `VOCAB_SIZE = 50257` constant, there is
# no single correct vocab_size to declare here ahead of time. DEFAULT_VOCAB_SIZE is only
# a ModelConfig placeholder (prepare.py's own default target vocab size); every real
# model-build call overrides it via `dataclasses.replace(model_cfg,
# vocab_size=meta["vocab_size"])` read from data/meta.json — the tokenizer's *actual*
# trained vocab size, which can differ slightly from the target (BPE training doesn't
# always hit the requested size exactly).
TOKENIZER_NAME = "custom-bpe"
DEFAULT_VOCAB_SIZE = 4096


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. `param_count()` is exact — it mirrors model.py's actual layers.

    Field defaults are this project's own real architecture, so `PRESETS["6m"]` can be
    derived from them rather than restated — the two cannot drift apart. Same GPT-2-style
    shape as the custom-gpt-153m sibling (learned position embeddings, weight-tied head,
    LayerNorm) — see `param_count()`'s formula, which is identical to that project's.
    """

    context_length: int = 256
    embed_size: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    vocab_size: int = DEFAULT_VOCAB_SIZE

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
            f"parameters: {total:,} ({total / 1e6:.2f}M)  "
            f"[vocab_size={self.vocab_size} placeholder — real value comes from "
            f"data/meta.json at model-build time]"
        )


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters, defaulted for this project's real usage: a laptop
    CPU/MPS box, not a rented GPU (unlike the bigger custom-gpt-153m/200m/350m siblings).

    A **step is one micro-batch** forward/backward, not one optimizer update — so
    `steps` and `batch_size` together set the token budget:

        tokens = steps * batch_size * context_length
               = 5_000 * 32 * 256 = 40.96M

    `ATTN_IMPL`, `AMP`, and `GRAD_CHECKPOINT` are deliberately NOT fields here — they
    stay plain env vars read directly in each trainer module, exactly like the sibling
    projects treat `ATTN_IMPL` (see custom-gpt-153m/src/gpt/training/trainer.py). This
    also preserves this project's own AMP mechanism unchanged (real fp16+GradScaler on
    CUDA, bf16 on MPS, a documented no-op on CPU — see docs/EFFICIENT_TRAINING.md)
    rather than folding it into the siblings' different `precision="auto"` scheme.
    """

    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    steps: int = 5_000
    eval_interval: int = 250
    eval_batches: int = 20
    save_every_steps: int = 500
    seed: int = 42
    max_new_tokens: int = 120     # demo completion printed at the end of a run
    demo_prompt: str = "Once upon a time,"


@dataclass(frozen=True)
class MLMExtraConfig:
    """The masked-LM objective's one extra knob, on top of the shared ModelConfig/TrainConfig."""

    mask_prob: float = 0.15


@dataclass(frozen=True)
class ContrastiveExtraConfig:
    """The contrastive (SimCSE/InfoNCE) objective's extra knobs."""

    proj_dim: int = 128
    temperature: float = 0.05


@dataclass(frozen=True)
class DistributedConfig:
    """DDP/FSDP demo launch parameters — see training/distributed.py.

    `master_port` defaults differ between DDP (29500) and FSDP (29501) so both demos can
    run their own `mp.spawn` process groups without colliding if launched back to back;
    each CLI passes its own default explicitly rather than sharing one hardcoded constant.
    """

    world_size: int = 2
    master_port: int = 29500


# Named sizes. Parameter counts are computed, never hardcoded, so they cannot drift out
# of sync with the architecture. Only one real preset exists — this project has one real
# architecture, unlike the bigger siblings' multiple trained sizes.
PRESETS = {
    "6m": ModelConfig(),
}

DEFAULT_PRESET = "6m"

_ENV_OVERRIDES = {
    "context_length": ("GPT_CONTEXT_LENGTH", int),
    "embed_size": ("GPT_EMBED_SIZE", int),
    "num_heads": ("GPT_NUM_HEADS", int),
    "num_layers": ("GPT_NUM_LAYERS", int),
    "dropout": ("GPT_DROPOUT", float),
}


def resolve_model_config(preset_name=None):
    """Build the active ModelConfig: a named preset, plus any per-field env overrides.

    Returns (config, label). `label` names the resulting size — the preset name, or a
    descriptive `custom-...` string when overrides changed it. Checkpoints are stored
    per-label so switching sizes never silently overwrites another model's weights.
    """
    preset_name = preset_name or os.getenv("GPT_PRESET", DEFAULT_PRESET)
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS)
        raise ValueError(f"Unknown GPT_PRESET {preset_name!r}. Available: {available}")

    base = PRESETS[preset_name]
    overrides = {}
    for field_name, (env_var, cast) in _ENV_OVERRIDES.items():
        raw = os.getenv(env_var)
        if raw is not None:
            overrides[field_name] = cast(raw)

    if not overrides:
        return base, preset_name

    cfg = replace(base, **overrides)
    label = (
        f"custom-e{cfg.embed_size}-l{cfg.num_layers}-h{cfg.num_heads}-c{cfg.context_length}"
    )
    return cfg, label


def resolve_vocab_size(model_cfg: ModelConfig, meta: dict) -> ModelConfig:
    """Override the placeholder vocab_size with the real, trained tokenizer's size.

    `meta` is data/meta.json's parsed contents (see data/prepare.py) — always the source
    of truth for vocab_size, since this project's tokenizer is trained fresh per corpus
    rather than being a fixed external constant.
    """
    return replace(model_cfg, vocab_size=meta["vocab_size"])


_MLM_ENV_OVERRIDES = {"mask_prob": ("GPT_MASK_PROB", float)}
_CONTRASTIVE_ENV_OVERRIDES = {
    "proj_dim": ("GPT_PROJ_DIM", int),
    "temperature": ("GPT_TEMPERATURE", float),
}


def resolve_mlm_config() -> MLMExtraConfig:
    overrides = {}
    for field_name, (env_var, cast) in _MLM_ENV_OVERRIDES.items():
        raw = os.getenv(env_var)
        if raw is not None:
            overrides[field_name] = cast(raw)
    return replace(MLMExtraConfig(), **overrides) if overrides else MLMExtraConfig()


def resolve_contrastive_config() -> ContrastiveExtraConfig:
    overrides = {}
    for field_name, (env_var, cast) in _CONTRASTIVE_ENV_OVERRIDES.items():
        raw = os.getenv(env_var)
        if raw is not None:
            overrides[field_name] = cast(raw)
    return replace(ContrastiveExtraConfig(), **overrides) if overrides else ContrastiveExtraConfig()


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. Checkpoints are namespaced per model size AND per training
    objective (`objective`) — unlike the siblings (one objective each), this project has
    three independently-resumable checkpoint lineages (causal/mlm/contrastive) sharing
    one architecture label, plus DDP/FSDP demo output.
    """

    label: str
    objective: str = "causal"
    data_dir: Path = Path("data")
    checkpoint_root: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

    @property
    def meta_json(self) -> Path:
        return self.data_dir / "meta.json"

    @property
    def tokenizer_json(self) -> Path:
        return self.data_dir / "tokenizer.json"

    @property
    def train_bin(self) -> Path:
        return self.data_dir / "train.bin"

    @property
    def val_bin(self) -> Path:
        return self.data_dir / "val.bin"

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_root / self.label / self.objective

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
        return self.log_dir / f"train_eval_history_{self.label}_{self.objective}.csv"


_TRAIN_ENV_OVERRIDES = {
    "batch_size": ("GPT_BATCH_SIZE", int),
    "grad_accum_steps": ("GPT_GRAD_ACCUM", int),
    "lr": ("GPT_LR", float),
    "min_lr": ("GPT_MIN_LR", float),
    "steps": ("GPT_STEPS", int),
    "eval_interval": ("GPT_EVAL_INTERVAL", int),
    "eval_batches": ("GPT_EVAL_BATCHES", int),
    "save_every_steps": ("GPT_SAVE_EVERY", int),
}


def resolve_train_config():
    """TrainConfig with any `GPT_*` env overrides applied — no code edit needed for a
    quick smoke run (`GPT_STEPS=20 make train`)."""
    overrides = {}
    for field_name, (env_var, cast) in _TRAIN_ENV_OVERRIDES.items():
        raw = os.getenv(env_var)
        if raw is not None:
            overrides[field_name] = cast(raw)
    return replace(TrainConfig(), **overrides) if overrides else TrainConfig()


def resolve_distributed_config(default_master_port: int = 29500) -> DistributedConfig:
    world_size = int(os.getenv("WORLD_SIZE", DistributedConfig().world_size))
    master_port = int(os.getenv("MASTER_PORT", default_master_port))
    return DistributedConfig(world_size=world_size, master_port=master_port)


def load_settings(preset_name=None, objective="causal"):
    """One call for everything an entrypoint needs: (model_cfg, train_cfg, paths, label)."""
    model_cfg, label = resolve_model_config(preset_name)
    return model_cfg, resolve_train_config(), Paths(label=label, objective=objective), label
