"""Every tunable knob in one place: model size, training hyperparameters, paths.

Model size is fully data-driven — pick a named preset or override individual fields,
and everything downstream (parameter count, checkpoint location, the model itself)
follows automatically. Nothing else in the package hardcodes a dimension.

    GPT_PRESET=small gpt-train        # train the ~55M preset instead of the ~347M default
    GPT_EMBED_SIZE=512 gpt-train      # or override one field on top of a preset

`TrainConfig`'s defaults target a **single rented GPU** (see docs/GPU_TRAINING.md),
not a laptop — unlike the sibling custom-gpt-{10m,50m} projects, whose `batch_size=1`
defaults exist for MPS. Every training knob also takes an env override so a laptop
smoke test needs no code edit:

    GPT_BATCH_SIZE=1 GPT_GRAD_ACCUM=8 GPT_STEPS=200 gpt-train   # local sanity run
"""

from dataclasses import dataclass, replace
import os
from pathlib import Path

# GPT-2 BPE via tiktoken. Declared here so parameter counts can be computed without
# loading the tokenizer; verified against the real tokenizer at training time.
# This project trains its OWN tokenizer (see cli/train_tokenizer.py) rather than
# reusing GPT-2's. Two reasons, both about reasoning rather than tidiness:
#   * GPT-2's 50,257-token vocabulary would cost 51.4M of a 350M budget (15%) at
#     E=1024. A 32,768 vocabulary costs 33.6M, redirecting 17.9M into transformer
#     blocks — real capacity rather than lookup table.
#   * GPT-2's BPE merges digits inconsistently ("1234" may be one token, "1235"
#     three), which directly damages arithmetic. Ours splits digits individually,
#     so every number has one canonical representation.
# TOKENIZER_PATH is a trained tokenizer.json; VOCAB_SIZE must match what it holds
# (verified against the real tokenizer at training time).
TOKENIZER_NAME = "oxide-bpe-32k"
TOKENIZER_PATH = "tokenizer/tokenizer.json"
VOCAB_SIZE = 32768


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. `param_count()` is exact — it mirrors model.py's actual layers.

    Field defaults are this project's own ~347M architecture, so `PRESETS["350m"]`
    can be derived from them rather than restated — the two cannot drift apart.
    """

    context_length: int = 2048
    embed_size: int = 1024
    num_heads: int = 16          # head_dim = 64, the size SDPA's fused kernels want
    num_layers: int = 25
    ffn_hidden: int = 2720       # ~8/3 * embed_size — see below
    rope_theta: float = 10000.0
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
          token embedding    V*E          (lm_head is weight-tied, so it adds 0)
          per block          4E^2         attention qkv+out, no biases
                           + 3*E*F        SwiGLU gate/up/down, no biases
                           + 2E           two RMSNorm weight vectors
          final RMSNorm      E
        There is no position-embedding term: RoPE is computed from position, not
        learned, so it contributes zero parameters (and no context ceiling).
        """
        e, v, f, layers = self.embed_size, self.vocab_size, self.ffn_hidden, self.num_layers
        token_emb = v * e
        per_block = 4 * e * e + 3 * e * f + 2 * e
        final_norm = e
        return token_emb + layers * per_block + final_norm

    def param_breakdown(self) -> dict:
        e, v, f, layers = self.embed_size, self.vocab_size, self.ffn_hidden, self.num_layers
        return {
            "token_embedding": v * e,
            "attention": layers * (4 * e * e),
            "swiglu_mlp": layers * (3 * e * f),
            "rmsnorms": layers * (2 * e) + e,
            "total": self.param_count(),
        }

    def describe(self) -> str:
        total = self.param_count()
        return (
            f"context_length={self.context_length}  embed_size={self.embed_size}  "
            f"num_heads={self.num_heads} (head_dim={self.head_dim})  "
            f"num_layers={self.num_layers}  ffn_hidden={self.ffn_hidden}  "
            f"rope_theta={self.rope_theta:g}  dropout={self.dropout}\n"
            f"parameters: {total:,} ({total / 1e6:.2f}M)"
        )


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters, defaulted for one 24 GB GPU (see docs/GPU_TRAINING.md).

    A **step is one micro-batch** forward/backward, not one optimizer update — so
    `steps` and `batch_size` together set the token budget:

        tokens = steps * batch_size * context_length
               = 150_000 * 16 * 2048 = 4.92B

    That is ~14 tokens per parameter against this model's 347.4M — noticeably under
    the Chinchilla-optimal ~20:1 (unlike the 200m sibling this was scaffolded from,
    which sits a bit past it at the same step/batch/context settings copied over
    unchanged). `steps` was not retuned for the larger parameter count — raise it
    (e.g. toward 215,000 for ~20:1, i.e. ~7.05B tokens) before a real run, and get a
    fresh GPU-hours/cost estimate from `gpt-benchmark` + AWS_RUNBOOK.md rather than
    reusing the 200m sibling's 44-55 GPU-hour figure, which does not transfer.
    Changing `batch_size` without changing `steps` silently rescales the whole
    budget.
    """

    # VRAM is architecture- and context-length-dependent, so the 153m sibling's ~10 GB
    # figure does not transfer here (deeper model, 2x context, no attention mask
    # materialised via SDPA either way) — unmeasured. Run `gpt-benchmark --sweep-batch`
    # on the actual instance before committing a multi-hour run; see docs/GPU_TRAINING.md.
    batch_size: int = 16
    grad_accum_steps: int = 4    # effective batch 64 seqs = 131,072 tokens per update
    lr: float = 4e-4             # 2e-4 is conservative at this size/batch (GPT-3 125M: 6e-4)
    min_lr: float = 4e-5
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    steps: int = 150_000         # 4.92B tokens — see the class docstring
    # Each eval is eval_batches*2 forward passes at full batch_size, so it is far more
    # expensive here than in the batch_size=1 sibling projects. 500 keeps it under ~3%
    # of step time; it is telemetry only and safe to change between resumes.
    eval_interval: int = 500
    # `best.pt` is gated on this sample, so its noise decides which checkpoint gets
    # kept. Each eval draws eval_batches * batch_size * context_length tokens per
    # split — 40*16*2048 = 1,310,720 here, vs the 50m sibling's 20*1*1024 = 20,480,
    # whose measured per-eval sigma of ~0.14 let a single lucky draw hold `best.pt`
    # for 60,000+ steps while the true loss kept falling. The 153m sibling's matching
    # 40*16*1024 sample measured sigma near 0.025; this project's eval draws twice
    # the tokens, so sigma should sit lower still (unmeasured here) — for ~5% of wall
    # clock. Raising eval_batches further buys little (sigma only falls as 1/sqrt(n))
    # and the residual pathology is better fixed by smoothing than by more samples.
    eval_batches: int = 40
    # A checkpoint is ~4.2 GB (347.4M params x 4 bytes fp32 weights + 2x4 bytes AdamW
    # moments = 12 bytes/param), so saving every 200 steps would spend more time on
    # I/O than on training. Wall-clock per 2000 steps is unmeasured here (bigger and
    # deeper than the 200m sibling this was scaffolded from) — read the actual
    # steps/sec off `gpt-benchmark` before tuning this further.
    save_every_steps: int = 2_000
    seed: int = 42
    # "auto" = bfloat16 on CUDA, fp32 everywhere else. bf16 (not fp16) because it needs
    # no GradScaler; Ampere/Ada support it, Turing (T4) does not — see docs/GPU_TRAINING.md.
    precision: str = "auto"
    max_new_tokens: int = 80     # demo completion printed at the end of a run
    demo_prompt: str = "The quick brown fox"


# Named sizes. Parameter counts are computed, never hardcoded, so they cannot drift
# out of sync with the architecture.
PRESETS = {
    "tiny": ModelConfig(context_length=256, embed_size=128, num_heads=4,
                        num_layers=4, ffn_hidden=352),
    "small": ModelConfig(context_length=1024, embed_size=512, num_heads=8,
                         num_layers=12, ffn_hidden=1376),
    # This project's own default architecture, derived from ModelConfig's field
    # defaults so the two cannot drift apart.
    "350m": ModelConfig(),
}

DEFAULT_PRESET = "350m"

_ENV_OVERRIDES = {
    "context_length": ("GPT_CONTEXT_LENGTH", int),
    "embed_size": ("GPT_EMBED_SIZE", int),
    "num_heads": ("GPT_NUM_HEADS", int),
    "num_layers": ("GPT_NUM_LAYERS", int),
    "ffn_hidden": ("GPT_FFN_HIDDEN", int),
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


_TRAIN_ENV_OVERRIDES = {
    "batch_size": ("GPT_BATCH_SIZE", int),
    "grad_accum_steps": ("GPT_GRAD_ACCUM", int),
    "lr": ("GPT_LR", float),
    "min_lr": ("GPT_MIN_LR", float),
    "steps": ("GPT_STEPS", int),
    "eval_interval": ("GPT_EVAL_INTERVAL", int),
    "eval_batches": ("GPT_EVAL_BATCHES", int),
    "save_every_steps": ("GPT_SAVE_EVERY", int),
    "precision": ("GPT_PRECISION", str),
}


def resolve_train_config():
    """TrainConfig with any `GPT_*` env overrides applied.

    Exists so the same checkout runs on a laptop and on a rented GPU without editing
    code — the GPU-sized defaults are wrong for a local smoke test, and a smoke test
    that requires a source edit is one people skip.
    """
    overrides = {}
    for field_name, (env_var, cast) in _TRAIN_ENV_OVERRIDES.items():
        raw = os.getenv(env_var)
        if raw is not None:
            overrides[field_name] = cast(raw)
    return replace(TrainConfig(), **overrides) if overrides else TrainConfig()


def load_settings(preset_name=None):
    """One call for everything an entrypoint needs: (model_cfg, train_cfg, paths, label)."""
    model_cfg, label = resolve_model_config(preset_name)
    return model_cfg, resolve_train_config(), Paths(label=label), label
