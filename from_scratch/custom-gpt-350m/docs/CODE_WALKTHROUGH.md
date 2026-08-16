# Code Walkthrough: Why This Is Built the Way It Is

The top-level [`README.md`](../README.md) has the *what* — the module tree and the
one-paragraph architecture diagram. The [LLM Engineering Curriculum](../../../docs/llm-engineering/00_roadmap.md)
(via [`LLM_DEV_GUIDE.md`](LLM_DEV_GUIDE.md)'s stage map) has the *general mechanism* —
why cross-entropy, why AdamW, why attention at all. This doc is neither of those: it's a
walk through this project's **own actual files**, in the order data and control really
flow through them, explaining the specific design decision each one embodies and why it
was made that way instead of some simpler or more obvious alternative. Where the general
mechanism is curriculum territory, this doc links out rather than re-deriving it — every
section here is about a choice *specific to this codebase*.

Read top to bottom for the full pipeline story, or jump to one file's section directly —
each is self-contained.

## `config.py` — every knob, computed, not hardcoded

**What**: `ModelConfig` (architecture), `TrainConfig` (hyperparameters), `Paths`
(filesystem layout), and `PRESETS` (named size configs from `tiny` to `153m`) — one
`dataclass` each, all frozen (immutable).

**Why frozen dataclasses instead of a plain dict or a YAML file**: a frozen dataclass
gets you three things a dict doesn't — `__post_init__` validation runs automatically on
construction (catching e.g. `embed_size % num_heads != 0` the moment a bad config is
built, not deep inside `model.py` on the first forward pass), attribute typos become
`AttributeError`s instead of silent `None`s from a dict `.get()`, and immutability makes a
`ModelConfig` safe to pass around and reuse without one caller accidentally mutating it
under another's feet.

**Why `param_count()` is a method on `ModelConfig`, computed from the same four numbers
that build the real model, rather than a number typed into `PRESETS` or `README.md`
directly**: a hardcoded count silently drifts the moment someone tweaks `num_layers` and
forgets to update it elsewhere. Computing it from formula means the number in `make
config`'s output and the number an instantiated `TinyGPT` actually reports
(`model.param_count()` in `model.py`) are mechanically guaranteed to agree — the project's
tests can (and do) assert they match exactly, which is a much stronger guarantee than
"someone remembered to update the docstring."

**Why `resolve_model_config()` builds a *label* (`custom-e192-l8-h8-c512`) for any
non-preset override combination, not just for named presets**: this label is what
`Paths.checkpoint_dir` is namespaced under. Without it, training with
`GPT_EMBED_SIZE=192` would write into the same `checkpoints/10m/` directory the real `10m`
preset uses, silently corrupting it with a shape mismatch on the next resume. Deriving the
label mechanically from whatever fields actually changed means every distinct
architecture gets its own checkpoint directory automatically — nobody has to remember to
pick a unique name.

**What each `ModelConfig` field costs, and how to pick a value for it** is its own,
longer question than "why is this a dataclass" — see
[`docs/MODEL_SIZING_GUIDE.md`](MODEL_SIZING_GUIDE.md) for `context_length`/`embed_size`/
`num_heads`/`num_layers`/`dropout`/`vocab_size` individually, with real parameter/compute
numbers derived from this project's own `param_count()` formula.

## `data/sources.py` — the dataset registry, not a list buried in a script

**What**: `DATASETS`, a tuple of `DatasetSource` entries (`hf_id`, `schema`,
`license_note`, etc.) — the single list both `data/prepare.py`'s download step and
[`docs/DATASETS.md`](DATASETS.md) read from.

**Why a registry instead of hardcoding five `snapshot_download()` calls in
`prepare.py`**: adding, removing, or gating a source becomes a data change (append a
`DatasetSource`), not a code change to the download/parse logic — `build_corpus()` in
`prepare.py` never mentions a specific dataset by name, only iterates `selected(...)`.

**Why `license_note` is a required field on every entry, not an afterthought**: this
project trains on data that is itself, in part, model-generated (`UltraChat 200k`'s own
summary notes it was built by two ChatGPT instances conversing) — its `license_note`
("MIT. Derived from OpenAI model outputs — check OpenAI terms for commercial use.")
exists because the dataset's own license and the licensing status of a commercial
provider's *outputs* inside that dataset are two different questions, and burying that
distinction in a README paragraph nobody reads before running `make data` is exactly how
it gets missed. See
[Chapter 33](../../../docs/llm-engineering/33_distilling_production_models_into_a_local_model.md)
for the fuller version of why this matters.

## `data/prepare.py` — one parser for five differently-shaped sources

**What**: downloads each source's parquet shards, parses rows into `(role, text)` turns,
filters for quality, shuffles, splits 90/10, writes `train.txt`/`test.txt`/`test_prompts.txt`.

**Why `extract_turns()` tries a "conversation" schema then falls back to an "instruction"
schema, instead of five source-specific parsers**: the five sources arrive in genuinely
two raw shapes — a list of role-tagged messages (UltraChat, OASST1, LMSYS) or flat
instruction/input/output columns (Dolly). Two schema parsers cover five sources because
the *shape* of the data, not the dataset identity, is what determines how to parse it —
adding a sixth source in either shape needs zero new parsing code.

**Why `is_quality_text()` checks printable-ratio, ASCII-ratio, alpha-density, and
placeholder markers all before a single token gets tokenized**: every one of these is a
cheap, source-agnostic proxy for "this text would teach the model noise, not language."
Catching this at parse time — before it ever reaches `train.txt` — is far cheaper than
discovering it via `make audit`'s `noise_line_rate` after a full corpus build, and far
cheaper than discovering it via generation quality after a full training run. The
specific placeholder check (`NAME_`, `PERSON_`, `EMAIL_`, `URL_`) exists because LMSYS
redacts real user entities into exactly these tokens — left in, they leak into
generations as literal nonsense the model has no way to know is a redaction artifact
rather than real vocabulary.

**Why `random.shuffle(all_conversations)` happens once, across all five sources pooled
together, before the train/test split — not per-source or not at all**: see
[`TRAINING_QA.md`'s full answer](TRAINING_QA.md#does-arranging-data-in-a-particular-way-increase-model-performance)
for the mechanism; short version, this is what prevents any single source from dominating
a contiguous stretch of the corpus, which matters even under `dataset.py`'s random-window
sampling (see below) because it's what determines *which* source a given window is drawn
from.

**Why conversations are joined with a plain `"\n\n"`** (`build_corpus()`'s
`"\n\n".join(...)`) rather than a dedicated boundary token: this is a known, currently
unfixed weak spot, not a considered trade-off — see
[`DATA_PREP_GUIDELINE.md`](DATA_PREP_GUIDELINE.md#4-a-strong-document-boundary-token-from-the-start)
for why a real reserved token would be the correct fix, deliberately deferred here because
the corpus already exists and changing the tokenizer's vocabulary is a one-way door (same
doc, "The one-way door worth naming explicitly").

## `data/audit.py` — a five-second gate before a multi-hour run

**What**: `make audit` computes noise-line rate, ASCII ratio, role-line counts, and
train/test leakage on the already-built `train.txt`/`test.txt` — no model, no training.

**Why leakage is measured as *exact assistant-line overlap*, not conversation-level or
fuzzy overlap**: an assistant turn appearing verbatim in both `train.txt` and `test.txt`
means the test split isn't actually testing generalization for that example — the model
may simply have memorized it. Checking at the line level (not the whole-conversation
level) catches the common case where the same canned assistant answer (e.g. a boilerplate
"I can't help with that" refusal) appears attached to many different user turns across
both splits — a real leakage source a conversation-level check would miss entirely.

**Why this exists as a separate, cheap, pre-training gate rather than just watching the
loss curve during training**: a corpus that's mostly non-ASCII noise or redaction
placeholders will still produce a loss curve that goes down — cross-entropy loss falling
means the model is learning to predict *something* consistently, and n-gram-level noise is
exactly the kind of easy, learnable-but-worthless pattern that happily lowers loss without
producing anything useful. `make audit` catches this in seconds, before hours of compute
are spent confirming it the expensive way.

## `data/dataset.py` — random windows, not a sequential walk

**What**: `encode_raw()` tokenizes the whole corpus into one flat token tensor;
`get_batch()` draws random `context_length`-token windows from it; `next_token_loss()` is
plain cross-entropy over every position.

**Why `get_batch()` samples a uniformly random starting index every call, instead of
walking sequentially through the corpus in order**: this is the single fact that makes
physical file order in `train.txt` irrelevant to training (per `TRAINING_QA.md` above) —
`ix = torch.randint(0, max_start, (batch_size,))` means every training step sees an
independently-chosen slice, so there's no notion of "where in the file training currently
is" for row order to affect. It also means a single epoch's "coverage" is statistical, not
guaranteed — some tokens get sampled more than once before others are sampled at all,
which is the standard, accepted trade-off for the implementation simplicity this buys
(no shuffled-index-queue bookkeeping across epoch boundaries).

**Why `next_token_loss()` is unconditional cross-entropy over every position, with no
mask for "this was a prompt token, not a response token"**: this is what makes the
model a raw base model rather than an instruction-tuned one — a deliberate simplicity
choice for this project's current stage, with a real, named consequence (see README's
"Training objective: raw, not instruction-tuned" section, and
[`DATA_PREP_GUIDELINE.md`'s note](DATA_PREP_GUIDELINE.md#what-consistent-structure-looks-like-concretely)
on why this matters even more for a future distillation-style corpus, where you'd want
the loss to land only on the teacher's response, not the prompt fed back into the model).

## `model.py` — the architecture, entirely config-driven

**What**: `TinyGPT` = token/position embeddings → `N` × `GPTBlock` (attention + MLP,
pre-norm residual) → final norm → tied output head.

**Why every dimension is threaded through from `ModelConfig` with nothing hardcoded**:
this is what makes `GPT_PRESET=153m make train` a config change instead of a code change
— the same `model.py` produces a 7M-parameter model or a 153M-parameter one, and the
`param_count()` guarantee from `config.py` above only holds because there's no dimension
here that could silently diverge from what the config declares.

**Why `self.lm_head.weight = self.token_emb.weight` (weight tying)**: at this model's
scale the token embedding is *most* of the parameter budget — 8,041,120 of 9,979,040
total at the `10m` preset (80.6%, per `make config`'s breakdown) — because the GPT-2
vocabulary (50,257 tokens) is fixed regardless of model size. Tying the input embedding
and output projection halves that single largest cost, and it's theoretically motivated,
not just a memory hack: a token's *input* representation (how it's embedded) and its
*output* representation (how likely it is to be predicted) should reasonably move
together — a token similar in meaning to another should be both easy to confuse as input
and easy to confuse as a predicted output.

**Why `attn_impl` is switchable at all, between `"naive"` (`nn.MultiheadAttention`) and
`"sdpa"` (`F.scaled_dot_product_attention`)**: same math, different memory-access
pattern — see [Chapter 25](../../../docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md)
for the general mechanism. This project measured a real, reproducible ~7% per-step
speedup from `sdpa` on Apple Silicon MPS at this model's size (see
[`TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md)), which is why it's the current default —
but the switch itself exists as a real, working option (not a one-way migration) because
attention-kernel performance is hardware- and version-dependent, and a config that was
faster on one machine/PyTorch version isn't guaranteed to stay faster forever. The two
implementations produce numerically-equivalent weights under different parameter names —
see `checkpoint.py`'s `remap_attn_impl` below for why that split needed its own function.

## `checkpoint.py` — self-describing, crash-safe, migration-safe

**What**: `atomic_save()` (write-then-rename), `make_payload()` (weights + full
architecture description bundled together), `is_compatible()` (architecture match check),
`remap_attn_impl()` (attention parameter key translation).

**Why `atomic_save()` writes to a `.tmp` file and renames, instead of `torch.save()`
directly to the real path**: `KeyboardInterrupt` (or any crash) can land at essentially
any point in Python bytecode, including mid-way through serializing a multi-hundred-MB
tensor — a direct `torch.save()` interrupted there leaves a truncated, unreadable
checkpoint file *at the path resuming code expects to find a good one*. Writing to a
sibling `.tmp` and using `Path.replace()` (an atomic filesystem rename) means any reader
sees either the complete old file or the complete new one, never a partial write — this is
the concrete mechanism behind "Ctrl-C is safe" everywhere else in this project's docs.

**Why a checkpoint stores its own architecture (`embed_size`, `num_layers`, `attn_impl`,
...) instead of only weights**: this is what makes `load_model()` able to rebuild the
*exact* model a checkpoint came from without being told its size externally, and what
makes `is_compatible()` able to refuse a mismatched resume loudly (wrong shapes) instead
of `load_state_dict` failing with a much less diagnosable low-level tensor-shape error, or
worse, partially succeeding into a corrupted model.

**Why `remap_attn_impl()` exists as a dedicated, tested function rather than expecting
users to always resume under the same `attn_impl` they last trained with**: it turned a
real incident (resuming under a different `ATTN_IMPL` than the checkpoint was trained
with) from "silently fails to load, or worse, loads garbage into mismatched parameter
slots" into "detected automatically, remapped correctly, logged so it's visible." The
remap itself is a pure key rename — `nn.MultiheadAttention`'s internal `in_proj_weight` is
already the exact same `(3*embed_size, embed_size)` matrix as a fused
`nn.Linear(embed_size, 3*embed_size).weight`, just addressed under a different module
path — so no reshape or transpose is needed, only knowing which name maps to which.

## `training/trainer.py` — the loop, and what makes resuming honest

**What**: `lr_for_step()` (warmup + cosine decay), the main loop (forward → loss →
backward → grad-accumulate → optimizer step → periodic eval/save), `_resume_into()`.

**Why a `step` is one micro-batch forward/backward pass, not one optimizer update** —
`grad_accum_steps=32` micro-batches accumulate gradients before a single
`optimizer.step()` fires: at `batch_size=1` (chosen for MPS/laptop VRAM headroom, per the
config docstring), a single example's gradient is noisy; accumulating 32 of them before
updating approximates the stability of a real batch size of 32 without ever holding 32
examples' activations in memory at once. See
[`TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md) for the full consequence this has for what
`TrainConfig.steps` actually means in wall-clock/epoch terms.

**Why `lr_for_step()` is warmup-then-cosine, and why it's computed fresh every step from
`train_cfg.steps` rather than a fixed schedule**: warmup exists because the model's
random initial weights produce large, noisy gradients — taking full-size steps
immediately can destabilize training before it's found any structure at all; cosine decay
exists so updates get finer as training approaches convergence rather than continuing to
take max-sized, potentially overshooting steps forever. Deriving both phases' lengths from
`train_cfg.steps` (not hardcoded step counts) means changing the total step budget
reshapes the *whole* schedule proportionally, not just where training happens to stop —
see [Chapter 3's deep-dive](../../../docs/llm-engineering/03_how_neural_networks_learn.md#deep-dive-what-the-learning-rate-schedule-is-actually-doing)
for the general mechanism.

**Why `_resume_into()` checks `is_compatible()` before loading, and separately checks
`attn_impl` before deciding whether to remap**: these are two independent failure modes
with two independent guards — a genuine architecture mismatch (wrong `embed_size`) should
fail loudly and refuse to resume; an `attn_impl` mismatch is *not* actually incompatible
(same weights, different key names) so it's handled by fixing the keys rather than
refusing to proceed. Conflating these would either wrongly block a safe resume or wrongly
allow an unsafe one.

**What this file deliberately does *not* guard against**: two `gpt-train` processes
resumed from the same checkpoint and run concurrently — nothing here detects that,
because it's a process-level concern, not a within-process one. That guard lives in the
`Makefile` (`train`/`train-fresh`/`train-bg` all `pgrep` for an existing `gpt-train`
process first) precisely because this project hit that failure mode for real once — see
the "Only one run at a time" section of the top-level [`README.md`](../README.md).

## `inference/generate.py` — one generation implementation, used everywhere

**What**: `apply_repetition_penalty()`, `sample_next_token()` (temperature/top-k/top-p),
`generate_text()` — the autoregressive loop every caller (CLI, server, eval) goes through.

**Why this is one shared function instead of the CLI, the server, and the evaluator each
implementing their own generation loop**: sampling behavior — exactly how temperature,
top-k, and top-p interact — is subtle enough that three independent implementations would
inevitably drift, and a drift here is invisible in code review (it only shows up as
"the API server's output feels different from `make infer`'s for the same prompt/seed").
One function, three call sites, means there's structurally nothing to drift.

**Why top-k is applied *before* converting to probabilities, and top-p *after*
`softmax`**: top-k is a coarse, cheap hard cutoff on raw logits (keep the k largest,
`-inf` everything else) — applying it before `softmax` means the excluded tokens
contribute exactly zero probability mass, not a vanishingly small one. Top-p then
operates on the *already-normalized* probability distribution because "smallest set of
tokens whose cumulative probability exceeds `p`" is a statement about probabilities, not
raw logits — it has to run after normalization to mean what it claims to mean.

**Why `generate_text()` takes a `postprocess` flag instead of always trimming at role
markers (`"\nUser:"`, etc.)**: `evaluation/quality.py`'s `role_leak_rate` metric
specifically needs to *see* a leaked `"\nUser:"` continuation in order to count it — if
`generate_text()` always silently trimmed that away, the one metric designed to catch
"model is bleeding into the next turn instead of stopping" would always read zero,
regardless of whether the model actually does this. The flag exists so the same function
serves both "give me a clean completion" (CLI/server) and "give me the raw, unfiltered
output so I can measure how often it misbehaves" (evaluation) without duplicating the
generation loop for either.

## `inference/server.py` — HTTP as a thin layer over the same package

**What**: a FastAPI app with `/health` and `/generate`, loading one checkpoint at startup
and holding the model in `app.state`.

**Why the server contains essentially no model logic of its own** — `create_app()` calls
straight into `checkpoint.load_model()`, `checkpoint.resolve_serving_checkpoint()`, and
`generate.generate_text()`, the exact same functions the CLI uses: this is the same
"one implementation, multiple callers" principle as `generate.py` above, applied one
layer up — the HTTP layer's only job is translating a request into a function call and a
response, so a bug can't exist *only* in the server's copy of generation logic, because
there is no server's copy.

**Why `trim_at_role_markers` defaults to `False` in the API, when the CLI and eval
both use `postprocess=False`/raw output for their own reasons too**: this model is a raw
base model, not a chat-tuned assistant (see `data/dataset.py` above) — it *continues*
text rather than *answering* a turn and stopping. Defaulting to no trimming is the honest
default for what the model actually does; a caller who wants chat-style trimming opts in
explicitly, rather than the API quietly reshaping raw continuations into something that
looks more chat-like than the underlying model actually is.

## `evaluation/quality.py` — catching what a loss number can't see

**What**: generates completions for a fixed prompt set, scores them on non-empty rate,
repetition, ASCII ratio, role leakage, and placeholder noise, combines them into one
`heuristic_quality_score_0_to_100`, and tracks deltas against the previous run.

**Why this exists as a separate signal from `test_loss` at all**: loss falling means the
model is getting better at predicting the *specific* next tokens in the held-out set — it
does not directly measure whether generated *text* is any good. A model can lower loss
while generation quality genuinely degrades (e.g., collapsing toward a narrow, repetitive
mode that happens to score well on average next-token probability) — this is exactly the
"loss improved but generations got worse" case the module's own docstring names, and it's
invisible to `train_eval_history_<label>.csv` alone.

**Why the score is a hand-weighted combination of five cheap heuristics instead of a
single "better" metric**: each heuristic catches a different, specific failure mode a
small from-scratch model is actually prone to — empty output, degenerate repetition,
non-ASCII noise, leaking into the next conversational turn, and redaction-placeholder
artifacts bleeding through from the corpus (`data/prepare.py`'s own filters, imperfectly
enforced). None of these needs a reference answer or a second model to compute, which is
what makes this cheap enough to run after every `make train` cycle rather than being a
rare, expensive evaluation event.

## `cli/` — thin argparse wrappers, nothing else

**What**: one file per console script (`gpt-train`, `gpt-infer`, `gpt-config`, ...), each
parsing arguments and calling straight into the package (e.g. `cli/train.py` is ~20 lines:
parse `--preset`/`--no-resume`, call `config.load_settings()` then `training.train()`).

**Why business logic never lives in `cli/`**: every one of these functions
(`load_settings`, `train`, `generate_text`, `audit`, ...) is directly importable and
testable without going through `argparse` or a subprocess at all — the CLI layer's only
job is translating command-line flags/env vars into a function call, the same "thin
adapter over one real implementation" principle `server.py` follows for HTTP. It's also
why `Makefile` targets are one-line `uv run gpt-<x>` calls: there's no meaningful logic
for `make` to own either.

## `runtime.py` — one function, extracted because it was copy-pasted everywhere

**What**: `get_device()` — `cuda` (with TF32 enabled) → `mps` → `cpu`, in that order.

**Why this earns its own file for one function**: per its own docstring, this exact
device-selection logic was previously duplicated, identically, into every entrypoint
script — a single shared function means a future change (e.g. a new backend, or a
device-specific flag) happens once, not once per script with the risk of one copy
silently falling out of sync with the others.

## Tracing one full run, file by file

```
make data-public
  -> data/sources.py   (which datasets)
  -> data/prepare.py   (download, parse, filter, shuffle, split -> train.txt/test.txt)
  -> data/audit.py     (make audit: sanity-gate the result before spending compute)

make train
  -> config.py          (resolve preset + env overrides -> ModelConfig/TrainConfig/Paths)
  -> data/dataset.py     (tokenize train.txt/test.txt once, into flat token tensors)
  -> model.py             (build TinyGPT from ModelConfig)
  -> checkpoint.py         (resume: load + is_compatible()/remap_attn_impl() if needed)
  -> training/trainer.py    (the loop: get_batch -> forward -> loss -> backward -> step)
  -> checkpoint.py           (atomic_save: latest.pt / best.pt / final.pt)
  -> evaluation/quality.py    (make eval: heuristic score against inference/generate.py)

make infer / make serve
  -> checkpoint.py      (load_model: rebuild the exact architecture from the checkpoint)
  -> inference/generate.py OR inference/server.py  (same generate_text() either way)
```

Every arrow above is a real import, not a conceptual one — this is the actual call graph,
which is what makes it possible to unit-test any single stage (e.g. `data/prepare.py`'s
parsing) without needing a trained model, or a GPU, or the other stages to exist yet.
