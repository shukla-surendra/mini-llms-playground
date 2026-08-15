---
language:
- en
license: mit
library_name: pytorch
tags:
- text-generation
- causal-lm
datasets:
- HuggingFaceH4/ultrachat_200k
- OpenAssistant/oasst1
- HuggingFaceTB/smoltalk
- lmsys/lmsys-chat-1m
pipeline_tag: text-generation
---

# custom-gpt — a from-scratch GPT with a configurable size

Part of [mini-llms-playground](../../README.md)'s **from-scratch track** — see the
[top-level README](../../README.md) and [docs index](../../docs/README.md) for how this
relates to the [fine-tuning track](../../fine_tuning/tinyllama-1.1b-lora/README.md).

Trains a GPT-style decoder from zero — custom architecture, custom training loop, no
pretrained weights — on the same corpus as the sibling
[`custom-gpt-153m`](../custom-gpt-153m/README.md) project.

**The model size is a setting, not a rewrite.** The default is ~10M parameters for fast
laptop iteration, but the same code trains anything from ~7M to ~153M by changing one
environment variable. Everything downstream — parameter count, checkpoint location, the
model itself — follows automatically.

```bash
make presets            # see every size and its exact parameter count
GPT_PRESET=30m make train
```

## Quickstart

```bash
make setup              # uv sync: create .venv, install the package
make config             # what will I train? exact parameter count, no guessing
make data-public        # build the corpus (4 public datasets, no HF token needed)
make train              # train (Ctrl-C is safe — it saves a resumable checkpoint)
make infer              # generate from the trained checkpoint
make serve              # FastAPI server on :8000
```

`make data` also includes the gated LMSYS-Chat-1M set and needs `HF_TOKEN`; see
[docs/DATASETS.md](docs/DATASETS.md).

> Ctrl-C during `make train` is safe — see
> [Start, stop, and resume training](#start-stop-and-resume-training) below.

## Choosing a model size

```
preset             params    ctx  embed  heads  layers
--------------------------------------------------------
tiny            7,259,008    256    128      4       4
10m             9,979,040    512    160      8       6
30m            30,142,848    512    384      6       6
50m            51,475,968   1024    512      8       8
153m          152,791,296   1024    768     12      16
```

```bash
GPT_PRESET=30m make train                        # a named preset
GPT_EMBED_SIZE=192 GPT_NUM_LAYERS=8 make train   # or override individual fields
```

Overrides get their own label (`custom-e192-l8-h8-c512`) and their own checkpoint
directory, so switching sizes never overwrites another model's weights. Invalid
combinations are rejected up front with an explanation rather than failing deep inside
the model:

```
ValueError: embed_size (100) must be divisible by num_heads (8) — each head gets
embed_size/num_heads dimensions.
```

The counts above are computed from the architecture, not hardcoded, and are verified
exact against instantiated models. The `153m` preset reproduces the sibling project's
architecture precisely (152,791,296 parameters).

### Where the parameters go

At small sizes the token embedding dominates, because the GPT-2 vocabulary (50,257
tokens) is fixed regardless of how small the model gets:

```
token_embedding         8,041,120  ( 80.6%)
position_embedding         81,920  (  0.8%)
transformer_blocks      1,855,680  ( 18.6%)
final_layernorm               320  (  0.0%)
total                   9,979,040
```

That is why shrinking `num_layers` below the `10m` preset barely moves the total — to go
meaningfully smaller you need a smaller vocabulary, not fewer layers. `make config` prints
this breakdown for whatever size you have selected.

## Choosing an attention kernel

```bash
ATTN_IMPL=sdpa make train    # F.scaled_dot_product_attention — the default
ATTN_IMPL=naive make train   # explicit nn.MultiheadAttention — the original implementation
```

Same math, different memory-access pattern — see
[Chapter 25 — Efficient Attention: Flash Attention and SDPA](../../docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md)
for the mechanism. `sdpa` is the default (measured faster than `naive` for this project's
sizes on Apple Silicon MPS — see [docs/TRAINING_SCHEDULE.md](docs/TRAINING_SCHEDULE.md));
`naive` remains available as an explicit opt-out, e.g. to reproduce results predating the
switch.

Resuming across a *different* `ATTN_IMPL` than the checkpoint was saved with remaps the
attention weights automatically (same values, different parameter names) — no progress
lost, no flag needed; see `checkpoint.remap_attn_impl`. The trainer prints which kernel is
actually active at the start of every run (`attn_impl=...` in the startup line), and if a
remap happens it prints that too — always check this line rather than assuming, especially
after resuming a run that predates a switch.

## Layout

```
src/gpt/
├── config.py          # every knob: presets, hyperparameters, paths
├── model.py           # the architecture (TinyGPT and its parts)
├── checkpoint.py      # atomic save/load; checkpoints carry their own architecture
├── runtime.py         # device selection (cuda / mps / cpu)
├── data/
│   ├── sources.py     # the dataset registry — what we train on
│   ├── prepare.py     # download, parse, filter, split
│   ├── dataset.py     # tokenize, batch, loss
│   ├── prompts.py     # held-out prompt loading
│   └── audit.py       # corpus quality gate
├── training/trainer.py
├── inference/
│   ├── generate.py    # sampling + generation loop (one implementation, shared)
│   └── server.py      # FastAPI app
├── evaluation/quality.py
└── cli/               # thin argparse entrypoints -> console scripts
```

One model, assembled from its parts — `TinyGPT` is the only class you construct:

```
TinyGPT                       the model you train and talk to
 ├─ token_emb / pos_emb
 ├─ blocks: GPTBlock × N
 │   ├─ CausalSelfAttention    tokens look at earlier tokens
 │   └─ MLP                    each token processed independently
 └─ ln_f + lm_head             final norm → next-token logits
```

Console scripts (`gpt-config`, `gpt-data`, `gpt-audit`, `gpt-train`, `gpt-infer`,
`gpt-serve`, `gpt-eval`) are what the `make` targets call; use them directly for finer
control, e.g. `uv run gpt-infer --prompt "The quick brown fox" --max-new-tokens 100`.

## Training objective: raw, not instruction-tuned

Training is plain next-token prediction over **every** token — the same objective that
pretrains a base model like GPT-2. There is no chat template and no per-turn loss
masking, even though the corpus contains chat transcripts.

The consequence is worth understanding: this produces a **base model**, not a turn-taking
assistant. Prompted with `User: how do I...`, it continues the transcript and may write
the next `User:` turn itself, because that is what the training text does. Making it
behave like an assistant would require instruction tuning as a separate stage.

## Data

Five datasets — UltraChat 200k, OASST1, Dolly 15k, SmolTalk, and (gated) LMSYS-Chat-1M —
merged through schema-aware parsing and quality filters into `data/train.txt`.

Full detail, including per-dataset licensing, the filter thresholds and why each exists,
and the complete raw-data-to-`train.txt` pipeline: **[docs/DATASETS.md](docs/DATASETS.md)**.

```bash
gpt-data --list     # the registry, printed from code
make audit          # corpus quality gate before a long run
```

## Start, stop, and resume training

```bash
make train                     # start (or resume, if a checkpoint already exists)
```

```bash
# stop: press Ctrl-C at any time
```
`gpt-train` catches the interrupt, saves `checkpoints/<label>/latest.pt`, and exits —
`make` then prints `Error 130`, which is just the interrupt signal propagating through
`make`, not a failed run. The checkpoint from the step you stopped at is already safely
on disk before that message appears.

```bash
make train                     # resume: re-run the same command, picks up from latest.pt
```
Resuming is the *default* behavior of `make train` — it happens automatically whenever
`checkpoints/<label>/latest.pt` exists, no flag needed. The resumed run also verifies the
checkpoint's saved architecture matches the current config before loading, so resuming
after an accidental preset/override change fails loudly instead of silently corrupting
the run.

```bash
make train-fresh               # start over, ignoring any existing checkpoint
# or, equivalently:
RESUME_TRAINING=0 make train
```
Use this when you deliberately want to discard progress and retrain from step 0 under the
same label — otherwise `make train` always continues where the last run left off.

See [Chapter 27 — Checkpointing and Resuming Training](../../docs/llm-engineering/27_checkpointing_and_resuming_training.md)
for why this is safe (atomic saves, self-describing checkpoints) and
[docs/MIGRATION.md](docs/MIGRATION.md) for resuming on a *different* machine.

### Running in the background

`make train` runs in the foreground and ties up the terminal for the whole run. Use
`make train-bg` to start it detached instead — same env-var overrides as `make train`
(`GPT_PRESET`, `ATTN_IMPL`, etc.) apply unchanged:

```bash
make train-bg                              # detached, using whatever ATTN_IMPL/GPT_PRESET defaults apply
GPT_PRESET=30m ATTN_IMPL=sdpa make train-bg # or override the same way you would for `make train`
```

```bash
make train-status   # is it running? PID + the last progress line
make train-stop      # stop it gracefully (SIGINT — saves checkpoints/<label>/latest.pt, same as Ctrl-C)
make train-logs       # tail -f the live output
```

`make train-bg` refuses to start if a `gpt-train` process is already running — see
"Only one run at a time" below for why this guard exists. `make train-stop` sends
`SIGINT`, not `SIGKILL`/`kill -9`, specifically because `SIGKILL` skips the interrupt
handler that saves `latest.pt`.

Under the hood, `train-bg` is exactly:

```bash
nohup uv run gpt-train > logs/train_stdout.log 2>&1 &
```

— `nohup` so it survives the terminal closing, output redirected to
`logs/train_stdout.log` instead of lost. If you're driving this from a script/agent rather
than an interactive shell and want it fully detached from the current shell's job table
too, add `disown` right after.

### Preventing the machine from sleeping mid-run

`nohup`/`make train-bg` keeps the *process* alive if the terminal closes — it does
nothing to stop the *machine* from sleeping. This matters more than it sounds like:
`total_training_hours` (`logs/train_eval_history_<label>.csv`, and `tqdm`'s `total_h`) is
computed from wall-clock time (`time.time()` deltas in `training/trainer.py`'s
`elapsed()`), which cannot tell "the GPU/CPU was actually computing" apart from "the
machine was asleep and this background process got almost no scheduling time." A laptop
left to sleep overnight mid-run doesn't cleanly pause — it typically keeps the process
alive but severely throttled (a handful of steps trickling through every several
minutes instead of a hard stop), and every minute of that gets silently counted as
"training time" in the log, right alongside minutes that were genuinely at full speed.
This is a real, measured failure mode this project hit, not a hypothetical one — a
`~30 steps/sec` run near-stalled to a fraction of a step/sec for several hours overnight,
with `total_training_hours` reporting the full elapsed duration as if it had all been
productive.

The fix is to explicitly keep the machine awake for the duration of the run — every OS
has a way to do this, scoped to just the training process so nothing needs to be manually
undone afterward:

**macOS** — `caffeinate`, built in, no install needed:

```bash
caffeinate -i uv run gpt-train                                       # foreground
caffeinate -i nohup uv run gpt-train > logs/train_stdout.log 2>&1 &   # background
```

`-i` prevents idle sleep (the display can still turn off; the machine won't suspend).
Wrapping the command ties the awake-request to it directly — `caffeinate` exits the
moment `gpt-train` does, automatically releasing the sleep-prevention with nothing to
remember to undo.

**Linux** — `systemd-inhibit`, built in on any systemd-based distro (which covers
essentially every modern desktop distro — Ubuntu, Fedora, Debian, and derivatives):

```bash
systemd-inhibit --what=idle:sleep --why="gpt-train run" uv run gpt-train
```

Same scoping behavior as `caffeinate` — the inhibitor lock is held only while the wrapped
command runs. This mainly matters on a Linux **laptop/desktop**; a headless remote GPU
box (the other half of [`docs/MIGRATION.md`](docs/MIGRATION.md)'s workflow) generally
isn't configured to suspend on idle in the first place, so this is rarely needed there.

**Windows** — no single built-in CLI equivalent; two real options:

```powershell
# Option 1: Microsoft PowerToys' Awake module (free, official, closest match to
# caffeinate — keeps the system awake exactly as long as a given process ID is alive)
awake.exe --pid <gpt-train's PID>

# Option 2: no extra install, but NOT auto-scoped — you must remember to revert it
# after the run, unlike the two options above:
powercfg /change standby-timeout-ac 0    # before starting training
powercfg /change standby-timeout-ac 30   # revert afterward (30 = whatever it was before)
```

PowerToys' `Awake` is the closer match in behavior (tied to the process, not a manual
system-wide setting) and is Microsoft-maintained — prefer it over the `powercfg` fallback
when it's available.

### Only one run at a time

`make train`, `make train-fresh`, and `make train-bg` all refuse to start if a `gpt-train`
process is already running (checked via `pgrep`) — this isn't a style preference, it's a
direct response to a real, measured incident on this project: two `gpt-train` processes
resumed from the same checkpoint and left running concurrently don't fail loudly, they
silently race to write `checkpoints/<label>/latest.pt` and `best.pt`, each getting
roughly half the GPU (a real, measured ~2x throughput drop when this happened once here),
and whichever process saves last silently wins, discarding the other's progress. If you
ever bypass `make` and invoke `uv run gpt-train` / `nohup ...` directly, that guard
doesn't apply — check `make train-status` first.

## Checkpoints

Namespaced per model size, so multiple sizes coexist:

```
checkpoints/<label>/
├── best.pt       # lowest test loss
├── latest.pt     # periodic, for resuming
├── serving.pt    # what the API serves
└── final.pt      # end of a completed run
```

Every checkpoint stores its own architecture (`embed_size`, `num_layers`, …), so loading
never requires telling the code what size it was — and resuming with a mismatched config
is detected and refused rather than silently corrupting a run.

Saves are atomic (write to `.tmp`, then rename), so interrupting training cannot leave a
truncated checkpoint behind.

## Monitoring a long run

- `logs/train_eval_history_<label>.csv` — train/test loss, perplexity, tokens, wall-clock,
  appended every `eval_interval` steps.
- `make eval` — heuristic generation-quality report (empty output, repetition, ASCII
  noise, role leakage) appended to `logs/quality_history_<label>.jsonl`, with a delta
  against the previous run.

Loss dropping while the quality score falls usually means data-format noise is hurting
generation — check `make audit`.

**"Is `steps: int = 1_000_000` the right target, and should I raise it?"** — see
[docs/TRAINING_SCHEDULE.md](docs/TRAINING_SCHEDULE.md): what `step` actually counts here
(a micro-batch, not an optimizer update), how it drives the warmup/cosine LR schedule, and
a three-question framework for telling a real plateau from an artifact of where you are on
the schedule or noisy eval sampling.

**"Does training for multiple epochs help?" / "Does arranging data in a particular way
increase model performance?"** — see [docs/TRAINING_QA.md](docs/TRAINING_QA.md): this
run's actual epoch math (≈2.95 epochs at the configured step budget), and why storage
order in `train.txt` has zero effect on training (random-window sampling) while the
`"\n\n"` conversation-boundary separator is a real, currently weak spot.

## Other docs

- [`../../docs/llm-engineering/`](../../docs/llm-engineering/00_roadmap.md) — the
  from-first-principles curriculum every concept below links back to
- [docs/DATASETS.md](docs/DATASETS.md) — every dataset, filter, and the corpus pipeline
- [docs/API_SERVER.md](docs/API_SERVER.md) — serving endpoints
- [docs/LLM_DEV_GUIDE.md](docs/LLM_DEV_GUIDE.md) — quickstart map: which curriculum
  chapter covers each pipeline stage, plus this project's exact command for each
- [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) — a file-by-file tour of `src/gpt/`
  in execution order, explaining *why* each module is built the way it is, not just what
  it does
- [docs/MODEL_SIZING_GUIDE.md](docs/MODEL_SIZING_GUIDE.md) — every `ModelConfig` field:
  what it costs in real parameters/compute, its hard limitations, and what value fits
  which use case
- [docs/MIGRATION.md](docs/MIGRATION.md) — moving a run between a GPU box and a laptop
- [docs/TRAINING_SCHEDULE.md](docs/TRAINING_SCHEDULE.md) — what `steps` means, the LR
  schedule, and judging whether a longer run still helps
- [docs/TRAINING_QA.md](docs/TRAINING_QA.md) — running log of specific questions asked
  while training this project's model, answered against its actual code and numbers
- [docs/DATA_PREP_GUIDELINE.md](docs/DATA_PREP_GUIDELINE.md) — ranked checklist for
  preparing a domain-specialized corpus for maximum quality-per-parameter, including the
  tokenizer-vocab lever this project's own parameter breakdown motivates
- [docs/BOOKS_CORPUS_INTEGRATION.md](docs/BOOKS_CORPUS_INTEGRATION.md) — running log of
  enriching the corpus with book-derived text via `tools/corpus-extractor`, including the
  boundary-marking and mix-ratio issues that make this more than "run the tool and merge"
