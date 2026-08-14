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

> Stopping training with Ctrl-C makes `make` print `Error 130`. That is just the
> interrupt propagating through make — the checkpoint is saved and `make train` resumes
> from it.

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

## Other docs

- [docs/DATASETS.md](docs/DATASETS.md) — every dataset, filter, and the corpus pipeline
- [docs/API_SERVER.md](docs/API_SERVER.md) — serving endpoints
- [docs/LLM_DEV_GUIDE.md](docs/LLM_DEV_GUIDE.md) — end-to-end walkthrough of each stage
- [docs/MIGRATION.md](docs/MIGRATION.md) — moving a run between a GPU box and a laptop
