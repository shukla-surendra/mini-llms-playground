# custom-gpt-nano — a GPT small enough to read end to end

Part of [mini-llms-playground](../../README.md)'s **from-scratch track**, sitting
*below* the six size-ladder projects tracked in [`../models.md`](../models.md) (6m
through 350m). This project isn't on that ladder and never will be — it exists to
teach the mechanism, not to compete on output quality. If you're new to PyTorch and
deep learning and want to actually understand what a "GPT" is doing, line by line,
start here before opening [`../custom-gpt-50m/`](../custom-gpt-50m/) or any of its
siblings.

## Why a separate, smaller project instead of just reading `custom-gpt-6m`

`custom-gpt-6m` (5,853,184 params) is real, working, and already the smallest model in
this workspace — but it's written as *production-shaped* code: a `gpt.cli` package with
five different training objectives (causal/MLM/contrastive/DDP/FSDP), a FastAPI serving
layer, resumable multi-day checkpointing, a 4,096-entry custom-trained vocabulary. All
of that is the right call for a project meant to be *used*, but it means the actual
"here's what an attention layer computes" logic is one file among two dozen, wrapped in
concerns a first read doesn't need yet.

`custom-gpt-nano` strips every one of those concerns out on purpose:

| | `custom-gpt-6m` and siblings | `custom-gpt-nano` |
|---|---|---|
| Parameters | 5.85M – 347M | **~800,000** |
| Tokenizer | GPT-2 BPE (50,257) or a trained custom vocab (4,096) | **character-level** (~65 symbols, built from `data/corpus.txt` itself) |
| Attention | `F.scaled_dot_product_attention` (fused, fast, opaque) | **hand-written** Q/K/V matmuls, one line per step |
| Training data | Downloaded HuggingFace datasets, gigabytes | **one bundled ~5KB text file**, already in this repo |
| Training objectives | causal, MLM, contrastive, DDP, FSDP | **causal only** |
| Serving | FastAPI (`make serve`) | none — `make generate` is a plain script |
| Source files | ~25 | **6** |

Nothing here is a "toy" simplification of the *math* — the attention mechanism, the
training loop, weight tying, all of it is the real thing, just without the production
machinery layered on top. Once this makes sense, every one of those bigger projects is
the same ideas at a larger scale plus infrastructure.

## The one design choice that matters most: character-level tokenization

Every other project in this workspace uses a subword tokenizer (GPT-2's BPE, or a
custom-trained one) with tens of thousands of vocabulary entries. At small model sizes,
that vocabulary's embedding table (`vocab_size x n_embd`) can dominate the *entire*
parameter budget — `custom-gpt-50m`'s own docs measure it at over 80% of parameters at
that project's smallest preset. A model this small, given a 50,257-word vocabulary,
would be almost entirely embedding table and barely any Transformer at all.

This project instead tokenizes character-by-character: every distinct character in
`data/corpus.txt` (about 65 of them — letters, digits, punctuation, space, newline) gets
one integer id. See `src/nanogpt/tokenizer.py`'s docstring for the full reasoning and
the honest tradeoff (the model has to learn to *spell*, not just to sequence words).

## Where the ~800,000 parameters actually go

```
$ make config
vocab_size  = 43
block_size  = 64
n_embd      = 128
n_head      = 4  (head_size = 32)
n_layer     = 4

parameter breakdown:
  token_embedding        5,504  ( 0.7%)  [reused as the output layer — see model.py's weight-tying note]
  position_embedding     8,192  ( 1.0%)
  4 x transformer_block ~198,336 each  (98.3% combined)
  total                807,040
```

(Exact `vocab_size` and totals depend on `data/corpus.txt`'s exact character set —
run `make config` yourself to see the real numbers.) Compare this to the "token
embedding = 80% of parameters" problem the workspace's larger projects hit at small
sizes: here it's ~2% combined, and virtually the whole budget goes to the actual
attention + MLP mechanism this project exists to teach.

## Quickstart

```bash
make setup                          # uv sync: create .venv, install torch (only dependency)
make config                         # print the breakdown above, no training
make train                          # trains in well under a minute on CPU alone
make generate PROMPT="The cat"      # sample text from the trained checkpoint
```

No HuggingFace account, no dataset download, no GPU required — everything needed to
train is already in this folder (`data/corpus.txt`).

## What to actually expect from the output

This is an ~800K-parameter model trained on ~5KB of text for 2,000 steps. It will not
hold a conversation, answer a question, or reason about anything. What it *will* do,
and what's worth watching for: `data/corpus.txt` is written as short, heavily repeated
sentence templates ("The cat sat on the mat. / The dog sat on the log. / ..."). Within a
few hundred steps you should see the model start reliably completing those exact
patterns — that's the model **overfitting a tiny dataset**, which here is a feature, not
a bug: it's the most direct, visible proof that gradient descent (`train.py`) is
actually working. Compare `train loss` against `val loss` in the training output
(`make train`'s printed lines) — a val loss that stops improving (or gets worse) while
train loss keeps dropping is overfitting made visible as two numbers, the exact concept
`custom-gpt-50m/docs/TRAINING_SCHEDULE.md`'s stopping-criteria framework is built around
at a much larger scale.

## Reading order

Start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full technical
specification — what a Transformer actually is (architecture vs. model vs. algorithm),
this project's exact spec, a parameter-by-parameter breakdown, a shape trace through one
forward pass, and how this compares to the other six `from_scratch/` projects. Then the
code itself is meant to be read top to bottom, each file's module docstring first:

1. `src/nanogpt/tokenizer.py` — text ↔ integers
2. `src/nanogpt/config.py` — every hyperparameter, and what each one controls
3. `src/nanogpt/model.py` — the architecture: embeddings → attention → MLP → output
4. `src/nanogpt/data.py` — how training windows get sampled from the corpus
5. `src/nanogpt/train.py` — the actual learning loop (forward, loss, backward, step)
6. `src/nanogpt/generate.py` — using the trained model to produce new text

Every file cross-references the relevant chapter of this workspace's
[LLM Engineering Curriculum](../../docs/llm-engineering/00_roadmap.md) for the deeper
"why does this concept exist at all" theory — the comments in this codebase focus on
the narrower, and often harder-to-find-explained, question of "what does this look like
as actual PyTorch tensors and operations."

## Next step once this makes sense

Move to [`../custom-gpt-50m/`](../custom-gpt-50m/) or
[`../custom-gpt-10m/`](../custom-gpt-10m/) — same core mechanism, but with the GPT-2
subword tokenizer, `F.scaled_dot_product_attention` instead of hand-written matmuls, a
real multi-source text corpus, and the resumable/monitored training setup a multi-hour
(or multi-day) run actually needs. `../models.md` is the map across all six of those.
