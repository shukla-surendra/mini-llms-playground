# SmolLM2-135M + LoRA on Dolly-15k — Teaching Instruction-Following From Scratch

Part of [mini-llms-playground](../../README.md)'s **fine-tuning track**, sibling to
[`tinyllama-1.1b-lora`](../tinyllama-1.1b-lora/). See the
[top-level README](../../README.md) and [docs index](../../docs/README.md) for how this
relates to the rest of the repo, and the
[LLM Engineering Curriculum's Part 3](../../docs/llm-engineering/00_roadmap.md#part-3--fine-tuning-adapting-an-existing-model)
for the full conceptual background (the fine-tuning landscape, LoRA's mechanism, SFT,
RLHF/DPO, and evaluation methodology) every design choice here is grounded in.

## What makes this project different from `tinyllama-1.1b-lora`

`tinyllama-1.1b-lora` fine-tunes a model that's **already** instruction-tuned
(`TinyLlama-1.1B-Chat`) further, using its own existing chat template — an incremental
adaptation. This project does something more dramatic, specifically to make a real
before/after comparison obvious: it fine-tunes
[**`HuggingFaceTB/SmolLM2-135M`**](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), a
genuine **base** model that has never seen a single instruction/response pair, and
teaches it to follow instructions essentially from scratch — see
[`docs/APPROACH.md`](docs/APPROACH.md) for the full reasoning behind this choice.

Want to run the original, unmodified base checkpoint on its own (no adapter) instead of
always seeing it paired with LoRA? See
[`base_models/smollm2-135m-base-serving`](../../base_models/smollm2-135m-base-serving/) —
same model, its own endpoint, original-author repo details included.

## Quickstart

```bash
cd fine_tuning/smollm2-135m-dolly-lora

# 1. Train the LoRA adapter (real, observed: a few minutes on Apple Silicon MPS
#    for a few thousand examples — see docs/TRAINING_RESULTS.md for exact numbers)
uv run train_lora.py

# 2. See the real difference — generates from BOTH the original base model and
#    the fine-tuned model, on the same prompts, side by side
uv run compare_before_after.py
```

Everything runs through [`uv`](https://docs.astral.sh/uv/) — `uv run` provisions
`.venv` from [`pyproject.toml`](pyproject.toml) automatically, no manual setup needed.

## What gets fine-tuned, and how

- **Base model**: `HuggingFaceTB/SmolLM2-135M` — 135M parameters, Apache 2.0 licensed,
  Llama-family architecture, a true base (pretraining-only) model with no chat template.
- **Dataset**: [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) —
  ~15,000 real, human-written instruction/response pairs (CC-BY-SA-3.0).
- **Technique**: LoRA (not QLoRA — see
  [Chapter 17](../../docs/llm-engineering/17_lora_and_qlora.md#deep-dive-qlora-and-why-it-isnt-what-this-repo-uses-on-a-macbook)
  for why QLoRA isn't a working option on Apple Silicon MPS today), `r=16`, targeting
  every attention and MLP projection matrix.
- **Prompt format**: the classic Alpaca-style template (`### Instruction: ... ###
  Response: ...`) — this base model has no built-in chat template to reuse, so
  `train_lora.py` defines one explicitly. Full reasoning in
  [`docs/APPROACH.md`](docs/APPROACH.md).

## Real results

Actually run: 4,000 Dolly examples, 3 epochs, 750 steps, **~63.5 minutes on Apple Silicon
MPS**. Real, unedited output for the prompt *"Explain what a black hole is in one
sentence"* — the clearest example in the full comparison:

**BEFORE (base model)**: `Explain what a black hole is in one sentence. ### Instruction: Explain what a black hole is in one sentence. ### Response: Explain what a black hole is in one sentence...` (just echoes the instruction back, in a loop — never attempts an answer)

**AFTER (LoRA fine-tuned)**: `A black hole is a region of space where the gravity is so strong that nothing, not even light, can escape. Black holes are created when a massive star runs out of fuel and collapses...` (a genuinely correct, substantive answer)

See [`docs/TRAINING_RESULTS.md`](docs/TRAINING_RESULTS.md) for the full loss curve and
timing, and [`docs/BEFORE_AFTER_COMPARISON.md`](docs/BEFORE_AFTER_COMPARISON.md) for all
four prompts' real output — including one case (summarization) where the improvement was
real but more modest, documented honestly rather than cherry-picked.

## Full docs

- [`docs/APPROACH.md`](docs/APPROACH.md) — why this model, this dataset, this technique,
  and why a base (not chat) model makes the comparison more informative.
- [`docs/TRAINING_RESULTS.md`](docs/TRAINING_RESULTS.md) — the real training run.
- [`docs/BEFORE_AFTER_COMPARISON.md`](docs/BEFORE_AFTER_COMPARISON.md) — real generated
  output, base vs. fine-tuned, side by side.
- [`../../docs/llm-engineering/00_roadmap.md#part-3--fine-tuning-adapting-an-existing-model`](../../docs/llm-engineering/00_roadmap.md#part-3--fine-tuning-adapting-an-existing-model) —
  the full conceptual curriculum this project is a hands-on instance of.
