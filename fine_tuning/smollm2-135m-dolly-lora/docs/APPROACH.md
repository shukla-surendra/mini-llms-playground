# Why This Model, This Dataset, This Technique

## The actual goal: a before/after comparison where the difference is unmistakable

The point of this project is specifically to *see* fine-tuning's effect clearly, not just
read about it. That goal shaped every choice below — each one was picked to make the
before/after contrast as clean and legible as possible, not just to build "a" fine-tuning
project.

## Why a base model, not an already-instruct-tuned one

[`../../tinyllama-1.1b-lora/`](../../tinyllama-1.1b-lora/) fine-tunes `TinyLlama-1.1B-
Chat` — a model that's **already** instruction-tuned. Further LoRA fine-tuning there
produces a real, but *incremental* change: the model already knows how to follow
instructions; fine-tuning shifts its style/domain slightly.

This project deliberately picks
[`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — a
genuine **base** model (confirmed directly: its `tokenizer_config.json` has no
`chat_template` field at all) that has only ever seen the raw next-token-prediction
pretraining objective ([Chapter 8](../../../docs/llm-engineering/08_what_is_a_language_model.md)),
never a single structured instruction/response example. Fine-tuning it produces the
**emergence** of instruction-following behavior, not a refinement of behavior that
already existed — a categorically bigger, more visible change, exactly matching this
project's actual goal.

## Why SmolLM2-135M specifically

- **Small enough to LoRA-train quickly on a MacBook** — 135M parameters is comfortably
  within Apple Silicon MPS's practical range for this kind of experiment (confirmed:
  real training runs in minutes, not hours — see
  [`TRAINING_RESULTS.md`](TRAINING_RESULTS.md)).
- **Apache 2.0 licensed** — fully permissive, no usage restrictions.
- **A real, modern architecture** — `LlamaForCausalLM` (confirmed via its `config.json`),
  the same family [Chapter 10](../../../docs/llm-engineering/10_transformer_architecture.md)
  and [Chapter 17](../../../docs/llm-engineering/17_lora_and_qlora.md) already cover, so
  everything already documented about attention/MLP/LoRA target modules applies directly
  — no new architecture concepts needed to understand this project.
- **Confirmed, not assumed, base-model status** — before committing to this model, its
  actual `tokenizer_config.json` was checked directly rather than trusting the name
  alone; genuinely no chat template exists.

## Why Dolly-15k

[`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) —
~15,000 instruction/response pairs, written by real Databricks employees (not
model-generated), covering open Q&A, closed Q&A (with provided context), classification,
summarization, and more. Two things make it a good fit specifically for *this* project:

- **Genuinely diverse task types** in one dataset — the before/after comparison can show
  the model learning several different instruction-following *shapes* at once (answering
  directly, summarizing provided context, choosing between options), not just one
  narrow pattern.
- **A real, human-written dataset**, not synthetic — a meaningful, honest signal for
  what actual instruction-following data looks like, as opposed to a toy/synthetic
  example set built purely for a demo.

## Why the Alpaca-style prompt format specifically

Since SmolLM2-135M has no chat template, *some* explicit format has to be chosen and
taught. The Alpaca-style format (`### Instruction:\n...\n\n### Response:\n...`, with an
optional `### Input:` section for examples that include context) is used here because
it's a well-established, simple, clearly-delimited format — easy for a model with no
prior instruction-following exposure to learn the *structure* of quickly (clear section
markers), and easy for a human reader to verify the model actually learned to respect
the format correctly, not just produce vaguely-relevant text.

## Why LoRA (not full fine-tuning, not QLoRA)

- **Not full fine-tuning**: per
  [Chapter 16](../../../docs/llm-engineering/16_fine_tuning_landscape.md#deep-dive-why-peft-specifically-matters-for-this-curriculums-no-gpu-constraint),
  even at 135M parameters, full fine-tuning's gradient+optimizer-state overhead is
  unnecessary when LoRA achieves the same demonstration goal at a fraction of the
  trainable-parameter count (confirmed: 3.5% trainable, per the real
  `print_trainable_parameters()` output in
  [`TRAINING_RESULTS.md`](TRAINING_RESULTS.md)) — and LoRA is the technique actually worth
  learning and demonstrating here, since it's what makes fine-tuning *larger* models
  (like `tinyllama-1.1b-lora`'s 1.1B-parameter model) feasible on this same hardware.
- **Not QLoRA**: per
  [Chapter 17's deep-dive](../../../docs/llm-engineering/17_lora_and_qlora.md#deep-dive-qlora-and-why-it-isnt-what-this-repo-uses-on-a-macbook),
  QLoRA's 4-bit quantization (`bitsandbytes`) doesn't have working MPS support — plain
  LoRA at float16 is the correct, actually-functional choice on this hardware regardless
  of model size.
