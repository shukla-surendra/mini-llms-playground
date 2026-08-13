# SmolLM2-135M — Real, Verified Model Details

Every fact in this doc was pulled directly from the model's own real
[model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) and `config.json` on the
Hugging Face Hub — not recalled from memory, not assumed.

## Architecture — verified against the checkpoint's own `config.json`

```
architectures: LlamaForCausalLM
hidden_size: 576
num_hidden_layers: 30
num_attention_heads: 9
vocab_size: 49152
model_max_length: 8192
chat_template: NONE — confirmed absent from tokenizer_config.json
```

Same broad architecture family (`LlamaForCausalLM`) as
[`../tinyllama-1.1b-base-serving/`](../../tinyllama-1.1b-base-serving/)'s model, but a
genuinely different shape — narrower (`hidden_size=576` vs. 2048) and deeper relative to
its width (30 layers) than TinyLlama's 22 layers at 2048-wide, a real, checkable
architectural difference between the two models this repo serves side by side.

**No chat template, confirmed directly**: unlike TinyLlama-1.1B-Chat, this checkpoint's
`tokenizer_config.json` has no `chat_template` field at all — the single fact this
project's entire design (per
[`../../../fine_tuning/smollm2-135m-dolly-lora/docs/APPROACH.md`](../../../fine_tuning/smollm2-135m-dolly-lora/docs/APPROACH.md))
is built around: this is a true base model, never exposed to structured conversation
data, which is exactly why [`api_server.py`](../api_server.py) only offers plain-text
completion, not chat.

## Pretraining — from the official model card

- **2 trillion tokens**, using a mix of [`FineWeb-Edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (filtered, education-quality web text), [`DCLM`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
  (DataComp-LM's curated web corpus), and [`The Stack`](https://huggingface.co/datasets/bigcode/the-stack)
  (code), plus additional curated datasets the SmolLM team developed themselves.
- **Hardware**: 64× H100 GPUs.
- **Training framework**: [`nanotron`](https://github.com/huggingface/nanotron)
  (Hugging Face's own distributed-training library).
- **Precision**: bfloat16.

## The instruction-tuned sibling's recipe (for context — NOT what this checkpoint is)

Worth naming precisely, since it directly parallels
[`../../tinyllama-1.1b-base-serving/docs/MODEL_DETAILS.md`](../../tinyllama-1.1b-base-serving/docs/MODEL_DETAILS.md)'s
findings: SmolLM2 also has an `-Instruct` sibling checkpoint, built via the exact same
SFT→DPO pattern — SFT on a curated dataset mix
([`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)),
then DPO alignment using `UltraFeedback` (the same preference dataset TinyLlama's chat
model used). **This repo deliberately uses the base (`SmolLM2-135M`), not the
`-Instruct` variant** — see
[`../../../fine_tuning/smollm2-135m-dolly-lora/docs/APPROACH.md`](../../../fine_tuning/smollm2-135m-dolly-lora/docs/APPROACH.md)
for why: fine-tuning a true base model produces the dramatic, unmistakable before/after
contrast this project's whole design goal depends on, which fine-tuning an
already-Instruct model further would not.

## Benchmarks — real, from the official model card (base model, zero-shot unless noted)

| Metric | SmolLM2-135M | Predecessor (SmolLM-135M) |
|---|---|---|
| HellaSwag | **42.1** | 41.2 |
| ARC (Average) | **43.9** | 42.4 |
| PIQA | 68.4 | 68.4 |
| MMLU (cloze) | **31.5** | 30.2 |
| CommonsenseQA | **33.9** | 32.7 |
| TriviaQA | 4.1 | 4.3 |
| Winogrande | 51.3 | 51.3 |
| OpenBookQA | **34.6** | 34.0 |
| GSM8K (5-shot) | **1.4** | 1.0 |

Evaluated using [`lighteval`](https://github.com/huggingface/lighteval), per the model
card. Note the very low `GSM8K` (grade-school math) score across both generations —
consistent with this repo's own observation in
[`../../../fine_tuning/smollm2-135m-dolly-lora/docs/BEFORE_AFTER_COMPARISON.md`](../../../fine_tuning/smollm2-135m-dolly-lora/docs/BEFORE_AFTER_COMPARISON.md)
that compositional/reasoning-heavy tasks (like true summarization) show the least
fine-tuning improvement of the tasks tested — a real, independently-corroborated
limitation of a model this small, not specific to this repo's own fine-tuning.

## Known limitations (from the model card, verbatim reasoning)

- Primarily English — content in other languages is not reliably supported.
- "The generated content may not always be factually accurate, logically consistent, or
  free from biases present in the training data" — the model card's own explicit
  framing, worth repeating precisely rather than softened.
- Should be used as an assistive tool, not a definitive information source — official
  guidance, not this repo's own added caveat.
- **This repo's own addition, confirmed directly in
  [`BEFORE_AFTER_COMPARISON.md`](../../../fine_tuning/smollm2-135m-dolly-lora/docs/BEFORE_AFTER_COMPARISON.md)**:
  as a true base model with no instruction-following training at all, this specific
  checkpoint (unlike the `-Instruct` sibling) does not reliably answer questions or stop
  generating at a natural point — it will continue producing text indefinitely up to
  `max_new_tokens`, a direct, observed consequence of having no chat/instruction training,
  not a bug in this serving setup.
