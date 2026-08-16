---
language:
- en
license: mit
library_name: pytorch
tags:
- text-generation
- causal-lm
- gpt
- small-language-model
- from-scratch
datasets:
- roneneldan/TinyStories
pipeline_tag: text-generation
---

# TinyStories GPT (~5.85M params)

A small, decoder-only GPT-style Transformer, trained completely from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — short children's
stories written with a deliberately restricted vocabulary, specifically so a very small
model can still produce grammatically sensible, locally coherent text.

**This is not a general-purpose or instruction-following model.** It only continues
short-story-style prompts. See
[the project's full documentation](https://github.com/shukla-surendra/mini-llms-playground/tree/main/from_scratch/custom-gpt-6m)
for the complete build, training, and design reasoning.

## Model details

| | |
|---|---|
| Architecture | Decoder-only Transformer (causal self-attention, pre-norm residual blocks, weight-tied output head) |
| Parameters | 5,853,184 (~5.85M) |
| Vocabulary | 4,096 tokens — a custom byte-level BPE tokenizer trained on TinyStories itself (not GPT-2's tokenizer) |
| Context length | 256 tokens |
| Layers / heads / embed size | 6 / 8 / 256 |
| Training data | 100,000 TinyStories stories (~22.4M tokens) |
| Training hardware | Apple Silicon MPS (a MacBook, no GPU) |

## Real training results

Final evaluation (step ~5,000): `val_loss ≈ 2.42`, `val_perplexity ≈ 11.2`.

Real, unedited sample (`temperature=0.8`):

> Once upon a time, there was a little girl named Lily. She loved to play with her toys
> and her toy cars. One day, she saw a small boy walking by the park. He was scared
> because he didn't know what to do.
>
> Lily said, "I don't want to go and see my toy car!" But then, she heard a loud noise.
> It was a scary dog that was running towards him...

## How to use this model

This is **not** a standard `transformers`-library model — it's a custom PyTorch
architecture, so it needs its own loading code (`model.py` and `inference.py`, both
included in this repo) rather than `AutoModelForCausalLM`.

```bash
pip install torch tokenizers huggingface_hub

python -c "
from huggingface_hub import hf_hub_download
import torch
from tokenizers import Tokenizer

repo_id = 'YOUR-USERNAME/custom-gpt-6m'  # replace with the actual repo you download from
ckpt_path = hf_hub_download(repo_id, 'custom_gpt_6m_checkpoint.pt')
tok_path = hf_hub_download(repo_id, 'tokenizer.json')
model_py = hf_hub_download(repo_id, 'model.py')

import importlib.util
spec = importlib.util.spec_from_file_location('model', model_py)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

ckpt = torch.load(ckpt_path, map_location='cpu')
tokenizer = Tokenizer.from_file(tok_path)
model = model_module.build_model(
    vocab_size=ckpt['vocab_size'], context_length=ckpt['context_length'],
    embed_size=ckpt['embed_size'], num_heads=ckpt['num_heads'],
    num_layers=ckpt['num_layers'], dropout=0.0,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

ids = torch.tensor([tokenizer.encode('Once upon a time,').ids])
for _ in range(80):
    logits = model(ids)[:, -1, :]
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    ids = torch.cat([ids, next_id], dim=1)
print(tokenizer.decode(ids[0].tolist()))
"
```

Or clone the [full project](https://github.com/shukla-surendra/mini-llms-playground) and
use `src/gpt/inference/generate.py` (`gpt-infer`) / `src/gpt/inference/server.py`
(`gpt-serve`) directly, which handle sampling (temperature/top-k/top-p/repetition-penalty),
not just greedy decoding.

## Why this model exists

Built as a hands-on exploration of training a language model completely from scratch on
consumer hardware (no GPU), with the explicit goal of "small and coherent," not "large and
capable." Full architecture reasoning, training methodology, and a from-first-principles
curriculum covering how every piece works are in the
[source repository's docs](https://github.com/shukla-surendra/mini-llms-playground/tree/main/from_scratch/custom-gpt-6m/docs).

## Limitations

- Only produces TinyStories-style short narrative text — not a general assistant.
- 4,096-token vocabulary means text outside TinyStories' simple register (technical
  terms, unusual proper nouns) tokenizes inefficiently and generates poorly.
- 256-token context window — no long-document or long-conversation capability.
- No instruction tuning, no RLHF/DPO, no safety filtering — a raw pretrained model.
