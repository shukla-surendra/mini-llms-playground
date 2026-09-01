# Qwen2.5-7B-Instruct — QLoRA fine-tuning on a single L4

Reminder note only, 2026-08-31 — nothing implemented yet, full build planned
for a future session. Placeholder for what to build, matching this repo's
`fine_tuning/tinyllama-1.1b-lora/` conventions (train script + serve script +
requirements.txt + scripts/train.sh, scripts/serve.sh).

## Why this needs QLoRA specifically, not plain LoRA

An L4 has 24 GB VRAM. Qwen2.5-7B-Instruct's own weights in bf16 are
`7.6B × 2 bytes ≈ 15.2 GB` before any optimizer state, gradients, or
activations — too tight for plain LoRA (frozen bf16 base + adapter) to leave
comfortable headroom. QLoRA loads the frozen base in 4-bit (`bitsandbytes` NF4)
instead (`≈3.8 GB`), freeing the rest of the 24 GB for adapter
gradients/optimizer state and real activation memory at a workable batch size —
the standard technique for fine-tuning a 7B-class model on one card this size.

## To build, when picked back up

- `train_qwen_lora.py` — QLoRA training (4-bit `BitsAndBytesConfig` base load,
  `peft.LoraConfig` targeting Qwen2.5's Llama-style module names: `q_proj`,
  `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- `serve_qwen_lora.py` — base model + adapter inference server.
- `requirements.txt`, `scripts/train.sh`, `scripts/serve.sh` — same shape as
  the `tinyllama-1.1b-lora` sibling project.
- Dataset not yet chosen.
- Not run on real hardware — the memory math above is a real, checkable
  estimate, not a measured result yet.
