"""
Generate from the ORIGINAL base model and the LoRA-fine-tuned model, on the SAME
prompts, under the SAME decoding settings, side by side — the fair-comparison
methodology from ../../docs/llm-engineering/20_evaluating_a_fine_tuned_model.md.

Usage:
    python compare_before_after.py
    python compare_before_after.py --adapter-path outputs/smollm2_dolly_lora/final
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    {"instruction": "What is the capital of France?", "context": ""},
    {"instruction": "Explain what a black hole is in one sentence.", "context": ""},
    {"instruction": "List three benefits of regular exercise.", "context": ""},
    {"instruction": "Summarize the following text in one sentence.",
     "context": "The Great Wall of China is a series of fortifications that were built "
                 "across the historical northern borders of China to protect against "
                 "invasions. Construction began as early as the 7th century BC."},
]


def build_prompt(instruction: str, context: str) -> str:
    if context:
        return (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes "
            f"the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}"
            "\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request.\n\n### Instruction:\n{instruction}"
        "\n\n### Response:\n"
    )


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy — deterministic, fair comparison
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base vs. LoRA-fine-tuned model outputs")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--adapter-path", default="outputs/smollm2_dolly_lora/final")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = detect_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    print(f"[runtime] device={device} dtype={dtype}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[loading] base model: {args.model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    base_model.eval()

    print(f"[loading] LoRA adapter: {args.adapter_path}\n")
    # IMPORTANT: PeftModel.from_pretrained wraps base_model's modules IN PLACE — after
    # this call, base_model and model share the same underlying weights, so calling
    # base_model.generate(...) directly would ALSO reflect the LoRA adapter (a real bug
    # this script had until this fix — see docs/BEFORE_AFTER_COMPARISON.md for the full
    # story). model.disable_adapter() is PEFT's own, correct mechanism for temporarily
    # reverting to pure base-model behavior on the SAME loaded model, no second copy
    # needed — the memory-efficient, idiomatic way to do this comparison.
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    for i, ex in enumerate(DEFAULT_PROMPTS, 1):
        prompt = build_prompt(ex["instruction"], ex["context"])
        print(f"{'=' * 70}")
        print(f"PROMPT {i}: {ex['instruction']}")
        if ex["context"]:
            print(f"  (with context: {ex['context'][:80]}...)")
        print(f"{'=' * 70}")

        with model.disable_adapter():
            base_output = generate(model, tokenizer, prompt, device, args.max_new_tokens)
        print(f"\n--- BEFORE (base model, no fine-tuning) ---\n{base_output}\n")

        ft_output = generate(model, tokenizer, prompt, device, args.max_new_tokens)
        print(f"--- AFTER (LoRA fine-tuned on Dolly-15k) ---\n{ft_output}\n")


if __name__ == "__main__":
    main()
