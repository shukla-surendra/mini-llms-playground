"""
LoRA instruction-tuning of a BASE (non-chat) model: HuggingFaceTB/SmolLM2-135M on
databricks/databricks-dolly-15k.

Unlike ../tinyllama-1.1b-lora/ (which fine-tunes an ALREADY chat-tuned model using its
own built-in chat template), SmolLM2-135M is a true base model with no chat template at
all — it has never seen structured instruction/response data. This script teaches that
behavior essentially from scratch, using the classic Alpaca-style prompt format, which is
why the before/after difference from compare_before_after.py is dramatic rather than
incremental. See docs/APPROACH.md for the full reasoning.
"""
import argparse
import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

PROMPT_WITH_CONTEXT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{context}

### Response:
{response}"""

PROMPT_NO_CONTEXT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


def format_example(example: dict[str, Any]) -> dict[str, str]:
    if example.get("context"):
        text = PROMPT_WITH_CONTEXT.format(
            instruction=example["instruction"], context=example["context"],
            response=example["response"],
        )
    else:
        text = PROMPT_NO_CONTEXT.format(
            instruction=example["instruction"], response=example["response"],
        )
    return {"text": text}


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_dtype(device: str, force_float32: bool) -> torch.dtype:
    if force_float32:
        return torch.float32
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def find_last_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def step(p: Path) -> int:
        suffix = p.name.replace("checkpoint-", "")
        return int(suffix) if suffix.isdigit() else -1

    checkpoints.sort(key=step)
    return str(checkpoints[-1]) if step(checkpoints[-1]) >= 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of SmolLM2-135M (base) on Dolly-15k")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    parser.add_argument("--max-samples", type=int, default=4000,
                         help="Rows to use out of ~15,011 available")
    parser.add_argument("--output-dir", default="outputs/smollm2_dolly_lora")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--force-float32", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    dtype = model_dtype(device, args.force_float32)
    print(f"[runtime] device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    raw_ds: Dataset = load_dataset(args.dataset, split="train")
    if args.max_samples > 0:
        raw_ds = raw_ds.shuffle(seed=42).select(range(min(args.max_samples, len(raw_ds))))

    text_ds = raw_ds.map(format_example, remove_columns=raw_ds.column_names, desc="Formatting")

    def tokenize_rows(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        enc = tokenizer(batch["text"], truncation=True, max_length=args.seq_len, padding=False)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    tokenized = text_ds.map(tokenize_rows, batched=True, remove_columns=["text"], desc="Tokenizing")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    samples_per_update = max(1, args.batch_size * args.grad_accum)
    updates_per_epoch = max(1, math.ceil(len(tokenized) / samples_per_update))
    total_steps = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"[train] updates_per_epoch={updates_per_epoch} total_steps~{total_steps} "
          f"warmup_steps={warmup_steps}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer_kwargs = {
        "model": model, "args": training_args,
        "train_dataset": tokenized, "data_collator": data_collator,
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        try:
            trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
        except TypeError:
            trainer = Trainer(**trainer_kwargs)

    resume_checkpoint = args.resume_from_checkpoint or (find_last_checkpoint(output_dir) if args.resume else None)
    if resume_checkpoint:
        print(f"[resume] continuing from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        print("[resume] no checkpoint found; starting fresh")
        trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[done] LoRA adapter saved at {final_dir}")


if __name__ == "__main__":
    main()
