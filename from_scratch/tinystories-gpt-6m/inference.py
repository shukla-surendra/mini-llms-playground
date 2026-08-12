"""
Generate text from a trained checkpoint.

Usage:
    python inference.py --prompt "Once upon a time"
    python inference.py --prompt "Once upon a time" --do-sample --temperature 0.8
"""
import argparse

import torch
from tokenizers import Tokenizer

from model import build_model, detect_device


def sample_next_token(logits, temperature=0.8, top_k=40, top_p=0.9):
    logits = logits / max(temperature, 1e-5)
    k = min(top_k, logits.size(-1))
    vals, idx = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(vals, dim=-1)
    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
        probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    chosen = torch.multinomial(probs, num_samples=1)
    return idx.gather(-1, chosen)


def apply_repetition_penalty(logits, ids, penalty=1.15, window=64):
    if penalty <= 1.0:
        return logits
    adjusted = logits.clone()
    recent = ids[0, -window:].unique()
    adjusted[:, recent] = adjusted[:, recent] / penalty
    return adjusted


@torch.no_grad()
def generate(model, tokenizer, prompt, ctx_len, max_new_tokens, device, do_sample=True,
             temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.15):
    model.eval()
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    for _ in range(max_new_tokens):
        window = ids[:, -ctx_len:]
        logits = model(window)[:, -1, :]
        logits = apply_repetition_penalty(logits, ids, penalty=repetition_penalty, window=ctx_len)
        if do_sample:
            next_id = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        if next_id.item() == eot_id:
            break
        ids = torch.cat([ids, next_id], dim=1)
    return tokenizer.decode(ids[0].tolist())


def load_model_and_tokenizer(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    tokenizer = Tokenizer.from_file(ckpt["tokenizer_path"])
    model = build_model(
        vocab_size=ckpt["vocab_size"],
        context_length=ckpt["context_length"],
        embed_size=ckpt["embed_size"],
        num_heads=ckpt["num_heads"],
        num_layers=ckpt["num_layers"],
        dropout=0.0,  # no dropout at inference time
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="tinystories_gpt_checkpoint.pt")
    p.add_argument("--prompt", default="Once upon a time,")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--do-sample", action="store_true", default=True)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    args = p.parse_args()

    device = detect_device()
    model, tokenizer, ckpt = load_model_and_tokenizer(args.checkpoint, device)
    print(f"[model] loaded step={ckpt.get('step')} params={model.num_parameters():,} device={device}")

    text = generate(
        model, tokenizer, args.prompt,
        ctx_len=ckpt["context_length"],
        max_new_tokens=args.max_new_tokens,
        device=device,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print("\n--- Generated ---")
    print(text)


if __name__ == "__main__":
    main()
