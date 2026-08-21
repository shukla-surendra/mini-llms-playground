"""Generate word tokens autoregressively from a saved checkpoint."""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from .model import GPT
from .runtime import pick_device
from .train import CHECKPOINT_PATH


def load_model(device: str) -> tuple[GPT, dict]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("No checkpoint yet. Run `make train` before `make generate`.")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = GPT(checkpoint["gpt_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def decode(ids: list[int], itos: dict) -> str:
    result = ""
    for token_id in ids:
        token = itos[int(token_id)]
        result = result.rstrip() + token + " " if token in ".,!?;:" else result + token + " "
    return result.strip()


@torch.no_grad()
def generate(model: GPT, stoi: dict, itos: dict, prompt: str, count: int, temperature: float, device: str) -> str:
    # Same simple tokenization rule as WordTokenizer, replicated only for checkpoint-only inference.
    from .tokenizer import WordTokenizer
    unk = stoi["<unk>"]
    ids = [stoi.get(token, unk) for token in WordTokenizer.tokenize(prompt)]
    if not ids:
        raise ValueError("Prompt must contain at least one word or . , ! ? ; : punctuation token.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(count):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return decode(idx[0].tolist(), itos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue a prompt with the trained word GPT.")
    parser.add_argument("--prompt", default="the cat")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    model, checkpoint = load_model(pick_device())
    print(generate(model, checkpoint["stoi"], checkpoint["itos"], args.prompt, args.max_new_tokens, args.temperature, pick_device()))


if __name__ == "__main__":
    main()
