"""Continue a prompt with the trained model. No KV cache (right-sized-subset scope
choice - see docs/llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md
for what that trades away and why it's a legitimate choice at this scale)."""

import argparse

import torch
import torch.nn.functional as F

from ..checkpoint import load_model, resolve_serving_checkpoint
from ..config import load_settings
from ..runtime import get_device


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, device):
    ids = tokenizer.encode_ordinary(prompt)
    if not ids:
        raise ValueError("Prompt must encode to at least one token.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        window = idx[:, -model.context_length:]
        logits, _ = model(window)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return tokenizer.decode(idx[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue a prompt with the trained model.")
    _, train_cfg, paths = load_settings()
    parser.add_argument("--prompt", default=train_cfg.demo_prompt)
    parser.add_argument("--max-new-tokens", type=int, default=train_cfg.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")

    device = get_device()
    checkpoint, tokenizer, model = load_model(resolve_serving_checkpoint(paths), device)
    print(generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, device))


if __name__ == "__main__":
    main()
