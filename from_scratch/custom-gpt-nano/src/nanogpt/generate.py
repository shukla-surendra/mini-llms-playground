"""
Autoregressive generation: use the trained model to produce new text, one token at a
time, by repeatedly feeding its own output back in as the next input.

WHAT "autoregressive" means here, concretely: the model was only ever trained to answer
one question — "given these tokens, what's the single most likely next one?" (see
train.py). It has no separate "generate a whole sentence" ability. Producing a longer
passage is just that one-token question asked over and over, each time with the
previously generated token appended to the context. Deep dive on this loop plus the
optimizations (KV-caching) production serving code adds on top of it:
docs/llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md.

Run with: `python -m nanogpt.generate --prompt "The cat"` (or `make generate PROMPT=...`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import GPT
from .train import pick_device

_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "nano.pt"


def load_checkpoint(device: str, path: Path = _CHECKPOINT_PATH) -> tuple[GPT, dict]:
    """Rebuild a trained model from its checkpoint file. Shared by the CLI below,
    report.py, and server.py, so there's exactly one place that knows the checkpoint's
    on-disk shape (see train.py's `torch.save(...)` call for what it contains).
    Returns `(model, checkpoint)` — the checkpoint dict still has `stoi`/`itos` (the
    tokenizer's lookup tables, saved alongside the weights so nothing downstream needs
    `data/corpus.txt` to still exist or be unchanged just to run inference), plus
    `step`/`final_train_loss`/`final_val_loss` for display purposes."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GPT(checkpoint["gpt_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # disables any train-only behavior (this model has none, but it's the
    # universal PyTorch convention — always eval() before inference, always train()
    # before resuming training).
    return model, checkpoint


class UnknownCharacterError(ValueError):
    """Raised when a prompt contains a character this tokenizer has no id for — see
    tokenizer.py's `CharTokenizer.encode` docstring for why this isn't silently
    dropped or substituted. A CLI run failing loudly here is a minor inconvenience;
    server.py catches this same error and turns it into a clean HTTP 400 instead,
    since a served API is a boundary where arbitrary outside input has to be validated
    rather than trusted."""


def encode_prompt(prompt: str, stoi: dict) -> list[int]:
    unknown = sorted({ch for ch in prompt if ch not in stoi})
    if unknown:
        raise UnknownCharacterError(
            f"prompt contains character(s) this model was never trained on: "
            f"{unknown!r}. The tokenizer only knows the {len(stoi)} characters that "
            f"appear in data/corpus.txt (see tokenizer.py) — nothing else can be "
            f"encoded, by construction."
        )
    return [stoi[c] for c in prompt]


@torch.no_grad()  # generation never needs gradients — see estimate_loss in train.py
# for the identical reasoning.
def generate(
    model: GPT,
    stoi: dict,
    itos: dict,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> str:
    decode = lambda ids: "".join(itos[int(i)] for i in ids)  # noqa: E731 — tiny, local

    idx = torch.tensor([encode_prompt(prompt, stoi)], dtype=torch.long, device=device)  # (1, T)

    for _ in range(max_new_tokens):
        # The model was only ever trained with sequences up to block_size long (its
        # position embedding table literally has no row for a position beyond that —
        # see model.py's GPT.forward assertion). Once the running sequence is longer
        # than block_size, keep only the most recent block_size tokens as context —
        # exactly the "context window" limit every real LLM has, just visible here as
        # one explicit line instead of an invisible platform limit.
        idx_cond = idx[:, -model.cfg.block_size :]

        logits, _ = model(idx_cond)  # (1, T, vocab_size); no targets -> loss is None

        # We only care about the prediction *after the last token currently in the
        # sequence* — that's the actual "next token" question at this point in
        # generation. Every other position's logits (predictions for positions we
        # already know the true next token for, from the prompt/earlier generation)
        # are simply not useful here.
        logits = logits[:, -1, :] / temperature
        # Temperature divides the logits before softmax. Softmax of small/similar
        # numbers is close to uniform (more random); softmax of the same numbers scaled
        # *up* (temperature < 1) is much more peaked around the highest one (more
        # deterministic, more repetitive). temperature=1.0 leaves the model's own
        # learned distribution untouched.

        probs = F.softmax(logits, dim=-1)  # (1, vocab_size), sums to 1

        # Sample one token id from that probability distribution, weighted by
        # probability — NOT `argmax` (always pick the single highest-probability
        # token). Always taking argmax is deterministic and tends to fall into short
        # repeating loops ("the the the the..."); sampling lets lower-probability but
        # still-reasonable tokens occasionally get picked, which is what makes
        # generated text vary between runs at all.
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        idx = torch.cat([idx, next_id], dim=1)  # grow the sequence by one token

    return decode(idx[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from the trained nanoGPT.")
    parser.add_argument("--prompt", default="The cat", help="Text to continue from.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Lower = more repetitive/predictable, higher = more random.",
    )
    args = parser.parse_args()

    device = pick_device()
    model, checkpoint = load_checkpoint(device)

    text = generate(
        model,
        checkpoint["stoi"],
        checkpoint["itos"],
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
