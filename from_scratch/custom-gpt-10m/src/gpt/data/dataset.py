"""Tokenization, batching, and the training loss.

Raw language modeling: every token in the text is a training target, exactly how a
base model like GPT-2 is pretrained. The corpus happens to contain chat transcripts
(see docs/DATASETS.md), but nothing here treats them specially — no chat template,
no per-turn loss masking.
"""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .prepare import DOCUMENT_SEPARATOR


def load_text(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` to download and build the corpus first."
        )
    return path.read_text(encoding="utf-8")


def encode_raw(tokenizer, text, device):
    """Tokenize text into one flat token stream to train over.

    `allowed_special={DOCUMENT_SEPARATOR}` matters, not just `disallowed_special=()`:
    without it, the literal string "<|endoftext|>" in the corpus (prepare.py's document
    boundary marker) tokenizes as 7 ordinary subword pieces ("<", "|", "end", "of",
    "text", "|", ">"), not GPT-2's real reserved special-token id — silently defeating
    the entire point of using it as a boundary marker (confirmed empirically, not assumed:
    encode("<|endoftext|>", disallowed_special=()) alone gives 7 tokens; adding
    allowed_special gives the single real token id 50256). disallowed_special=() is still
    needed too — it's what stops the tokenizer from raising on encountering the string at
    all before allowed_special gets a chance to recognize it.
    """
    return torch.tensor(
        tokenizer.encode(text, allowed_special={DOCUMENT_SEPARATOR}, disallowed_special=()),
        device=device,
    )


def effective_context_length(configured, *token_streams):
    """Shrink the context window if a dataset is too small to fill it.

    get_batch needs at least ctx_len+1 tokens to build an (input, target) pair, so a
    small smoke-test corpus would otherwise crash instead of just training short.
    """
    limit = min(len(stream) - 1 for stream in token_streams)
    return max(1, min(configured, limit))


def get_batch(tokens, ctx_len, batch_size, device):
    """Random windows of `ctx_len` tokens; targets are the same window shifted by one."""
    max_start = len(tokens) - ctx_len
    ix = torch.randint(0, max_start, (batch_size,), device=device)
    x = torch.stack([tokens[i:i + ctx_len] for i in ix])
    y = torch.stack([tokens[i + 1:i + ctx_len + 1] for i in ix])
    return x, y


def next_token_loss(logits, targets, vocab_size):
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
