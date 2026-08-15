"""Dataset registry, corpus construction, tokenization/batching, and quality audit."""

from .dataset import (
    TOKEN_DTYPE,
    build_token_bin,
    effective_context_length,
    encode_raw,
    get_batch,
    load_text,
    load_token_array,
    next_token_loss,
    token_bin_path,
)
from .prompts import DEFAULT_PROMPTS, load_prompts

__all__ = [
    "TOKEN_DTYPE",
    "build_token_bin",
    "effective_context_length",
    "encode_raw",
    "get_batch",
    "load_text",
    "load_token_array",
    "next_token_loss",
    "token_bin_path",
    "load_prompts",
    "DEFAULT_PROMPTS",
]
