"""Turn-aware tokenization and masked batch construction for SFT.

Unlike the pretraining path (dataset.py's get_batch(), random windows over a flat
uint16 memmap with no notion of turn boundaries), SFT needs to know exactly which
tokens are an Assistant turn's, so masked_next_token_loss() (dataset.py) can zero out
gradient from everything else (System/User turns, and the prompt in general) — the
"real SFT" mechanism this project didn't have before (see sft_prepare.py's docstring).

The SFT corpus is small enough (tens of thousands of conversations, not billions of
tokens) to tokenize once and hold fully in memory as a list of (ids, is_assistant)
arrays, unlike the pretraining corpus's disk-backed memmap.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import tokenizer_fingerprint


def _cache_path(jsonl_path, context_length, fingerprint):
    jsonl_path = Path(jsonl_path)
    key = json.dumps(
        {"context_length": context_length, "tokenizer": fingerprint}, sort_keys=True,
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return jsonl_path.with_suffix(f".{digest}.examples.pt")


def load_sft_examples(jsonl_path, tokenizer, context_length, rebuild=False):
    """Tokenize every conversation in `jsonl_path` (sft_prepare.py's output format —
    one {"turns": [{"role", "text"}, ...]} object per line) into (ids, is_assistant)
    pairs, cached to a sibling .pt file keyed by (context_length, tokenizer fingerprint)
    so unrelated runs never silently reuse a stale tokenization — same staleness
    principle as dataset.py's load_token_array, just keyed by content hash instead of
    mtime comparison since this file (unlike a giant pretraining .bin) is cheap to
    fully re-hash-check via its cache key.

    Returns a list of (ids: np.ndarray[int64], is_assistant: np.ndarray[bool]) pairs,
    same length per pair, ready for make_sft_batch(). Conversations that end up with no
    surviving Assistant-turn tokens after truncation are dropped — see the truncation
    note below.
    """
    jsonl_path = Path(jsonl_path)
    fingerprint = tokenizer_fingerprint(tokenizer)
    cache_path = _cache_path(jsonl_path, context_length, fingerprint)

    if not rebuild and cache_path.exists() and cache_path.stat().st_mtime >= jsonl_path.stat().st_mtime:
        return torch.load(cache_path, weights_only=False)

    examples = []
    dropped_empty = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turns = json.loads(line)["turns"]
            example = _tokenize_conversation(turns, tokenizer, context_length)
            if example is None:
                dropped_empty += 1
                continue
            examples.append(example)

    if dropped_empty:
        print(
            f"[info] {jsonl_path}: dropped {dropped_empty:,} conversation(s) with no "
            f"surviving Assistant-turn tokens after truncation to context_length="
            f"{context_length}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, cache_path)
    return examples


def _tokenize_conversation(turns, tokenizer, context_length):
    """Tokenize one conversation's turns, tag which tokens are loss-eligible
    (Assistant-turn tokens, including that turn's own "Assistant: " prefix and
    trailing newline — the model needs gradient on emitting the turn marker itself to
    learn turn-taking, not just the reply content), and truncate to context_length by
    dropping whole turns from the front (never mid-turn), so the final — usually most
    informative — Assistant reply is always what survives a too-long conversation.

    Returns None if nothing fits, or if the surviving span has zero Assistant tokens
    (a training example with an all-masked target is pure waste — its loss is 0/0 —
    so it's dropped rather than shipped to the batcher).
    """
    per_turn = []
    for turn in turns:
        role, text = turn["role"], turn["text"]
        ids = tokenizer.encode(f"{role}: {text}\n", disallowed_special=())
        per_turn.append((ids, role == "Assistant"))

    # Drop whole turns from the front until the total fits context_length + 1 (need
    # one extra token for the input/target shift in make_sft_batch).
    start = 0
    total = sum(len(ids) for ids, _ in per_turn)
    while total > context_length + 1 and start < len(per_turn):
        total -= len(per_turn[start][0])
        start += 1
    kept = per_turn[start:]
    if not kept:
        return None

    ids, is_assistant = [], []
    for turn_ids, is_asst in kept:
        ids.extend(turn_ids)
        is_assistant.extend([is_asst] * len(turn_ids))

    if not any(is_assistant):
        return None

    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(is_assistant, dtype=bool),
    )


def make_sft_batch(examples, indices, device, pad_id):
    """Build one right-padded, masked training batch from examples[indices].

    Returns (x, y, attn_mask):
      x          (batch, seq_len-1)  input ids
      y          (batch, seq_len-1)  targets, -100 at every non-Assistant/padding
                                      position (masked_next_token_loss's ignore_index)
      attn_mask  (batch, 1, seq_len-1, seq_len-1)  additive causal+padding mask for
                                      model(x, attn_mask=attn_mask)

    Right-padding (not generate_text_batch()'s left-padding) means position 0 is real
    content for every row, so there is no fully-masked-query-row NaN case to guard
    against here — that guard in generate_text_batch exists specifically for
    left-padding's empty-prefix rows, which right-padded training batches never have.
    """
    batch = [examples[i] for i in indices]
    max_len = max(len(ids) for ids, _ in batch)

    ids_padded = np.full((len(batch), max_len), pad_id, dtype=np.int64)
    valid = np.zeros((len(batch), max_len), dtype=bool)
    label_mask = np.zeros((len(batch), max_len), dtype=bool)
    for b, (ids, is_assistant) in enumerate(batch):
        n = len(ids)
        ids_padded[b, :n] = ids
        valid[b, :n] = True
        label_mask[b, :n] = is_assistant

    ids_t = torch.from_numpy(ids_padded).to(device)
    valid_t = torch.from_numpy(valid).to(device)
    label_mask_t = torch.from_numpy(label_mask).to(device)

    x = ids_t[:, :-1]
    y = ids_t[:, 1:].clone()
    y[~label_mask_t[:, 1:]] = -100

    seq_len = x.size(1)
    causal = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1,
    )
    padding_bias = torch.zeros((len(batch), seq_len), device=device)
    padding_bias.masked_fill_(~valid_t[:, :-1], float("-inf"))
    attn_mask = causal.unsqueeze(0).unsqueeze(0) + padding_bias[:, None, None, :]

    return x, y, attn_mask
