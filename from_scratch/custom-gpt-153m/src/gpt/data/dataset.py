"""Tokenization, batching, and the training loss.

Raw language modeling: every token in the text is a training target, exactly how a
base model like GPT-2 is pretrained. The corpus happens to contain chat transcripts
(see docs/DATASETS.md), but nothing here treats them specially — no chat template,
no per-turn loss masking.

## Two token-storage paths, and why

`encode_raw` (the original path) tokenizes the whole corpus into one `torch.tensor`
on the training device. That is fine at 10M-1B token scale and is what the sibling
custom-gpt-{10m,50m} projects use, but it does not survive the corpus this project
is sized for:

  * `torch.tensor(list_of_ints)` is **int64** — 8 bytes per token. A 2.5B-token
    corpus is 20 GB, on a card with 24 GB total.
  * `load_text` materializes the entire corpus as one Python `str` first — ~10 GB
    for 2.5B tokens, on a g6.xlarge with 16 GB of system RAM.

Either one OOMs before step 0. So the production path is a **flat `uint16` file**
(`train.bin`/`test.bin`) built once by `gpt-tokenize` and read back as a
`numpy.memmap`: 2 bytes per token (GPT-2's 50,257-token vocabulary fits in uint16),
5 GB on disk for 2.5B tokens, and essentially zero resident memory because the OS
pages in only the windows actually sampled.

`encode_raw` is kept for small corpora and smoke tests, where building a `.bin` is
more ceremony than it is worth.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .prepare import DOCUMENT_SEPARATOR

#: On-disk token dtype. GPT-2's vocab is 50,257, comfortably under uint16's 65,536 —
#: switching tokenizers to anything larger requires widening this (and rebuilding
#: every `.bin`, since the width is baked into the file).
TOKEN_DTYPE = np.uint16

#: Text read per streaming-tokenize chunk. Big enough that per-call overhead is
#: irrelevant, small enough that a chunk plus its token output stays comfortably
#: inside a 16 GB machine.
CHUNK_CHARS = 8 * 1024 * 1024

#: Text used to fingerprint a tokenizer. Deliberately mixes ASCII words, a digit run,
#: the document separator and a non-ASCII character, so two tokenizers that differ in
#: vocabulary size, merge table, digit handling, or special-token treatment produce
#: different ids for it.
FINGERPRINT_PROBE = "The quick brown fox 12345 " + DOCUMENT_SEPARATOR + " caf\u00e9"


def tokenizer_fingerprint(tokenizer):
    """A small dict identifying *behaviourally* which tokenizer produced a .bin.

    A `.bin` is a bare uint16 array — nothing in the bytes says which vocabulary the
    ids index into. Training on a stream produced by a different tokenizer is silent
    corruption, not an error: every id is still a valid row number, just the wrong
    one. That is a live risk here because sibling projects share a corpus directory
    while `custom-gpt-200m` uses a 32,768-entry vocabulary of its own.

    Fingerprinting behaviour rather than a name also catches the subtler case: a
    tokenizer retrained at the same vocab size produces different merges, so old
    `.bin` files silently become wrong.
    """
    ids = tokenizer.encode(FINGERPRINT_PROBE, disallowed_special=())
    digest = hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()[:16]
    return {"n_vocab": int(tokenizer.n_vocab), "probe_ids": len(ids), "probe_sha256": digest}


def bin_meta_path(bin_path):
    return Path(bin_path).with_suffix(".bin.json")



def load_text(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` to download and build the corpus first."
        )
    return path.read_text(encoding="utf-8")


def encode_raw(tokenizer, text, device):
    """Tokenize text into one flat token stream to train over.

    `disallowed_special=()` matters even though DOCUMENT_SEPARATOR (plain "\\n\\n") is
    not itself a special token: by default tiktoken *raises* if the input text
    contains the literal substring of any of GPT-2's real reserved special tokens
    (e.g. an actual "<|endoftext|>" appearing inside some source document's real
    content, unrelated to this project's own boundary marker). `disallowed_special=()`
    means "never raise on that, just tokenize it as ordinary subword pieces" — this
    project deliberately never invokes the special-token id path anywhere, so a
    literal occurrence in the corpus is treated as plain text, not specially
    recognized and not a fatal error either.

    Prefer `load_token_array` for anything large enough to care about — see this
    module's docstring for why this path does not scale.
    """
    return torch.tensor(
        tokenizer.encode(text, disallowed_special=()),
        device=device,
    )


def encode_chunk(tokenizer, text):
    """`encode_raw`'s tokenizer call, without building a tensor. Same
    `disallowed_special=()` handling — see `encode_raw` for why it's needed even
    though DOCUMENT_SEPARATOR isn't a special token."""
    return tokenizer.encode(text, disallowed_special=())


def token_bin_path(text_path):
    """`data/train.txt` -> `data/train.bin`."""
    return Path(text_path).with_suffix(".bin")


def _split_at_document_boundary(buffer, is_final, max_buffer):
    """Return (encodable_now, carry_over).

    Chunking must be invisible to the tokenizer: the token stream has to come out
    identical to tokenizing the whole file at once. A naive fixed-size read can split
    mid-word, which stops BPE from forming merges across the cut and costs real
    tokens. (Measured: encoding a test corpus in 23-char chunks that way produced
    5,418 tokens where the whole file produces 4,375 — a 24% inflation of pure noise.)

    So we prefer cutting at a document separator — DOCUMENT_SEPARATOR ("\\n\\n"), a
    plain whitespace run, so a cut there can never land mid-word — carrying an
    incomplete tail forward until one shows up. `max_buffer` bounds that carry for
    the pathological case of a single document longer than the cap; there we cut
    immediately *before* a space instead, which GPT-2's pre-tokenizer treats as a
    word boundary (a leading space binds to the following word, so " word" stays one
    token) — the same safety property as a document-separator cut, just without
    waiting for one to appear.
    """
    if is_final:
        return buffer, ""
    cut = buffer.rfind(DOCUMENT_SEPARATOR)
    if cut != -1:
        cut += len(DOCUMENT_SEPARATOR)
        return buffer[:cut], buffer[cut:]
    if len(buffer) < max_buffer:
        return "", buffer  # nothing safe to encode yet — keep reading
    space = buffer.rfind(" ")
    if space <= 0:
        return buffer, ""  # no whitespace at all; accept the split rather than hang
    return buffer[:space], buffer[space:]


def build_token_bin(tokenizer, text_path, bin_path=None, chunk_chars=CHUNK_CHARS,
                    max_buffer_chars=None, progress=None):
    """Stream-tokenize `text_path` into a flat uint16 `.bin`. Returns the token count.

    Streams rather than reading the file whole, so corpus size is bounded by disk
    rather than by RAM.
    """
    text_path = Path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(f"{text_path} not found — nothing to tokenize.")
    bin_path = Path(bin_path) if bin_path else token_bin_path(text_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    # Cap on how much separator-free text we will hold before force-splitting. Real
    # documents here are ~1024 tokens (~4 KB), so this never triggers in practice.
    max_buffer = max_buffer_chars or max(chunk_chars * 4, 1 << 20)

    tmp_path = bin_path.with_suffix(bin_path.suffix + ".tmp")
    total = 0
    carry = ""
    max_id = 0

    with open(text_path, "r", encoding="utf-8") as src, open(tmp_path, "wb") as dst:
        while True:
            chunk = src.read(chunk_chars)
            is_final = not chunk
            buffer = carry + chunk
            if not buffer:
                break
            encodable, carry = _split_at_document_boundary(buffer, is_final, max_buffer)
            if encodable:
                ids = encode_chunk(tokenizer, encodable)
                if ids:
                    max_id = max(max_id, max(ids))
                    np.asarray(ids, dtype=TOKEN_DTYPE).tofile(dst)
                    total += len(ids)
                    if progress:
                        progress(total)
            if is_final:
                break

    if max_id > np.iinfo(TOKEN_DTYPE).max:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Token id {max_id} exceeds {TOKEN_DTYPE.__name__}'s range — widen "
            f"dataset.TOKEN_DTYPE and rebuild every .bin."
        )

    bin_meta_path(bin_path).write_text(json.dumps(
        {"tokenizer": tokenizer_fingerprint(tokenizer), "tokens": total,
         "source": str(text_path)}, indent=2))
    tmp_path.replace(bin_path)  # atomic: a killed run never leaves a half-written .bin
    return total


def load_token_array(text_path, tokenizer=None, rebuild=False):
    """Memmap `<stem>.bin`, building it from `<stem>.txt` first if needed.

    Returns a read-only `numpy.memmap` of `TOKEN_DTYPE`. Rebuilds when the `.bin` is
    missing, older than its source text, or `rebuild=True` — so editing the corpus
    can't silently train on stale tokens.
    """
    text_path = Path(text_path)
    bin_path = token_bin_path(text_path)

    stale = (
        rebuild
        or not bin_path.exists()
        or (text_path.exists() and bin_path.stat().st_mtime < text_path.stat().st_mtime)
    )
    if stale:
        if tokenizer is None:
            raise ValueError(
                f"{bin_path} is missing or stale and no tokenizer was supplied. "
                f"Run `gpt-tokenize` to build it."
            )
        build_token_bin(tokenizer, text_path, bin_path)


    meta_path = bin_meta_path(bin_path)
    if tokenizer is not None and meta_path.exists():
        stored = json.loads(meta_path.read_text())
        current = tokenizer_fingerprint(tokenizer)
        if stored.get("tokenizer") != current:
            raise ValueError(
                f"{bin_path} was built by a DIFFERENT tokenizer than the one configured "
                f"now.\n"
                f"  stored : {stored.get('tokenizer')}\n"
                f"  current: {current}\n"
                f"Token ids are vocabulary-specific: training on this file would index "
                f"the wrong embedding rows silently rather than fail. Rebuild with "
                f"`gpt-tokenize --force` (and note that any checkpoint trained on the "
                f"old stream is not resumable against the new one)."
            )

    return np.memmap(bin_path, dtype=TOKEN_DTYPE, mode="r")


def effective_context_length(configured, *token_streams):
    """Shrink the context window if a dataset is too small to fill it.

    get_batch needs at least ctx_len+1 tokens to build an (input, target) pair, so a
    small smoke-test corpus would otherwise crash instead of just training short.
    """
    limit = min(len(stream) - 1 for stream in token_streams)
    return max(1, min(configured, limit))


def get_batch(tokens, ctx_len, batch_size, device):
    """Random windows of `ctx_len` tokens; targets are the same window shifted by one.

    Slicing happens on the host and only the assembled batch crosses to the device —
    which is what lets the token stream stay a disk-backed memmap instead of having to
    live in VRAM. The copy is trivial next to the step itself (batch 16 x 1024 int64 is
    128 KB), and `pin_memory` + `non_blocking` overlap it with compute on CUDA.
    """
    if isinstance(tokens, torch.Tensor):
        # Legacy in-memory path (encode_raw): index directly, already on-device.
        max_start = len(tokens) - ctx_len
        ix = torch.randint(0, max_start, (batch_size,), device=tokens.device)
        x = torch.stack([tokens[i:i + ctx_len] for i in ix])
        y = torch.stack([tokens[i + 1:i + ctx_len + 1] for i in ix])
        return x.to(device), y.to(device)

    max_start = len(tokens) - ctx_len - 1
    if max_start < 1:
        raise ValueError(
            f"Corpus has {len(tokens):,} tokens, too few for context_length={ctx_len}."
        )
    ix = np.random.randint(0, max_start, size=batch_size)
    # astype(int64) also materialises the memmap windows into real memory.
    x = torch.from_numpy(
        np.stack([tokens[i:i + ctx_len] for i in ix]).astype(np.int64)
    )
    y = torch.from_numpy(
        np.stack([tokens[i + 1:i + ctx_len + 1] for i in ix]).astype(np.int64)
    )
    if device.startswith("cuda") if isinstance(device, str) else False:
        return (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    return x.to(device), y.to(device)


def next_token_loss(logits, targets, vocab_size):
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def masked_next_token_loss(logits, labels, vocab_size, ignore_index=-100):
    """Same next-token cross-entropy as next_token_loss, but positions in `labels`
    equal to `ignore_index` contribute no gradient — the SFT loss-masking mechanism:
    only assistant-turn tokens are ever real labels (see data/sft_dataset.py), so the
    model is updated on its own responses, not on the prompt/user turns it was given.
    """
    return F.cross_entropy(
        logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=ignore_index,
    )
