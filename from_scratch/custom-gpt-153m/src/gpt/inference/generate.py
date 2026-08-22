"""Token sampling and the generation loop — one canonical implementation.

Training, the CLI, the API server, and evaluation all generate through
generate_text() so sampling behavior can never diverge between them.

generate_text_batch() is a separate, batched counterpart used only by
gpt-qa-report (many independent prompts, no shared context) — same
sample_next_token/postprocess_completion math per row, just N prompts sharing
one forward pass per step instead of N independent passes. See its own
docstring for the left-padding/masking mechanism that keeps its output
equivalent to calling generate_text() once per prompt.
"""

import re

import torch

_SENTENCE_END_RE = re.compile(r'[.!?][\'")\]]*(?=\s|$)')


def apply_repetition_penalty(next_logits, ids, penalty=1.0, window_size=None):
    """Discourage repeating recently-generated tokens by shrinking their logits.

    penalty <= 1.0 is a no-op (used by callers that never wanted this).
    """
    if penalty <= 1.0:
        return next_logits
    adjusted = next_logits.clone()
    effective_window = ids.size(1) if window_size is None else max(1, int(window_size))
    recent_ids = ids[0, -effective_window:].unique()
    adjusted[:, recent_ids] = adjusted[:, recent_ids] / penalty
    return adjusted


def apply_repetition_penalty_batch(next_logits, ids, valid_mask, penalty=1.0, window_size=None):
    """Batched counterpart to apply_repetition_penalty — same per-row effect, but only
    over that row's REAL tokens (valid_mask=True), so a left-padding filler token is
    never treated as a "repeated" token. ids/valid_mask share shape (batch, seq_len).
    """
    if penalty <= 1.0:
        return next_logits
    adjusted = next_logits.clone()
    effective_window = ids.size(1) if window_size is None else max(1, int(window_size))
    window_ids = ids[:, -effective_window:]
    window_valid = valid_mask[:, -effective_window:]
    for b in range(ids.size(0)):
        recent_ids = window_ids[b][window_valid[b]].unique()
        if recent_ids.numel():
            adjusted[b, recent_ids] = adjusted[b, recent_ids] / penalty
    return adjusted


def sample_next_token(logits, do_sample=True, temperature=1.0, top_k=None, top_p=None):
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, k, dim=-1).values[..., -1].unsqueeze(-1)
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)

    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
        probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    return torch.multinomial(probs, num_samples=1)


def _trim_to_complete_sentence(text):
    """Trim back to the last '.'/'!'/'?' so callers never see a completion that
    stops mid-word. generate_text's loop has no early-stop condition — it always
    runs exactly max_new_tokens steps and returns whatever that lands on, which is
    as likely to be mid-word as at a sentence boundary. If no sentence end is found
    at all (very short/degenerate completions), return the text unchanged rather
    than trimming it to nothing."""
    matches = list(_SENTENCE_END_RE.finditer(text))
    if not matches:
        return text
    return text[:matches[-1].end()].rstrip()


def postprocess_completion(text):
    """Trim a chat-style completion at the next role marker, drop a leading
    'Assistant:' echo if the model reproduced it, and trim any trailing sentence
    fragment left by running out of the generation token budget."""
    cleaned = text.lstrip()
    if cleaned.startswith("Assistant:"):
        cleaned = cleaned[len("Assistant:"):].lstrip()
    for marker in ("\nUser:", "\nSystem:", "\nAssistant:"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx]
    return _trim_to_complete_sentence(cleaned.strip())


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    context_length,
    max_new_tokens,
    device,
    do_sample=True,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=1.0,
    postprocess=True,
):
    """Returns (full_text, completion) — completion is full_text with the prompt
    stripped off, and (if postprocess=True) any trailing role-marker chatter trimmed
    too (see postprocess_completion).

    Pass postprocess=False when the raw completion matters, e.g. eval_quality.py's
    role_leak_rate metric specifically needs to SEE leaked '\\nUser:'/'\\nSystem:'
    continuations rather than have them silently trimmed away before it can count them.
    """
    model.eval()
    ids = torch.tensor(
        tokenizer.encode(prompt, disallowed_special=()),
        device=device,
    ).unsqueeze(0)
    ids = ids[:, -context_length:]  # cap an over-long prompt, same as the old sliding window

    use_kv_cache = getattr(model, "attn_impl", "naive") == "sdpa"

    if use_kv_cache:
        # Prefill: one forward pass over the whole prompt, building the initial cache.
        # Every decode step after this processes exactly ONE new token instead of
        # reprocessing the whole sequence-so-far — see model.py's "KV caching" docstring
        # section. Capped so the KV cache (and position embeddings) never grow past
        # context_length; an already-full prompt yields 0 decode steps rather than
        # erroring, same spirit as the old code silently working with whatever fit.
        logits, past_kv = model(ids, use_cache=True)
        steps = max(0, min(max_new_tokens, context_length - ids.size(1)))

        for _ in range(steps):
            next_logits = apply_repetition_penalty(
                logits[:, -1, :], ids, penalty=repetition_penalty, window_size=context_length,
            )
            next_token = sample_next_token(
                next_logits, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
            )
            ids = torch.cat([ids, next_token], dim=1)
            logits, past_kv = model(next_token, past_kv=past_kv, use_cache=True, start_pos=ids.size(1) - 1)
    else:
        # "naive" attn_impl has no incremental-decoding path (see model.py) — fall back
        # to the original full-reprocess-every-step loop. Reached only when generate_text
        # is called with a model built directly under attn_impl="naive" (e.g. during
        # training's periodic sample generation) rather than via checkpoint.load_model(),
        # which defaults inference to "sdpa" specifically to avoid this path.
        for _ in range(max_new_tokens):
            window = ids[:, -context_length:]
            logits = model(window)
            next_logits = apply_repetition_penalty(
                logits[:, -1, :], ids, penalty=repetition_penalty, window_size=context_length,
            )
            next_token = sample_next_token(
                next_logits, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
            )
            ids = torch.cat([ids, next_token], dim=1)

    full_text = tokenizer.decode(ids[0].tolist())
    raw_completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    completion = postprocess_completion(raw_completion) if postprocess else raw_completion.strip()
    return full_text, completion


@torch.no_grad()
def generate_text_batch(
    model,
    tokenizer,
    prompts,
    context_length,
    max_new_tokens,
    device,
    do_sample=True,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=1.0,
    postprocess=True,
):
    """Batched counterpart to generate_text() — same sampling math (sample_next_token,
    postprocess_completion reused unchanged), same per-sequence generation semantics,
    but N independent prompts share one forward pass per step instead of running one
    at a time. Requires attn_impl="sdpa" (checkpoint.load_model()'s inference default):
    only that path implements both the KV cache and the attn_mask parameter this
    function needs — the "naive" CausalSelfAttention path has neither.

    Left-padding is what makes this work with no per-row bookkeeping at read-out time:
    every prompt is padded on the LEFT to the batch's longest prompt length, so the
    real prompt content — whatever its original length — always ends at the same final
    column. That means `logits[:, -1, :]` is the valid next-token prediction for every
    row after a single shared prefill call. Padding columns are excluded from attention
    via an explicit additive attn_mask (ordinary causal masking combined with a per-row
    padding mask) threaded through model.py's attn_mask parameter — plain causal
    masking alone does NOT exclude padding, since padding sits causally *before* the
    real tokens and would otherwise be attended to like any other prefix token.

    A subtlety this function has to handle that a naive port of the single-sequence
    loop would miss: a fully-padding query row (the columns left of a shorter prompt's
    real start) has EVERY key masked out by causal+padding combined, which produces an
    all -inf softmax row -> NaN. That NaN would land in the padding position's hidden
    state, get projected into K/V by the next layer, and then poison every later row's
    softmax that attends to that key — even though the key column is masked with -inf,
    `-inf + NaN == NaN`, not -inf, so the additive mask alone doesn't stop it once a NaN
    already exists upstream. The fix applied below: any row whose entire key mask is
    -inf gets its own diagonal (self) entry force-unmasked, producing a finite (if
    meaningless) value there instead of NaN. That position's output is never read (only
    the real, right-aligned tokens matter), so this is a pure numerical-stability fix
    with no effect on the actual generated text.

    Returns a list of (full_text, completion) tuples, one per prompt, same per-item
    contract as generate_text(), in the same order as `prompts`.
    """
    assert getattr(model, "attn_impl", "naive") == "sdpa", (
        "generate_text_batch requires attn_impl='sdpa' (checkpoint.load_model()'s "
        "inference default) — the 'naive' path has no attn_mask/KV-cache support."
    )
    model.eval()
    batch = len(prompts)

    pad_id = 0  # always masked out of attention and never read back out, so its
                # actual value is functionally arbitrary — 0 is just any valid token id.
    encoded = [
        tokenizer.encode(p, disallowed_special=())[-context_length:] for p in prompts
    ]
    prompt_lens = [len(e) for e in encoded]
    max_len = max(prompt_lens)

    ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
    valid = torch.zeros((batch, max_len), dtype=torch.bool, device=device)
    for b, enc in enumerate(encoded):
        pad_len = max_len - len(enc)
        ids[b, pad_len:] = torch.tensor(enc, device=device)
        valid[b, pad_len:] = True

    pad_lens = torch.tensor([max_len - l for l in prompt_lens], device=device)
    position_ids = (
        torch.arange(max_len, device=device).unsqueeze(0) - pad_lens.unsqueeze(1)
    ).clamp(min=0)

    causal = torch.triu(
        torch.full((max_len, max_len), float("-inf"), device=device), diagonal=1,
    )
    padding_bias = torch.zeros((batch, max_len), device=device)
    padding_bias.masked_fill_(~valid, float("-inf"))
    attn_mask = causal.unsqueeze(0).unsqueeze(0) + padding_bias[:, None, None, :]
    fully_masked = torch.isinf(attn_mask).all(dim=-1).squeeze(1)  # (batch, max_len)
    if fully_masked.any():
        b_idx, i_idx = fully_masked.nonzero(as_tuple=True)
        attn_mask[b_idx, 0, i_idx, i_idx] = 0.0  # see docstring: NaN-propagation fix

    logits, past_kv = model(ids, use_cache=True, attn_mask=attn_mask, position_ids=position_ids)
    steps = max(0, min(max_new_tokens, context_length - max_len))

    for _ in range(steps):
        next_logits = apply_repetition_penalty_batch(
            logits[:, -1, :], ids, valid, penalty=repetition_penalty, window_size=context_length,
        )
        next_token = sample_next_token(
            next_logits, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
        )
        ids = torch.cat([ids, next_token], dim=1)

        step_position_ids = valid.sum(dim=1, keepdim=True)
        valid = torch.cat([valid, torch.ones((batch, 1), dtype=torch.bool, device=device)], dim=1)
        step_attn_mask = torch.zeros((batch, 1, 1, valid.size(1)), device=device)
        step_attn_mask.masked_fill_(~valid[:, None, None, :], float("-inf"))

        logits, past_kv = model(
            next_token, past_kv=past_kv, use_cache=True,
            position_ids=step_position_ids, attn_mask=step_attn_mask,
        )

    results = []
    for b, prompt in enumerate(prompts):
        full_text = tokenizer.decode(ids[b][valid[b]].tolist())
        raw_completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        completion = postprocess_completion(raw_completion) if postprocess else raw_completion.strip()
        results.append((full_text, completion))
    return results
