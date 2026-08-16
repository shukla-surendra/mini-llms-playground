"""Token sampling and the generation loop — one canonical implementation.

Training, the CLI, the API server, and evaluation all generate through
generate_text() so sampling behavior can never diverge between them.
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
