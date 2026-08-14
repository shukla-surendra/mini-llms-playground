"""Token sampling and the generation loop — one canonical implementation.

Training, the CLI, the API server, and evaluation all generate through
generate_text() so sampling behavior can never diverge between them.
"""

import torch


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


def postprocess_completion(text):
    """Trim a chat-style completion at the next role marker, and drop a leading
    'Assistant:' echo if the model reproduced it."""
    cleaned = text.lstrip()
    if cleaned.startswith("Assistant:"):
        cleaned = cleaned[len("Assistant:"):].lstrip()
    for marker in ("\nUser:", "\nSystem:", "\nAssistant:"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx]
    return cleaned.strip()


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
    ids = torch.tensor(tokenizer.encode(prompt, disallowed_special=()), device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        window = ids[:, -context_length:]
        logits = model(window)
        next_logits = apply_repetition_penalty(
            logits[:, -1, :],
            ids,
            penalty=repetition_penalty,
            window_size=context_length,
        )
        next_token = sample_next_token(
            next_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        ids = torch.cat([ids, next_token], dim=1)

    full_text = tokenizer.decode(ids[0].tolist())
    raw_completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    completion = postprocess_completion(raw_completion) if postprocess else raw_completion.strip()
    return full_text, completion
