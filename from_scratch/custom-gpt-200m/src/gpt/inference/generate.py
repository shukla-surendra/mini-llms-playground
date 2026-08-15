"""Token sampling and the generation loop — one canonical implementation.

Training, the CLI, the API server, and evaluation all generate through
generate_text() so sampling behavior can never diverge between them.
"""

import re

import torch

from ..tokenizer import END_OF_TEXT as DOCUMENT_SEPARATOR

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
    eot_id = getattr(tokenizer, "eot_id", None)
    ids = torch.tensor(
        tokenizer.encode(prompt, allowed_special={DOCUMENT_SEPARATOR}, disallowed_special=()),
        device=device,
    ).unsqueeze(0)

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

        # Stop at the document boundary instead of generating through it. The sibling
        # projects have no early stop: they always run the full max_new_tokens, so once
        # the model correctly predicts "this reply is over" the loop keeps sampling and
        # what follows is the start of an unrelated document — which is exactly how a
        # hallucinated second "User:" turn ends up in their QA reports.
        if eot_id is not None and int(next_token.item()) == eot_id:
            break

    full_text = tokenizer.decode(ids[0].tolist())
    raw_completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    completion = postprocess_completion(raw_completion) if postprocess else raw_completion.strip()
    return full_text, completion
