"""Heuristic generation-quality scoring, tracked across runs.

Loss going down does not guarantee output getting better — these cheap surface
metrics (empty output, repetition, non-ASCII noise, redaction junk) catch the common
"loss improved but generations got worse" case that a loss curve alone hides.
"""

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from ..inference.generate import generate_text


def word_repetition_ratio(text):
    words = [w for w in text.lower().split() if w]
    if not words:
        return 1.0
    return 1.0 - (len(set(words)) / len(words))


def ascii_ratio(text):
    if not text:
        return 1.0
    return sum(ord(c) < 128 for c in text) / len(text)


def placeholder_noise_ratio(text):
    if not text:
        return 0.0
    return 1.0 if any(m in text for m in ("NAME_", "��", "�")) else 0.0


def load_last_report(path):
    path = Path(path)
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line.strip()
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def append_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")


def evaluate(model, tokenizer, checkpoint, prompts, device, *, max_new_tokens=100,
             do_sample=True, temperature=0.8, top_k=40, top_p=0.9,
             repetition_penalty=1.1, checkpoint_path="", examples_to_keep=3):
    non_empty = 0
    ascii_scores, repeat_scores, placeholder_noise, lengths = [], [], [], []
    role_leak_count = 0
    examples = []

    for prompt in prompts:
        # postprocess=False: role_leak_rate must SEE a leaked "\nUser:" continuation,
        # so trimming at role markers here would silently zero out the metric.
        _, completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            context_length=checkpoint["context_length"],
            max_new_tokens=max_new_tokens,
            device=device,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            postprocess=False,
        )
        if completion.strip():
            non_empty += 1
        if "\nUser:" in completion or "\nSystem:" in completion:
            role_leak_count += 1
        ascii_scores.append(ascii_ratio(completion))
        repeat_scores.append(word_repetition_ratio(completion))
        placeholder_noise.append(placeholder_noise_ratio(completion))
        lengths.append(len(completion))
        if len(examples) < examples_to_keep:
            examples.append((prompt, completion))

    n = max(1, len(prompts))
    score = 100.0
    score -= (1.0 - (non_empty / n)) * 35.0
    score -= max(0.0, statistics.mean(repeat_scores) - 0.35) * 35.0
    score -= max(0.0, 0.96 - statistics.mean(ascii_scores)) * 100.0
    score -= (role_leak_count / n) * 15.0
    score -= statistics.mean(placeholder_noise) * 15.0
    score = max(0.0, min(100.0, score))

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_best_test_loss": float(checkpoint.get("best_test_loss", -1.0)),
        "model_label": checkpoint.get("preset_label", "unknown"),
        "param_count": checkpoint.get("param_count"),
        "device": device,
        "do_sample": bool(do_sample),
        "prompts_count": len(prompts),
        "non_empty_rate": non_empty / n,
        "avg_completion_chars": statistics.mean(lengths) if lengths else 0.0,
        "avg_ascii_ratio": statistics.mean(ascii_scores) if ascii_scores else 1.0,
        "avg_word_repetition_ratio": statistics.mean(repeat_scores) if repeat_scores else 1.0,
        "role_leak_rate": role_leak_count / n,
        "placeholder_noise_rate": statistics.mean(placeholder_noise) if placeholder_noise else 0.0,
        "heuristic_quality_score_0_to_100": score,
    }
    return report, examples


def format_report(report, previous=None, examples=()):
    lines = [
        "=== Quality Report ===",
        f"model: {report['model_label']} ({report.get('param_count') or '?'} params)",
        f"checkpoint: {report['checkpoint']} (step {report['checkpoint_step']})",
        f"best_test_loss: {report['checkpoint_best_test_loss']:.4f}",
        f"prompts: {report['prompts_count']}",
        f"non_empty_rate: {report['non_empty_rate']:.3f}",
        f"avg_completion_chars: {report['avg_completion_chars']:.1f}",
        f"avg_ascii_ratio: {report['avg_ascii_ratio']:.3f}",
        f"avg_word_repetition_ratio: {report['avg_word_repetition_ratio']:.3f}",
        f"role_leak_rate: {report['role_leak_rate']:.3f}",
        f"placeholder_noise_rate: {report['placeholder_noise_rate']:.3f}",
        f"heuristic_quality_score_0_to_100: {report['heuristic_quality_score_0_to_100']:.1f}",
    ]

    if previous:
        delta = (report["heuristic_quality_score_0_to_100"]
                 - previous.get("heuristic_quality_score_0_to_100", 0.0))
        delta_loss = (report["checkpoint_best_test_loss"]
                      - previous.get("checkpoint_best_test_loss",
                                     report["checkpoint_best_test_loss"]))
        lines += [
            "",
            "=== Delta vs Previous Eval ===",
            f"quality_score_delta: {delta:+.2f}",
            f"best_test_loss_delta: {delta_loss:+.4f}",
        ]

    if examples:
        lines += ["", "=== Sample Outputs ==="]
        for i, (prompt, completion) in enumerate(examples, start=1):
            lines += [f"\n[{i}] PROMPT:\n{prompt}\n", "COMPLETION:",
                      completion if completion else "[empty]"]

    return "\n".join(lines)
