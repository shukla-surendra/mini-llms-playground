"""Corpus quality audit — run before committing to a long training run.

Cheap sanity gate: if the corpus is full of non-ASCII noise, redaction placeholders,
or the test split duplicates the train split, no amount of training will produce a
usable model.
"""

import statistics

from .dataset import load_text

NOISE_MARKERS = ("NAME_", "PERSON_", "EMAIL_", "URL_", "�")


def line_stats(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}

    lengths = [len(ln) for ln in lines]
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    noise_lines = sum(1 for ln in lines if any(m in ln for m in NOISE_MARKERS))

    assistant_lines = [ln for ln in lines if ln.startswith("Assistant:")]
    user_lines = [ln for ln in lines if ln.startswith("User:")]
    system_lines = [ln for ln in lines if ln.startswith("System:")]

    return {
        "chars": len(text),
        "lines": len(lines),
        "mean_line_chars": statistics.mean(lengths),
        "median_line_chars": statistics.median(lengths),
        "ascii_ratio": ascii_chars / max(1, len(text)),
        "noise_line_rate": noise_lines / len(lines),
        "assistant_lines": len(assistant_lines),
        "user_lines": len(user_lines),
        "system_lines": len(system_lines),
        "assistant_lines_set": {ln for ln in assistant_lines},
    }


def overlap_rate(train_stats, test_stats):
    """Fraction of test assistant turns that appear verbatim in train — leakage check."""
    test_set = test_stats.get("assistant_lines_set") or set()
    train_set = train_stats.get("assistant_lines_set") or set()
    if not test_set:
        return 0.0
    return len(test_set & train_set) / len(test_set)


def audit(paths):
    train_stats = line_stats(load_text(paths.train_data))
    test_stats = line_stats(load_text(paths.test_data))
    return {
        "train": train_stats,
        "test": test_stats,
        "assistant_exact_overlap_rate_vs_test": overlap_rate(train_stats, test_stats),
    }


def format_report(report):
    lines = []
    for split in ("train", "test"):
        s = report[split]
        if not s:
            lines.append(f"[{split}] empty")
            continue
        lines.append(
            f"[{split}] chars={s['chars']:,} lines={s['lines']:,} "
            f"mean_line_chars={s['mean_line_chars']:.1f} "
            f"ascii_ratio={s['ascii_ratio']:.4f} "
            f"noise_line_rate={s['noise_line_rate']:.4f}"
        )
        lines.append(
            f"        roles: system={s['system_lines']:,} "
            f"user={s['user_lines']:,} assistant={s['assistant_lines']:,}"
        )
    lines.append(
        f"[leakage] assistant_exact_overlap_rate_vs_test="
        f"{report['assistant_exact_overlap_rate_vs_test']:.4f}"
    )
    lines.append("")
    lines.append("Acceptance guide: noise_line_rate < 0.01, ascii_ratio > 0.98, overlap near 0.0")
    return "\n".join(lines)
