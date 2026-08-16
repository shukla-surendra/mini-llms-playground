"""The QA prompt set — what `gpt-qa-report` asks the model.

Loaded from `data/prompt.jsonl` (one JSON object per line) rather than hardcoded here —
growing, editing, or reordering the prompt set is now a data edit, not a code change.
Each line has `category`, `kind`, and `prompt`; lines corresponding to a registered chat
source also carry `source_hf_id`:

    {"category": "UltraChat-style (bulk everyday-assistant Q&A)", "kind": "mirror",
     "prompt": "What are three simple ways to stay productive while working from home?",
     "source_hf_id": "HuggingFaceH4/ultrachat_200k"}

Every category here is `kind: "mirror"` — one per training source (see docs/DATASETS.md),
a regression check: it asks the kind of thing that source actually contains, so a change
in the corpus shows up as a change in these answers. `kind` is still recorded per line
(rather than assumed) for schema consistency with the sibling custom-gpt-50m project's
prompt.jsonl, which also has "probe"/"sweep" kinds this project doesn't use.

Categories carrying `source_hf_id` are gated in `qa_report.py` against `data/raw/` — see
`CATEGORY_SOURCE_HF_ID` below — so a report never claims to test a source this checkpoint
never saw (relevant for the gated LMSYS-Chat-1M set, skipped by `make data-public`).
Categories without `source_hf_id` (e.g. Format-following) always run.
"""

import json
from pathlib import Path

DEFAULT_PROMPTS_FILE = Path("data") / "prompt.jsonl"


def _load(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — gpt-qa-report's prompt set now lives entirely in this "
            f"file (one JSON object per line: category/kind/prompt, +source_hf_id for "
            f"gated categories). See qa_prompts.py's module docstring for the exact shape."
        )

    categories: dict[str, list[str]] = {}
    category_source_hf_id: dict[str, str] = {}

    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            category = row["category"]
            categories.setdefault(category, []).append(row["prompt"])
            if "source_hf_id" in row:
                category_source_hf_id[category] = row["source_hf_id"]

    return categories, category_source_hf_id


_categories, CATEGORY_SOURCE_HF_ID = _load(DEFAULT_PROMPTS_FILE)

#: (category label, [prompts]) — same shape qa_report.py has always consumed.
QA_CATEGORIES = list(_categories.items())


def total_prompts():
    return sum(len(q) for _, q in QA_CATEGORIES)
