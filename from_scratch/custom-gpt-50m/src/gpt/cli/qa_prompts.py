"""The QA prompt set — what `gpt-qa-report` asks the model.

Loaded from `data/prompt.jsonl` (one JSON object per line) rather than hardcoded here —
growing, editing, or reordering the prompt set is now a data edit, not a code change.
Each line has `category`, `kind`, and `prompt`; "mirror"-kind lines that correspond to a
registered chat source also carry `source_hf_id`:

    {"category": "UltraChat-style (bulk everyday-assistant Q&A)", "kind": "mirror",
     "prompt": "What are three simple ways to stay productive while working from home?",
     "source_hf_id": "HuggingFaceH4/ultrachat_200k"}
    {"category": "Multi-step reasoning & logic (NOT in the corpus — expected to fail)",
     "kind": "probe", "prompt": "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?"}
    {"category": "Parameter sweep", "kind": "sweep",
     "prompt": "What is the capital of France?"}

Two kinds of category (`kind: "mirror"` vs `"probe"`), plus one flat pool of sweep
prompts (`kind: "sweep"`, category ignored for grouping purposes):

**Corpus-mirroring** (`kind: "mirror"`) — one per training source (UltraChat, OASST1,
Dolly, SmolTalk, No Robots, GSM8K, Wikipedia, Books, the practice repos). These are
regression checks: they ask the kind of thing that source actually contains, so a change
in the corpus shows up as a change in these answers. Lines carrying `source_hf_id` are
gated in `qa_report.py` against `data/raw/` — see `CATEGORY_SOURCE_HF_ID` below — so a
report never claims to test a source this checkpoint never saw; lines without
`source_hf_id` (Wikipedia/Books/repo-domain/format-following) always run.

**Capability probes** (`kind: "probe"`) — reasoning, commonsense, coding, agentic
planning, practical life, instruction-following, safety, self-knowledge. These
deliberately ask for things the corpus does **not** train for. A ~50M base model is
expected to fail most of them, and that is the point: they mark the distance between
"produces fluent text" and "useful", and they are where improvement from a bigger model
or a better corpus would first become visible. Do not read a wrong answer here as a bug.
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
            f"gated 'mirror' categories). See qa_prompts.py's module docstring for the "
            f"exact shape."
        )

    categories: dict[str, list[str]] = {}
    category_kind: dict[str, str] = {}
    category_source_hf_id: dict[str, str] = {}
    sweep: list[str] = []

    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            kind = row["kind"]
            prompt = row["prompt"]

            if kind == "sweep":
                sweep.append(prompt)
                continue

            category = row["category"]
            categories.setdefault(category, []).append(prompt)
            existing_kind = category_kind.setdefault(category, kind)
            if existing_kind != kind:
                raise ValueError(
                    f"{path}:{line_no}: category {category!r} has both kind="
                    f"{existing_kind!r} and kind={kind!r} across its lines — a category "
                    f"must be entirely 'mirror' or entirely 'probe'."
                )
            if "source_hf_id" in row:
                category_source_hf_id[category] = row["source_hf_id"]

    return categories, category_kind, category_source_hf_id, sweep


_categories, _category_kind, CATEGORY_SOURCE_HF_ID, SWEEP_PROMPTS = _load(DEFAULT_PROMPTS_FILE)

#: (category label, [prompts]) — same shape qa_report.py has always consumed.
QA_CATEGORIES = list(_categories.items())

#: Categories that deliberately ask for behaviour the corpus does NOT train for.
PROBE_CATEGORIES = frozenset(
    category for category, kind in _category_kind.items() if kind == "probe"
)

#: (label, kwargs for generate_text). `do_sample=False` is greedy — the same prompt
#: always gives the same answer, which is the honest way to see what the model most
#: believes rather than what it happened to roll. Decoding settings, not prompt text,
#: so these stay in code rather than moving into prompt.jsonl.
#:
#: no_repeat_ngram_size=3 is a hard block on repeating a 3-gram, present on every row
#: (including greedy, whose repetition_penalty stays at 1.0 — a no-op per
#: apply_repetition_penalty — specifically so greedy keeps showing the model's raw,
#: un-biased belief). Without it, greedy in particular has no repetition mitigation at
#: all and reliably mode-collapses into "the same clause, forever" once it revisits a
#: high-probability cycle; the hard ngram block stops that without softly reweighting
#: what greedy picks otherwise.
SWEEP_SETTINGS = [
    ("greedy (deterministic)",
     dict(do_sample=False, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0,
          no_repeat_ngram_size=3)),
    ("conservative  T=0.3 k=20",
     dict(do_sample=True, temperature=0.3, top_k=20, top_p=0.9, repetition_penalty=1.1,
          no_repeat_ngram_size=3)),
    ("default  T=0.8 k=40 p=0.9",
     dict(do_sample=True, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1,
          no_repeat_ngram_size=3)),
    ("creative  T=1.2 k=100",
     dict(do_sample=True, temperature=1.2, top_k=100, top_p=0.95, repetition_penalty=1.1,
          no_repeat_ngram_size=3)),
]


def total_prompts():
    return sum(len(q) for _, q in QA_CATEGORIES)
