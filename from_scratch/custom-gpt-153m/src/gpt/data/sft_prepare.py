"""SFT corpus construction — same download/parse/filter pipeline as prepare.py's
build_corpus(), but writing structured turn-preserving JSONL instead of flat text.

build_corpus() (prepare.py) is for the *pretraining* corpus, where dataset.py's
get_batch() samples random context-length windows from a flat token stream — turn
boundaries don't matter there, so turns_to_text() collapses each conversation to plain
"Role: message" lines. SFT is the opposite: masked_next_token_loss() (dataset.py) needs
to know exactly which tokens are an Assistant turn's, which flat text throws away. This
module keeps the [(role, text), ...] structure build_corpus() already produces
internally and writes it straight out, instead of flattening it.

Writes to data/sft/{train,test}.jsonl — deliberately separate from data/train.txt/
data/train.bin (the pretraining corpus this project is also using), so building the SFT
corpus can never overwrite or interfere with the base run's data.
"""

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .prepare import _download_and_parse_source
from .sources import DATASETS, selected


def sft_train_path(data_dir):
    return Path(data_dir) / "sft" / "train.jsonl"


def sft_test_path(data_dir):
    return Path(data_dir) / "sft" / "test.jsonl"


def _write_jsonl(path, conversations):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for turns in conversations:
            line = {"turns": [{"role": role, "text": text} for role, text in turns]}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def build_sft_corpus(
    data_dir,
    include_gated=True,
    token=None,
    max_per_dataset=100_000,
    min_turns=2,
    min_turn_chars=24,
    min_ascii_ratio=0.995,
    train_ratio=0.9,
    seed=42,
    skip_download=False,
    max_workers=None,
):
    """Download (unless skipped), parse, filter, shuffle, split, and write
    data/sft/{train,test}.jsonl. Same arguments and defaults as build_corpus(), same
    per-source pipeline (_download_and_parse_source), same reproducible shuffle+split —
    the only difference is the output shape (structured JSONL, not flat text).
    """
    random.seed(seed)
    data_dir = Path(data_dir)
    raw_root = data_dir / "raw"
    cache_dir = data_dir / "hf_cache"
    raw_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    chosen = selected(include_gated=include_gated)
    if not include_gated:
        skipped = ", ".join(d.hf_id for d in DATASETS if d.gated)
        print(f"[info] skipping gated dataset(s): {skipped}")

    all_conversations = []
    per_source_counts = {}

    workers = max(1, min(len(chosen), max_workers or os.cpu_count() or 1)) if chosen else 1
    print(f"[info] processing {len(chosen)} source(s) across {workers} worker process(es)")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            _download_and_parse_source,
            chosen,
            [raw_root] * len(chosen),
            [cache_dir] * len(chosen),
            [token] * len(chosen),
            [skip_download] * len(chosen),
            [max_per_dataset] * len(chosen),
            [min_turns] * len(chosen),
            [min_turn_chars] * len(chosen),
            [min_ascii_ratio] * len(chosen),
        )

        for hf_id, conversations, log_lines in results:
            for line in log_lines:
                print(line)
            per_source_counts[hf_id] = len(conversations)
            all_conversations.extend(conversations)

    if len(all_conversations) < 10:
        raise ValueError(
            "Too few valid conversations parsed. Check dataset access/schema — for the "
            "gated LMSYS set you must accept its terms and provide HF_TOKEN, or pass "
            "--no-gated to build from the public datasets only."
        )

    random.shuffle(all_conversations)
    split_idx = int(len(all_conversations) * train_ratio)
    train_rows = all_conversations[:split_idx]
    test_rows = all_conversations[split_idx:]

    _write_jsonl(sft_train_path(data_dir), train_rows)
    _write_jsonl(sft_test_path(data_dir), test_rows)

    return {
        "per_source": per_source_counts,
        "total_conversations": len(all_conversations),
        "train_conversations": len(train_rows),
        "test_conversations": len(test_rows),
    }
