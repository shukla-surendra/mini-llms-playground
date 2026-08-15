"""Corpus construction: download the registered datasets, normalize, filter, split.

Pipeline:
    HF dataset repos (sources.DATASETS)                    local extra_documents
      -> download parquet shards into data/raw/<slug>/     (e.g. tools/corpus-extractor
      -> parse each row into [(role, text), ...] turns      output — see
      -> quality filters (length, printable/ascii ratio,    docs/BOOKS_CORPUS_INTEGRATION.md)
         placeholder junk)                                        |
              \\                                                   /
               `-------------------- pooled, shuffled together ---'
                          |
                 split 90/10 -> data/train.txt, data/test.txt, data/test_prompts.txt

Output is plain text — "Role: message" lines (or, for extra_documents, whatever text
they already are), documents separated by DOCUMENT_SEPARATOR. Training reads it as an
undifferentiated token stream (see dataset.py).

DOCUMENT_SEPARATOR is a real reserved tokenizer special token, not a soft "\\n\\n" —
see docs/TRAINING_QA.md's "conversation-boundary marking" finding for why a soft
separator was a real, measurable weak spot: dataset.py's get_batch() samples random
windows, so a meaningful fraction of every batch straddles two unrelated documents, and
"\\n\\n" is a far weaker "this is unrelated, start fresh" signal than a token GPT-2's own
pretraining already primed the embedding space to recognize.
"""

import glob
import json
import random
import re
from pathlib import Path

from .sources import DATASETS, selected

PROMPT_FILE_SEPARATOR = "\n\n<END_PROMPT>\n\n"
DOCUMENT_SEPARATOR = "<|endoftext|>"

ROLE_MAP = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System",
    "human": "User",
    "gpt": "Assistant",
}
ALLOWED_ROLES = {"System", "User", "Assistant"}

# Some sources (notably LMSYS) redact entities into placeholders like NAME_1. Those
# leak into generations as nonsense, so drop any turn containing them.
PLACEHOLDER_PATTERNS = ("NAME_", "PERSON_", "EMAIL_", "URL_")


def normalize_role(role):
    if role is None:
        return None
    return ROLE_MAP.get(str(role).strip().lower())


def clean_text(text):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("�", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_quality_text(text, min_chars, min_ascii_ratio):
    """Reject turns that would teach the model noise rather than language."""
    if len(text) < min_chars:
        return False
    printable_ratio = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
    if printable_ratio < 0.98:
        return False
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / len(text)
    if ascii_ratio < min_ascii_ratio:
        return False
    alpha_count = sum(ch.isalpha() for ch in text)
    if alpha_count < max(5, len(text) // 25):
        return False
    if any(p in text for p in PLACEHOLDER_PATTERNS):
        return False
    if text.count("```") > 2:
        return False
    return True


def extract_turns_conversation(row, min_chars, min_ascii_ratio):
    """Schema: a list of {role/from, content/value} dicts (UltraChat, OASST1, LMSYS)."""
    conv = row.get("conversation") or row.get("conversations") or row.get("messages")
    if not isinstance(conv, list):
        return []

    turns = []
    for item in conv:
        if not isinstance(item, dict):
            continue
        role = normalize_role(item.get("role") or item.get("from"))
        if role not in ALLOWED_ROLES:
            continue
        text = item.get("content") or item.get("value")
        if not role or not text:
            continue
        text = clean_text(text)
        if not text or not is_quality_text(text, min_chars, min_ascii_ratio):
            continue
        turns.append((role, text))
    return turns


def extract_turns_instruction(row, min_chars, min_ascii_ratio):
    """Schema: flat instruction/input/output columns (Dolly, alpaca-likes)."""
    instruction = row.get("instruction") or row.get("prompt") or row.get("question")
    input_text = row.get("input")
    output = (
        row.get("output") or row.get("response") or row.get("answer") or row.get("completion")
    )
    if not instruction or not output:
        return []

    user_text = clean_text(instruction)
    if input_text:
        user_text = f"{user_text}\n{clean_text(input_text)}"
    assistant_text = clean_text(output)
    if not is_quality_text(user_text, min_chars, min_ascii_ratio):
        return []
    if not is_quality_text(assistant_text, min_chars, min_ascii_ratio):
        return []
    return [("User", user_text), ("Assistant", assistant_text)]


def extract_turns(row, min_chars, min_ascii_ratio):
    """Try the conversation schema, fall back to the instruction schema."""
    turns = extract_turns_conversation(row, min_chars, min_ascii_ratio)
    if turns:
        return turns
    return extract_turns_instruction(row, min_chars, min_ascii_ratio)


def turns_to_text(turns):
    return "\n".join(f"{role}: {text}" for role, text in turns)


def prompt_from_turns(turns):
    """Build a held-out prompt that stops right before the final assistant reply."""
    if len(turns) < 2:
        return None

    last_assistant_idx = -1
    for i in range(len(turns) - 1, -1, -1):
        if turns[i][0] == "Assistant":
            last_assistant_idx = i
            break

    prefix = turns[:last_assistant_idx] if last_assistant_idx > 0 else turns
    if not prefix:
        return None
    return turns_to_text(prefix) + "\nAssistant:"


def load_extra_documents(jsonl_path):
    """Load pre-extracted, pre-chunked text documents from tools/corpus-extractor's
    JSONL output (one {"text": ..., ...} record per line — see
    docs/BOOKS_CORPUS_INTEGRATION.md). Source-agnostic: books, extracted repo source
    code, anything corpus-extractor produced — same flat JSONL shape either way.
    Returns a plain list of strings, already token-chunked and quality-filtered by
    corpus-extractor itself, so no further cleaning/filtering happens here — this
    function is pure I/O, not a second filter pass duplicating what the Rust tool
    already did.
    """
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line)["text"])
    return documents


def download_source(source, raw_root, token=None):
    """Fetch one dataset's parquet shards into data/raw/<slug>/.

    Not every dataset repo publishes parquet on its main branch — some (e.g.
    databricks/databricks-dolly-15k) only ever shipped a raw .jsonl file. Hugging Face
    auto-converts every public dataset to parquet anyway, published on a standard
    `refs/convert/parquet` ref regardless of the source repo's native format — falling
    back to that ref when the main branch has no parquet means this pipeline can ingest
    any public HF dataset uniformly, without a second, format-specific code path per
    dataset that happens to not natively use parquet.
    """
    from huggingface_hub import snapshot_download

    out_dir = Path(raw_root) / source.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    def attempt(revision):
        print(f"[download] {source.hf_id} (revision={revision or 'main'}) -> {out_dir}")
        snapshot_download(
            repo_id=source.hf_id,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=["*.parquet"],
            token=token if token else True,
            revision=revision,
        )

    attempt(None)
    if not _parquet_files(out_dir):
        print(f"[info] no parquet on main for {source.hf_id}; trying refs/convert/parquet")
        attempt("refs/convert/parquet")
    return out_dir


def _parquet_files(directory):
    return sorted(
        p for p in glob.glob(str(Path(directory) / "**" / "*.parquet"), recursive=True)
        if Path(p).is_file()
    )


def load_conversations(parquet_files, cap, cache_dir, token, min_turns, min_chars, min_ascii_ratio):
    """Parse parquet shards into conversations, stopping once `cap` are collected."""
    from datasets import load_dataset

    collected = []
    for path in parquet_files:
        if len(collected) >= cap:
            break
        try:
            shard = load_dataset(
                "parquet",
                data_files=path,
                split="train",
                cache_dir=str(cache_dir),
                token=token if token else None,
            )
        except Exception as exc:
            print(f"[warn] skipping unreadable parquet file {path}: {exc}")
            continue

        for i in range(min(len(shard), cap - len(collected))):
            turns = extract_turns(shard[i], min_chars, min_ascii_ratio)
            if len(turns) < min_turns:
                continue
            roles = {r for r, _ in turns}
            if "Assistant" not in roles or "User" not in roles:
                continue
            collected.append(turns)
    return collected


OASST_ROLE_MAP = {"prompter": "User", "assistant": "Assistant"}


def load_oasst_conversations(parquet_files, cap, cache_dir, token, min_turns, min_chars,
                              min_ascii_ratio, lang="en"):
    """OASST1-specific: its raw schema is a flat table of individual messages forming
    reply trees (message_id/parent_id/rank), not a pre-assembled conversation — every
    other source's row-by-row extract_turns() dispatch structurally cannot parse this
    (confirmed empirically: it silently parsed to zero conversations before this existed).

    Reconstructs one linear conversation per tree by walking from each root message,
    picking the best-ranked reply at each level (rank 0 = highest quality, per OASST1's
    own human ranking of sibling replies) rather than an arbitrary child — the same
    "trust the human curation already present in the data" principle this project's
    other human-rated/hand-written sources (Dolly) benefit from for free.

    English-only by default (`lang="en"`): the rest of this corpus, and the GPT-2
    tokenizer/vocabulary it's paired with, are English-focused — mixing in OASST1's other
    34 languages would train a meaningful fraction of tokens on content the tokenizer and
    the rest of the corpus give the model little other exposure to.
    """
    from datasets import load_dataset

    collected = []
    for path in parquet_files:
        if len(collected) >= cap:
            break
        try:
            shard = load_dataset(
                "parquet",
                data_files=path,
                split="train",
                cache_dir=str(cache_dir),
                token=token if token else None,
            )
        except Exception as exc:
            print(f"[warn] skipping unreadable parquet file {path}: {exc}")
            continue

        by_id = {}
        children = {}
        for row in shard:
            if lang is not None and row.get("lang") != lang:
                continue
            message_id = row["message_id"]
            by_id[message_id] = row
            children.setdefault(row.get("parent_id"), []).append(message_id)

        # Best-ranked (rank 0) reply first at every branch point; unranked replies sort last.
        for kids in children.values():
            kids.sort(key=lambda mid: (by_id[mid].get("rank") if by_id[mid].get("rank") is not None else 10_000))

        for root_id in children.get(None, []):
            if len(collected) >= cap:
                break
            turns = []
            node_id = root_id
            while node_id is not None:
                row = by_id.get(node_id)
                if row is None:
                    break
                role = OASST_ROLE_MAP.get(row.get("role"))
                text = clean_text(row.get("text") or "")
                if role and text and is_quality_text(text, min_chars, min_ascii_ratio):
                    turns.append((role, text))
                kids = children.get(node_id, [])
                node_id = kids[0] if kids else None

            if len(turns) < min_turns:
                continue
            roles_present = {r for r, _ in turns}
            if "Assistant" not in roles_present or "User" not in roles_present:
                continue
            collected.append(turns)
    return collected


def build_corpus(
    paths,
    include_gated=True,
    token=None,
    max_per_dataset=100_000,
    min_turns=2,
    min_turn_chars=24,
    min_ascii_ratio=0.995,
    num_prompts=50,
    train_ratio=0.9,
    seed=42,
    skip_download=False,
    extra_documents=(),
):
    """Download (unless skipped), parse, filter, and write the train/test/prompt files.

    `extra_documents`: pre-built, pre-filtered plain-text documents from outside the HF
    registry (e.g. tools/corpus-extractor's book/repo-derived JSONL, via load_extra_documents())
    — pooled in and shuffled together with the chat conversations before the train/test
    split, the same "shuffle everything together" principle sources.py's five HF
    datasets already use (see docs/TRAINING_QA.md), just extended to a non-HF source.
    Held-out prompts (test_prompts.txt) are still derived only from chat conversations —
    prompt_from_turns()'s "stop right before the final Assistant reply" concept doesn't
    apply to a plain-prose book excerpt.
    """
    random.seed(seed)
    raw_root = paths.data_dir / "raw"
    cache_dir = paths.data_dir / "hf_cache"
    raw_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    chosen = selected(include_gated=include_gated)
    if not include_gated:
        skipped = ", ".join(d.hf_id for d in DATASETS if d.gated)
        print(f"[info] skipping gated dataset(s): {skipped}")

    all_conversations = []
    per_source_counts = {}

    for source in chosen:
        target_dir = raw_root / source.slug
        if not skip_download:
            try:
                download_source(source, raw_root, token=token)
            except Exception as exc:
                note = " (gated — accept terms and set HF_TOKEN)" if source.gated else ""
                print(f"[warn] could not download {source.hf_id}{note}: {exc}")

        files = _parquet_files(target_dir)
        if not files:
            print(f"[warn] no parquet files for {source.hf_id}, skipping")
            per_source_counts[source.hf_id] = 0
            continue

        loader = load_oasst_conversations if source.schema == "oasst_tree" else load_conversations
        conversations = loader(
            files,
            cap=max_per_dataset,
            cache_dir=cache_dir,
            token=token,
            min_turns=min_turns,
            min_chars=min_turn_chars,
            min_ascii_ratio=min_ascii_ratio,
        )
        per_source_counts[source.hf_id] = len(conversations)
        all_conversations.extend(conversations)
        print(f"[parsed] {source.hf_id}: {len(conversations):,} conversations kept")

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
    chat_train_docs = [turns_to_text(t) for t in train_rows]
    chat_test_docs = [turns_to_text(t) for t in test_rows]

    # extra_documents get the same shuffle-then-split treatment, independently of the
    # chat conversations (different document shapes — plain prose vs. structured turns —
    # so they can't share a single list until after turns_to_text() above), then the two
    # train pools (and the two test pools) are pooled and reshuffled together so a book
    # excerpt and a chat conversation can end up adjacent, not all-chat-then-all-books.
    extra_documents = list(extra_documents)
    random.shuffle(extra_documents)
    extra_split_idx = int(len(extra_documents) * train_ratio)
    book_train_docs = extra_documents[:extra_split_idx]
    book_test_docs = extra_documents[extra_split_idx:]

    train_docs = chat_train_docs + book_train_docs
    test_docs = chat_test_docs + book_test_docs
    random.shuffle(train_docs)
    random.shuffle(test_docs)

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.train_data.write_text(
        DOCUMENT_SEPARATOR.join(train_docs) + "\n", encoding="utf-8"
    )
    paths.test_data.write_text(
        DOCUMENT_SEPARATOR.join(test_docs) + "\n", encoding="utf-8"
    )

    prompts = [p for p in (prompt_from_turns(t) for t in test_rows[:num_prompts]) if p]
    # Trailing separator matters: a file with a single prompt would otherwise contain no
    # separator at all, and the reader's line-splitting fallback would shred that one
    # multi-line prompt into several bogus one-line prompts.
    paths.test_prompts.write_text(
        PROMPT_FILE_SEPARATOR.join(prompts) + PROMPT_FILE_SEPARATOR if prompts else "",
        encoding="utf-8",
    )

    return {
        "per_source": per_source_counts,
        "total_conversations": len(all_conversations),
        "train_conversations": len(train_rows),
        "test_conversations": len(test_rows),
        "extra_documents": len(extra_documents),
        "extra_train_documents": len(book_train_docs),
        "extra_test_documents": len(book_test_docs),
        "prompts": len(prompts),
    }
