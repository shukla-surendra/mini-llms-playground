"""Curate local Hugging Face downloads into a deduplicated train/test text corpus.

Input files are discovered recursively under --input-dir. Supported source formats are
JSONL, JSON arrays, CSV, Parquet, and plain UTF-8 text. Outputs are separate from the
active data/train.txt so review is explicit before replacing a training corpus.
"""
import argparse, csv, hashlib, json, os, sqlite3
from collections import Counter
from pathlib import Path

from ..data.prepare import clean_text, extract_turns, turns_to_text, is_quality_text, DOCUMENT_SEPARATOR

TEXT_KEYS = ("text", "content", "document", "body", "code", "article")
SUPPORTED = {".jsonl", ".json", ".csv", ".parquet", ".txt"}


def rows(path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try: yield json.loads(line)
                    except json.JSONDecodeError: continue
    elif suffix == ".json":
        with path.open(encoding="utf-8") as f: data = json.load(f)
        if isinstance(data, list): yield from data
        elif isinstance(data.get("data"), list): yield from data["data"]
        else: yield data
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f: yield from csv.DictReader(f)
    elif suffix == ".parquet":
        # Do not use datasets.load_dataset() here: it builds a complete Arrow cache
        # for each shard, which can exceed laptop memory on downloaded HF corpora.
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1_024):
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8", errors="replace") as f:
            # Plain-text corpora can be huge too; one line/document at a time keeps
            # curation bounded rather than materialising a whole file in RAM.
            for text in f:
                yield {"text": text}


def document(row, min_chars, ascii_ratio):
    if not isinstance(row, dict): return None
    turns = extract_turns(row, min_chars, ascii_ratio)
    if turns: return turns_to_text(turns)
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str):
            text = clean_text(value)
            if is_quality_text(text, min_chars, ascii_ratio): return text
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--test-fraction", type=float, default=.01)
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--min-ascii-ratio", type=float, default=.90)
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()
    if not args.input_dir.is_dir(): raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not 0 < args.test_fraction < 1: raise ValueError("--test-fraction must be between 0 and 1")
    files = sorted(f for f in args.input_dir.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED)
    if not files: raise FileNotFoundError(f"No supported files under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test = args.output_dir / "train.txt", args.output_dir / "test.txt"
    # Exact deduplication must not keep millions of SHA-256 strings in RAM. SQLite
    # provides a disk-backed set with a unique primary key and is in Python's stdlib.
    seen_path = args.output_dir / ".curation_seen.sqlite"
    seen_path.unlink(missing_ok=True)
    seen_db = sqlite3.connect(seen_path)
    seen_db.execute("PRAGMA journal_mode=OFF")
    seen_db.execute("PRAGMA synchronous=OFF")
    seen_db.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY)")
    stats, written = Counter(), 0
    with train.open("w", encoding="utf-8") as tr, test.open("w", encoding="utf-8") as te:
        for path in files:
            for row in rows(path):
                stats["rows"] += 1
                text = document(row, args.min_chars, args.min_ascii_ratio)
                if not text: stats["rejected"] += 1; continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                try:
                    seen_db.execute("INSERT INTO seen VALUES (?)", (digest,))
                except sqlite3.IntegrityError:
                    stats["duplicates"] += 1; continue
                # Stable hash split prevents train/test leakage and is reproducible.
                target = te if int(digest[:16], 16) / 2**64 < args.test_fraction else tr
                if target.tell(): target.write(DOCUMENT_SEPARATOR)
                target.write(text)
                written += 1; stats["documents"] += 1; stats["characters"] += len(text)
                if written % 10_000 == 0:
                    seen_db.commit()
                    print(f"curated {written:,} documents", flush=True)
                if args.max_docs and written >= args.max_docs: break
            if args.max_docs and written >= args.max_docs: break
    seen_db.commit()
    seen_db.close()
    seen_path.unlink(missing_ok=True)
    manifest = {"input_dir": str(args.input_dir), "files": [str(f) for f in files], "stats": dict(stats),
                "test_fraction": args.test_fraction, "separator": repr(DOCUMENT_SEPARATOR)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["stats"], indent=2))
    print(f"Review {train} and {test}; then copy them into data/ and run make tokenize.")


if __name__ == "__main__": main()
