#!/usr/bin/env python3
"""One-off: build data/train.txt + data/test.txt (the PRETRAINING-phase corpus, replacing
the old mixed chat+extra-documents corpus this project trained on before) from the 5
sources copied over from the sibling custom-gpt-153m project's enriched raw data pool
(see each folder's SOURCE.md, copied alongside).

Same method the 153m project itself used to build its own data/train.txt/test.txt
(scripts/build_pretrain_split.py there) — held out per-source (not one global shuffle,
to keep test representative of every source and avoid loading everything into memory at
once): the LAST ~1% of each source's documents become test, the rest become train.
Documents are separated by "\\n\\n" (DOCUMENT_SEPARATOR) in the source files already;
this script only re-partitions at those same boundaries.

    uv run python scripts/build_pretrain_split.py
"""

SOURCES = [
    "data/raw/HuggingFaceTB__smollm-corpus__cosmopedia-v2/text.txt",
    "data/raw/HuggingFaceTB__cosmopedia/text.txt",
    "data/raw/HuggingFaceTB__finemath__finemath-4plus/text.txt",
    "data/raw/wikimedia__wikipedia__hi/text.txt",
    # open-web-math__open-web-math/text.txt excluded: copied over empty (0 bytes) —
    # 153m's fetch for it hadn't started yet when this corpus was copied "as is" for
    # 50m's restart. Add it back here once 153m's own enrichment fills it in.
]
TEST_FRACTION = 0.01
SEP = "\n\n"


def split_source(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    docs = text.split(SEP)
    n_test = max(1, int(len(docs) * TEST_FRACTION))
    train_docs, test_docs = docs[:-n_test], docs[-n_test:]
    print(f"  {path}: {len(docs):,} docs -> train {len(train_docs):,} / test {len(test_docs):,}")
    return train_docs, test_docs


def main():
    all_train, all_test = [], []
    for src in SOURCES:
        train_docs, test_docs = split_source(src)
        all_train.extend(train_docs)
        all_test.extend(test_docs)

    with open("data/train.txt", "w", encoding="utf-8") as f:
        f.write(SEP.join(all_train))
    with open("data/test.txt", "w", encoding="utf-8") as f:
        f.write(SEP.join(all_test))

    print(f"\ndata/train.txt: {len(all_train):,} docs")
    print(f"data/test.txt:  {len(all_test):,} docs")


if __name__ == "__main__":
    main()
