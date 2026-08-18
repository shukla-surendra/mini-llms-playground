"""
A character-level tokenizer — the simplest possible way to turn text into numbers.

WHY character-level, and not the GPT-2 "BPE" tokenizer the other five projects in
`from_scratch/` use (50,257 vocabulary entries), or even the 6M sibling's own custom
4,096-entry vocabulary: at small model sizes, the token-embedding table
(`vocab_size x n_embd`, one learned vector per possible token) is often the single
*biggest* piece of the whole model — the sibling `custom-gpt-50m` project's own docs
note it's 80%+ of parameters at that project's smallest preset. A 50,257-word vocabulary
would completely dominate a model this size, leaving almost no parameters left for the
actual attention/reasoning mechanism this project exists to teach. A character-level
vocabulary is only as big as the number of distinct *characters* in the training text —
a few dozen, not tens of thousands — so nearly every one of this model's ~0.8M
parameters goes toward the Transformer mechanism itself, not a lookup table.

The tradeoff, honestly: the model has to spend some of its limited capacity learning
that "t", "h", "e", " " tend to appear in that order (i.e. learning to *spell* "the")
before it can learn anything about which words follow which — a word-level or subword
tokenizer gets "the" for free, as a single unit. That tradeoff is exactly why every
larger model in this workspace uses a subword tokenizer instead. Deep dive on
tokenization strategies in general: docs/llm-engineering/09_tokenization.md.
"""

from __future__ import annotations

from pathlib import Path


class CharTokenizer:
    """
    WHAT: a lossless, two-way mapping between text and a list of integers — one integer
    per character. No merging, no vocabulary file to download, no out-of-vocabulary
    fallback: every character that exists in the training corpus gets exactly one id,
    and (by construction) every character the model will ever be asked to encode during
    training already appears in that corpus.

    HOW: `set(text)` collects every *distinct* character that appears anywhere in the
    corpus (repeats collapse to one entry); `sorted(...)` puts them in a fixed,
    reproducible order so the same corpus always produces the same id for the same
    character, run after run. Then it's just two dictionaries — string-to-int and
    int-to-string — built once and reused for every encode/decode call.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Text -> list of ints. Raises KeyError on a character the corpus never
        contained (e.g. an emoji, if the corpus is plain English) — deliberately not
        caught here: it's a signal that you're prompting the model with something it
        has literally no way to represent, not a bug to paper over."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        """List of ints -> text. The exact inverse of `encode` — `decode(encode(s)) ==
        s` for any `s` built only from characters in the training corpus."""
        return "".join(self.itos[int(i)] for i in ids)

    @classmethod
    def from_file(cls, path: str | Path) -> "CharTokenizer":
        """Build the tokenizer's alphabet directly from a training-corpus file — the
        only way this project constructs one, so the vocabulary always exactly matches
        what the model will actually be trained on."""
        return cls(Path(path).read_text(encoding="utf-8"))
