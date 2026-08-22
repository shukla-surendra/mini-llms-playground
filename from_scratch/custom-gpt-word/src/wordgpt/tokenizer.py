"""A tiny word tokenizer.

Unlike ``custom-gpt-nano``'s character tokenizer, this tokenizer treats ``cat`` as one
token.  It uses a small regular expression rather than a production BPE algorithm:
words (including contractions) and punctuation are separate tokens.  That makes the
vocabulary and every input id easy to inspect, while preserving the key LLM pipeline:
text -> ids -> embeddings -> logits -> text.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]", re.IGNORECASE)
SPECIAL_TOKENS = ("<unk>",)


class WordTokenizer:
    """A corpus-built mapping from normalized word/punctuation tokens to integer ids.

    ``<unk>`` means "unknown". It lets generation accept a new prompt word, but the
    model has no learned meaning for it beyond one shared fallback vector.  This is a
    useful limitation to notice; real tokenizers avoid it with subword pieces.
    """

    def __init__(self, text: str, max_vocab_size: int | None = None) -> None:
        counts = Counter(self.tokenize(text))
        if max_vocab_size is None:
            words = sorted(counts)
        else:
            # Without a cap, every distinct string in the corpus becomes a permanent
            # token_emb row - fine for a hand-picked "the cat sat" corpus, but on real
            # book/PDF text most of the "vocabulary" ends up being one-off extraction
            # noise (broken hyphenation, glued words, OCR/font glyph errors). Keeping
            # only the most frequent words both bounds the embedding table and drops
            # that noise, since it's almost always the rarest "words" in the corpus.
            keep = max(0, max_vocab_size - len(SPECIAL_TOKENS))
            most_common = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:keep]
            words = sorted(word for word, _ in most_common)
        self.tokens = list(SPECIAL_TOKENS) + words
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for token, i in self.stoi.items()}
        self.vocab_size = len(self.tokens)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase and split text deterministically; spaces are separators, not ids."""
        return TOKEN_PATTERN.findall(text.lower())

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        """Join tokens back into readable text, attaching punctuation to its word."""
        output = ""
        for token_id in ids:
            token = self.itos[int(token_id)]
            if token in ".,!?;:":
                output = output.rstrip() + token + " "
            else:
                output += token + " "
        return output.strip()

    @classmethod
    def from_file(cls, path: str | Path, max_vocab_size: int | None = None) -> "WordTokenizer":
        return cls(Path(path).read_text(encoding="utf-8"), max_vocab_size=max_vocab_size)
