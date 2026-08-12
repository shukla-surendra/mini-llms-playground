# Tokenization: Turning Text Into Numbers

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 1 — Foundations. Builds on
[Chapter 8](08_what_is_a_language_model.md)'s claim that a model predicts "the next
token" — this chapter is about what a token actually is, and why the choice of
tokenization scheme has real, lasting consequences on everything downstream.

## In Plain English

A neural network is a big pile of numeric operations — matrix multiplications, additions,
nonlinear functions. It cannot directly consume the letter "c" or the word "cat"; it needs
numbers. Tokenization is the translation step: chop text into pieces, and assign each
distinct piece a unique integer ID. Those integer IDs are what the model actually sees and
predicts — not letters, not words, but a fixed vocabulary of these pieces.

## The First-Principles Explanation

### Why not just use whole words, or individual characters?

Both extremes have real, specific problems:

- **Word-level tokenization** (one token per whole word) — the vocabulary would need an
  entry for every word that might ever appear, including every typo, every rare proper
  noun, every word in every language the model might encounter. Any word not in the fixed
  vocabulary becomes an **out-of-vocabulary (OOV)** problem — the model has no way to
  represent it at all.
- **Character-level tokenization** (one token per letter) — solves the OOV problem
  completely (any text can be spelled out character by character), but makes sequences
  *much* longer for the same text, and asks the model to learn much more distant
  relationships (understanding a word means attending across many more token positions
  than if the whole word were one token).

### The middle ground: subword tokenization (BPE)

**Byte-Pair Encoding (BPE)**, the technique GPT-2 (and this repo's model) uses, finds a
middle ground algorithmically: start with individual characters, then repeatedly merge
the *most frequently occurring adjacent pair* into a new single token, building up a
vocabulary of common characters, common short pieces, and common whole words, all at
once — the merge process is run once, ahead of time, over a large text corpus, and the
resulting fixed vocabulary (e.g., GPT-2's ~50,257 tokens) is what gets used from then on.

```
Conceptually, BPE training on a toy corpus:
  Start: every character is its own token: "l", "o", "w", "e", "r", ...
  Round 1: "l" + "o" occurs very often together -> merge into new token "lo"
  Round 2: "lo" + "w" occurs often -> merge into "low"
  ... continue for many rounds ...
  Result: common whole words ("the", "and") often become single tokens,
          while rare/unusual words get split into a few subword pieces
          ("unbelievable" -> "un" + "believ" + "able", roughly)
```

**Why this specifically solves both extremes' problems**: common words end up as single
tokens (short sequences, like word-level), while rare or unseen words/typos can always be
represented by falling back to smaller, more common subword pieces or even individual
bytes (no OOV problem, like character-level) — genuinely getting most of the benefit of
both approaches.

### The vocabulary is fixed once trained — and this matters a lot

A tokenizer's vocabulary (the full list of token-to-ID mappings) is determined once,
during its own training process, and then frozen. **A model and its tokenizer are a
matched pair** — a model's embedding layer (see [Chapter 10](10_transformer_architecture.md))
has exactly one row per vocabulary entry, so using a different tokenizer than the one a
model was trained with produces meaningless results (the same integer ID would map to a
completely different piece of text).

## Grounded in This Repo's Code

[`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py) uses `tiktoken`
(OpenAI's fast BPE implementation) with GPT-2's own pretrained tokenizer, unchanged:

```python
enc = tiktoken.get_encoding("gpt2")
train_tokens, train_target_mask = encode_with_assistant_mask(enc, train_text)
vocab_size = enc.n_vocab   # 50,257 — this becomes the model's actual output-layer size
```

This single line, `enc = tiktoken.get_encoding("gpt2")`, is a real, consequential design
decision worth naming explicitly: **this project doesn't train its own tokenizer** — it
reuses GPT-2's, trained by OpenAI on a large English-heavy web corpus years ago. This is
a completely standard, reasonable choice for a small project (training a good tokenizer
needs its own large corpus and process), but it does mean this project's tokenization is
optimized for the kind of text GPT-2's tokenizer was built on, not specifically for the
LMSYS/chat conversational data this project actually trains on.

`vocab_size` (50,257) flows directly into the model architecture — it's the exact number
of rows in `TinyGPT`'s token embedding table and the exact output width of `lm_head`
(both covered in [Chapter 10](10_transformer_architecture.md)) — tokenization isn't a
preprocessing detail that's separate from the architecture, it directly determines the
shape of two of the model's layers.

The `encode_with_assistant_mask` function (line 116) is worth a specific look — it does
tokenization *and* something extra: as it encodes each line, it tracks which speaker role
(`System:`, `User:`, `Assistant:`) is currently active, and builds a parallel `mask`
array marking which tokens belong to assistant turns. This mask is what
`masked_next_token_loss` (line 136) later uses to compute loss *only* on the assistant's
tokens — a training-objective decision that lives right next to tokenization in this
code but is conceptually a separate concern (covered fully in
[Chapter 12](00_roadmap.md#part-2--pretraining-building-a-model-from-zero)).

## Deep-Dive: The Downstream Consequences of a Tokenizer Choice

Tokenization decisions made once, early, ripple through the entire system:

- **Sequence length, and therefore compute cost**: a tokenizer that splits text into more
  pieces per word means longer sequences for the same text, which directly costs more
  compute (attention's cost grows with sequence length) — this is *why* tokenizer
  efficiency on a target language/domain is a real, measurable engineering concern, not
  an academic one.
- **Non-English and code text**: GPT-2's tokenizer, trained mostly on English web text,
  is measurably less efficient (more tokens per character) on other languages and on
  code — a real, well-documented limitation that later tokenizers (GPT-4's `cl100k_base`,
  and others) specifically improved on.
- **Numbers**: BPE's frequency-based merging can split numbers inconsistently (e.g.,
  "1234" might tokenize differently than "123" + "4" would suggest) — a known source of
  LLMs' historically poor arithmetic, since the model never sees numbers in a
  consistent, place-value-aligned way the way a human would.

## Try It Yourself

- Install `tiktoken` (already in [`from_scratch/custom-gpt-153m/requirements.txt`](../../from_scratch/custom-gpt-153m/requirements.txt))
  and in a Python shell, run:
  ```python
  import tiktoken
  enc = tiktoken.get_encoding("gpt2")
  print(enc.encode("Hello, world!"))
  print(enc.encode("supercalifragilisticexpialidocious"))
  ```
  Compare the token count for a common phrase vs. an unusual word — you'll see the BPE
  behavior described above directly: common patterns compress to fewer tokens, unusual
  ones split into more subword pieces.
- Look at `encode_with_assistant_mask` in `tiny_llm.py` and trace through what happens to
  a line like `"Assistant: Sure, I can help.\n"` — which role gets detected, and which
  resulting tokens get `mask=1`.

## Common Misconceptions

- **"A token is basically a word."** Often close but frequently wrong — many common words
  are single tokens, but many others split into multiple subword pieces, and some tokens
  are just punctuation or whitespace variants. "Tokens ≈ words" is a rough intuition, not
  a rule.
- **"Any tokenizer works with any model."** No — a model's embedding table is sized
  exactly to its tokenizer's vocabulary; mismatching them produces garbage, not a
  slightly-worse result.
- **"Tokenization is a minor preprocessing detail."** As the deep-dive shows, it directly
  affects sequence length, compute cost, and even a model's arithmetic ability — it's an
  architectural decision with real downstream consequences, not a solved, ignorable step.

## Practice Questions

1. Why does BPE tend to produce shorter tokenized sequences for common words than for
   rare ones, and why does that matter for compute cost?
2. What would go wrong, concretely, if you loaded `tiny_llm_checkpoint.pt`'s weights but
   tokenized a new prompt using a *different* tokenizer than GPT-2's?
3. Why is GPT-2's tokenizer a reasonable but not perfectly optimal choice for this
   repo's LMSYS-derived conversational training data specifically?

## Key Terms

- **Token**: the smallest unit of text a model actually processes — a subword piece, a
  whole word, or punctuation, depending on the tokenizer.
- **Byte-Pair Encoding (BPE)**: a subword tokenization algorithm that iteratively merges
  the most frequent adjacent character/token pairs to build a fixed vocabulary.
- **Vocabulary / `vocab_size`**: the fixed set of all possible tokens a tokenizer can
  produce, and the exact number a model's embedding and output layers are sized to.
- **Out-of-vocabulary (OOV)**: a piece of text with no direct representation in a fixed
  vocabulary — the problem word-level tokenization suffers from and subword methods
  largely avoid.
- **`tiktoken`**: OpenAI's fast BPE tokenizer implementation, used in this repo via
  `tiktoken.get_encoding("gpt2")`.
