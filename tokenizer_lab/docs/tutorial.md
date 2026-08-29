# LLM Vocabulary & Tokenizers

## Tokenizer

A tokenizer converts text → tokens → token IDs. The model never sees raw text — every
input and output is a sequence of integers indexing into a fixed vocabulary.

Checkable, not hypothetical — this repo has three real tokenizers sitting on disk:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("../from_scratch/custom-gpt-350m/tokenizer/tokenizer.json")
tok.encode("I love AI").ids     # -> a real list of integers, not a made-up example
```

`tokenizer_lab/notebooks/01_comparing_this_repos_tokenizers.ipynb` runs exactly this
against all three tokenizers side by side — worth having open while reading the rest of
this doc.

## Vocabulary

A tokenizer has a **vocabulary**: a fixed collection of tokens mapped to integer IDs.
Each tokenizer has its own vocabulary and its own token-to-ID mapping — there is no
single universal vocabulary shared by all LLMs. Three real, different ones, all in this
repo:

| Tokenizer | Vocab size | Where it comes from |
| --- | --- | --- |
| `custom-gpt-6m` | 4,096 | Custom-trained BPE on 100k TinyStories |
| `custom-gpt-350m` | 32,768 | Custom-trained BPE (`oxide-bpe-32k`) on this project's own corpus |
| `custom-gpt-50m` | 50,257 | The real, unmodified public GPT-2 vocabulary (via `tiktoken`) |

## Vocabulary size

Vocabulary size is fixed once a tokenizer is built — you can't change it without
retraining the tokenizer itself, which means re-tokenizing every dataset built against
it. Common choices in practice: 16K, 32K, 64K, 128K (GPT-2 used 50,257; many modern
LLMs use 100K+). Changing vocabulary size means training a *different* tokenizer, full
stop — there's no way to resize one in place.

**The real trade-off, not just "pick a round number":** a bigger vocabulary compresses
text into fewer tokens (shorter sequences to process), but costs more parameters in the
embedding table below — and that cost is *relatively* larger on a smaller model. This
repo's own `custom-gpt-350m/src/gpt/tokenizer.py` computed this explicitly rather than
guessing:

```
E = 896 (embedding size)
V = 50,257 (GPT-2's vocab)  ->  45.0M embedding params  = 22.3% of a 202M model
V = 32,768 (this project's) ->  29.4M embedding params  = 14.6% of a 202M model
```

The 15.7M-parameter difference goes into actual transformer blocks instead of a lookup
table — real reasoning capacity, not vocabulary. This is *why* 32K, specifically, isn't
an arbitrary "reasonable learning experiment" number — see "What we recommended," below.

## Why does the model need `vocab_size` if the tokenizer already exists?

Because the model needs to build an **embedding matrix**:

```
vocab_size × hidden_size
```

Example: `vocab_size = 32,000`, `hidden_size = 768` → 32,000 embedding vectors, each
768-dimensional — one learned vector per possible token ID, looked up by index. The
tokenizer decides how many rows that table needs; the model doesn't get a say.

## Tokenizer and model must match

A model trained with a particular tokenizer needs that *exact* tokenizer at inference —
not just one with the same vocab size. Token ID 1234 in one 32K vocabulary can mean a
completely different subword than token ID 1234 in another 32K vocabulary; nothing about
"32,000 rows" tells you what each row *means*.

This isn't a footnote risk — it's silent, not loud. Swapping tokenizers doesn't crash
anything: every ID is still a valid row index into the embedding table, so the model
runs, generates fluent-looking tokens, and produces text that means nothing close to what
was intended. This repo's own `gpt.data.dataset.tokenizer_fingerprint()` (in
`custom-gpt-350m`/`custom-gpt-153m`) exists specifically to catch this class of mistake:
it hashes a probe string's encoded output and stores that hash alongside every `.bin`
file, so loading a corpus tokenized by a *different* tokenizer than the one currently
configured fails loudly at load time instead of corrupting training silently.

## BPE

**BPE = Byte Pair Encoding.** It's a *tokenization algorithm* — a recipe for building a
vocabulary from a training corpus — not one specific tokenizer. All three tokenizers in
this repo are BPE variants, and yet produce completely different vocabularies, because
BPE's output depends entirely on what text it was trained on.

**The mechanism**, not just the conceptual example: start with every training document
split into individual characters (or bytes). Count every adjacent pair of symbols across
the whole corpus. Merge the single most frequent pair into one new symbol. Repeat —
each repetition is one entry in the tokenizer's learned merge list, applied in the exact
order they were learned. `playing → play + ing` and `unbelievable → un + believe + able`
are the *result* of this process on a large corpus, not something BPE is told directly.
`tokenizer_lab/notebooks/01_comparing_this_repos_tokenizers.ipynb`'s Exercise 4 has you
implement exactly this merge step yourself on a tiny toy corpus, from scratch.

**One thing plain BPE doesn't decide on its own**: whether digits merge freely.
`custom-gpt-350m`'s tokenizer deliberately pre-splits every digit *before* BPE ever runs
(`Digits(individual_digits=True)`), specifically so a number's tokenization never depends
on how often that exact number showed up in training — plain GPT-2's BPE will happily
learn `"20"`+`"24"` as one segmentation and `"99"`+`"999"` as a different one for the same
reason (frequency), which is directly measurable: run Exercise 2 in the same notebook and
watch it happen on real GPT-2 output.

## Common tokenizer approaches

- **BPE / Byte-level BPE** — greedy: always merges the single most frequent adjacent
  pair. Common in GPT-style models; all three tokenizers in this repo use this family.
- **WordPiece** — famously used by BERT. Similar iterative-merge structure to BPE, but
  picks the merge that most improves the training corpus's *likelihood* under the
  resulting vocabulary, not simply the raw-frequency-highest pair.
- **Unigram** — commonly paired with SentencePiece. Works in the *opposite* direction
  from BPE/WordPiece: starts from a large candidate vocabulary and iteratively prunes
  the subwords that contribute least to corpus likelihood, rather than building up from
  individual characters.

Worth being honest about scope: nothing in this repo currently uses WordPiece or
Unigram — all three real tokenizers here are BPE family. If you want hands-on material
for those, that's a gap, not something `tokenizer_lab/` currently covers.

## What we recommended for your experiment

Use Hugging Face's `tokenizers` library, start with byte-level BPE, and a vocabulary
around 32K is a reasonable learning experiment. This wasn't arbitrary — it's almost
exactly what `custom-gpt-350m` actually shipped (32,768), for the concrete
embedding-table-share-of-params reason computed above, not a round-number guess. You can
train different vocabulary sizes later and compare them directly — `custom-gpt-6m`'s
4,096 and `custom-gpt-50m`'s 50,257 are two more real reference points already sitting in
this repo, at the opposite ends of the range.

## Model size vs. tokenizer size

Vocabulary size doesn't have to scale with model size — you could use the same tokenizer
for a 50M, 150M, or 1B model, and doing so is actually useful: it isolates the effect of
model size from confounding changes in tokenization when comparing runs.

But "doesn't have to" isn't "cost-free at every size" — the embedding-table calculation
above is the reason why. The same 50,257-row table costs 22.3% of a 202M model's
parameters but a much smaller share of a 7B model's. This is the same crossover logic
`custom-gpt-350m/docs/MODEL_SIZING_GUIDE.md` covers for embedding size vs. layer count in
general: at small model scale, vocabulary choice is a real design decision with a
measurable cost, not a free knob.

## The key mental model

```
YOUR TEXT
   ↓
TOKENIZER
   ↓
TOKEN IDs
   ↓
EMBEDDING TABLE
   ↓
LLM
   ↓
TOKEN IDs
   ↓
TOKENIZER
   ↓
TEXT
```

And the important relationship underneath it:

```
Tokenizer vocabulary
        ↓
   vocab_size
        ↓
Model embedding matrix
```

Tokenizer → vocabulary → `vocab_size` → model embeddings are tightly coupled — change
any one link and the others have to change with it.

**One caveat this diagram hides**: the round trip at the bottom (`TOKEN IDs → TOKENIZER →
TEXT`) is drawn as if it's always lossless. It isn't automatically — `custom-gpt-6m`'s
tokenizer genuinely fails to round-trip text containing characters it never saw during
training (mostly `<unk>`, and the decoded text comes back corrupted), while
`custom-gpt-350m` and `custom-gpt-50m` round-trip *any* Unicode text perfectly via true
byte-level fallback. Same "byte-level" label, different guarantee — Exercise 3 in the
notebook proves this with real Hindi text and explains exactly why the difference exists
(whether the trainer was given the full byte alphabet up front, not just whether it uses
a byte-level pre-tokenizer).
