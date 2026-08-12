# Embeddings: The General Idea

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 0 — Deep Learning
Foundations. This chapter deliberately zooms *out* from tokens specifically — embeddings
are a general deep-learning technique that shows up far beyond language models (user
embeddings in recommendation systems, category embeddings in tabular data, image
embeddings in vision models). [Chapter 10](10_transformer_architecture.md) covers the
*token*-embedding application in full detail; this chapter is the underlying idea that
application is one instance of.

## In Plain English

An embedding is a way of representing something discrete and arbitrary — a word, a user
ID, a product category — as a list of numbers (a vector), positioned in space so that
*similar* things end up *near* each other. The specific numbers don't mean anything in
isolation; what matters is their relative position to other embeddings. "King" and
"Queen" ending up near each other in embedding space isn't a coincidence or hand-coded
rule — it's what the model discovered, during training, was useful for predicting text
well.

## The First-Principles Explanation

### The problem embeddings solve: representing discrete categories usefully

Suppose you need to represent 50,000 different words as input to a neural network
([Chapter 1](01_neurons_layers_and_networks.md)), which only understands numbers. The
naive approach — **one-hot encoding** — represents each word as a 50,000-length vector
that's all zeros except a single `1` at that word's unique position:

```
"cat"  -> [0, 0, 1, 0, 0, ..., 0]   (a 1 at position 2, out of 50,000)
"dog"  -> [0, 0, 0, 0, 1, ..., 0]   (a 1 at position 4, out of 50,000)
```

**Two real problems with this**: it's enormous and mostly wasted space (49,999 zeros for
every word), and — critically — *every pair of words is equally "far apart"* in this
representation. "Cat" and "dog" are no more similar to each other, numerically, than
"cat" and "refrigerator" are. There's no way for the network to exploit any notion of
similarity, because none exists in this encoding.

### The embedding solution: a learned, dense, low-dimensional vector per category

An **embedding** replaces that 50,000-length sparse vector with a much shorter (say,
768-number) **dense** vector — and, crucially, that vector isn't hand-designed, it's
**learned** the same way any other parameter is
([Chapter 3](03_how_neural_networks_learn.md)): gradient descent adjusts each word's
embedding vector, over training, in whatever direction reduces the overall loss. Words
that behave similarly in the training data (appear in similar contexts, get used in
similar ways) end up with embedding vectors that are numerically close to each other —
not because anyone programmed that rule, but because it turns out to be a genuinely
useful representation for whatever task the network is being trained on.

### What "close" and "similar" mean numerically

Embedding vectors live in a high-dimensional space, and "closeness" is measured with
**cosine similarity** — comparing the *angle* between two vectors rather than their raw
values (two vectors pointing in nearly the same direction are "similar," regardless of
their length):

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)

Result close to 1: nearly the same direction (very similar)
Result close to 0: unrelated directions (not similar)
Result close to -1: nearly opposite directions (semantically opposite, in some cases)
```

A famous (if imperfect and often over-quoted) illustration of embedding spaces: taking
the vector for "king," subtracting "man," and adding "woman" lands *near* the vector for
"queen" — vector arithmetic capturing something like relational meaning. It's a real,
observed property of well-trained embedding spaces, worth knowing as an illustration —
not a guaranteed, precise operation you should expect to always work cleanly.

## Grounded in This Repo's Code

`nn.Embedding` in PyTorch is a direct implementation of exactly this idea — a lookup
table with one learned dense vector per discrete category:

```python
self.token_emb = nn.Embedding(vocab_size, embed_size)   # TinyGPT.__init__, line 286
```

This single line creates a `50257 × 768` table — one 768-number dense embedding vector
per vocabulary token ([Chapter 9](09_tokenization.md)), replacing what would otherwise
need to be a `50257`-length one-hot vector per token. Every one of those `50257 × 768 ≈
38.6M` numbers is a genuine parameter, initialized randomly and shaped entirely by
training, per
[`from_scratch/custom-gpt-153m/README.md`'s parameter breakdown](../../from_scratch/custom-gpt-153m/README.md#parameter-count-current-config).

The *same* mechanism, `nn.Embedding`, is used for a conceptually different purpose one
line later in the same file:

```python
self.pos_emb = nn.Embedding(context_length, embed_size)   # line 287
```

This is worth noting specifically: `pos_emb` is *also* an embedding table (position
number → dense vector), using the identical PyTorch layer type as `token_emb` — proof
that "embedding" is a general technique (discrete thing → learned dense vector), not
something inherently tied to words. [Chapter 10](10_transformer_architecture.md#positional-embeddings-telling-the-model-where-each-token-is)
covers what positional embeddings specifically encode.

## Deep-Dive: Embeddings Exist Far Beyond Language Models

Once you see embeddings as "a learned dense representation of a discrete category," it's
worth recognizing the same technique across deep learning generally:

- **Recommendation systems** — a user ID and a product ID each get their own embedding;
  the model learns to place users and products they'd likely match near each other in a
  shared embedding space.
- **Tabular deep learning** — a categorical column (e.g., "city," "product category") in
  a dataset gets embedded rather than one-hot encoded, for the same reasons described
  above.
- **Computer vision** — image embeddings (from models like CLIP) place semantically
  similar images near each other, the visual analog of what token embeddings do for text.

This is the concrete reason a solid grasp of embeddings, learned here at the general
level, transfers directly to any of these other domains — it's not an LLM-specific trick,
it's foundational deep-learning vocabulary.

### A forward-looking note: not every parameter update touches embeddings equally

Worth flagging here as a preview of [Part 3](00_roadmap.md#part-3--fine-tuning-adapting-an-existing-model):
full fine-tuning updates the embedding table just like any other parameter, but
**LoRA** (used in this repo's
[`fine_tuning/tinyllama-1.1b-lora/`](../../fine_tuning/tinyllama-1.1b-lora/)) deliberately
freezes the base model's embedding table and most other weights, only training small
additional adapter matrices layered on top. This means a LoRA-fine-tuned model's
*understanding* of individual tokens (their embeddings) stays exactly as the base model
learned it — only *how those understandings get combined and used* changes. The full
mechanism is [Chapter 17](00_roadmap.md#part-3--fine-tuning-adapting-an-existing-model).

## Try It Yourself

- After training `tiny_llm.py` for at least a few hundred steps (enough that embeddings
  have moved meaningfully from their random initialization), extract two token embeddings
  and compute their cosine similarity:
  ```python
  import torch
  ckpt = torch.load("tiny_llm_checkpoint.pt", map_location="cpu")
  emb = ckpt["model_state_dict"]["token_emb.weight"]
  # tiktoken.get_encoding("gpt2").encode(" cat"), (" dog") to find token IDs
  a, b = emb[some_id_1], emb[some_id_2]
  similarity = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
  print(similarity.item())
  ```
  Compare a plausible-similar pair against a plausible-unrelated pair — with enough
  training, you should see a real, measurable difference.
- Compare that similarity score against the *same* two tokens' embeddings taken from a
  freshly-initialized (untrained) model — the untrained embeddings should show no
  meaningful pattern at all, since they're still just random numbers at that point.

## Common Misconceptions

- **"Embeddings only apply to words/tokens."** As the deep-dive shows, it's a general
  technique for any discrete category — this repo's own `pos_emb` (positions, not words)
  is direct proof within the same file.
- **"An embedding vector's individual numbers mean something specific (e.g., dimension 5
  = 'animalness')."** Almost never — individual dimensions of a learned embedding
  typically don't correspond to any single human-interpretable concept; it's the
  *relative position* (distance/direction to other embeddings) that carries meaning, not
  any one coordinate read in isolation.
- **"The king - man + woman = queen example always works precisely."** It's a real,
  documented property that appears in well-trained embedding spaces to a meaningful
  degree, but treating it as a precise, guaranteed algebraic operation overstates how
  cleanly it actually holds in practice.

## Practice Questions

1. Why does one-hot encoding fail to capture any notion of similarity between two
   categories, while a learned embedding can?
2. `pos_emb` and `token_emb` use the identical `nn.Embedding` layer type. What's actually
   different between them — is it the mechanism, or what's being looked up?
3. Why does LoRA fine-tuning ([Part 3](00_roadmap.md#part-3--fine-tuning-adapting-an-existing-model))
   typically freeze the embedding table rather than updating it, and what does that imply
   about what LoRA fine-tuning does and doesn't change about a model's behavior?

## Key Terms

- **One-hot encoding**: representing a discrete category as a vector of all zeros except
  a single `1`, with no notion of similarity between categories.
- **Dense embedding**: a short, learned vector representation where distance/direction
  encodes meaningful similarity.
- **Embedding space**: the high-dimensional space all of a given embedding table's
  vectors live in.
- **Cosine similarity**: a measure of how similar two vectors' directions are, independent
  of their magnitude — the standard way to compare embeddings.
- **`nn.Embedding`**: PyTorch's lookup-table layer implementing this mechanism directly.
