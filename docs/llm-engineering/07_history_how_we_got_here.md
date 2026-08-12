# History: How We Got Here

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 1 — Foundations.

## In Plain English

Before you can appreciate why today's language models are built the way they are, it
helps to know what came before, and *why each generation was replaced* — not because the
old approach was "bad," but because it hit a specific, nameable wall that the next idea
solved. This isn't trivia — the wall each generation hit is the exact reason the current
architecture (the Transformer, which the model in this repo is a tiny example of) is
shaped the way it is.

## The First-Principles Explanation

### Generation 1: Counting words (n-gram models, 1990s and earlier)

The earliest practical language models were startlingly simple: count how often each
word follows each other word (or pair, or triple) in a large text corpus, and use those
counts as probabilities. "The cat sat on the ___" — count every time "the" followed "on
the," and predict whichever word came next most often.

**The wall**: this only remembers a fixed, tiny window (typically 2-3 words). It has no
way to connect "The trophy doesn't fit in the suitcase because **it** is too big" back to
"trophy" — the word "it" needs context from much earlier in the sentence than an n-gram
model can see. Language obviously depends on long-range context; n-grams structurally
cannot capture it.

### Generation 2: Networks with memory (RNNs and LSTMs, 2000s-2014)

Recurrent Neural Networks (RNNs) process a sequence one token at a time, carrying a
"hidden state" forward — a running summary of everything seen so far — updated at each
step. LSTMs (Long Short-Term Memory networks) added gating mechanisms specifically to
help that hidden state retain important information over longer stretches without it
decaying.

**The wall, two of them**:
1. **Sequential processing** — each step depends on the previous step's output, so an
   RNN/LSTM cannot process a sequence in parallel; training and inference are both
   inherently one-token-at-a-time, which becomes a severe speed bottleneck as sequences
   and datasets grow.
2. **Long-range information still degrades** — even with LSTM's gating, information from
   very early in a long sequence has to survive being repeatedly compressed into one
   fixed-size hidden state at every step; in practice, distant context still gets diluted.

### Generation 3: Attention as an add-on (2014-2017)

Researchers working on machine translation (translating one language to another) noticed
RNNs struggled specifically because the entire source sentence had to be compressed into
one fixed-size vector before translation could even start. The fix — **attention** — let
the decoder look back at *all* the encoder's hidden states directly, weighted by
relevance, rather than relying on one bottlenecked summary vector. This was a genuine
breakthrough, but it was still bolted onto an RNN — the sequential-processing bottleneck
remained.

### Generation 4: Attention *is* the architecture (Transformer, 2017)

The paper that changed everything, ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
(Vaswani et al., 2017), asked a genuinely radical question: what if you removed the RNN
entirely, and built a model using *only* attention, processing an entire sequence in
parallel rather than one token at a time?

This solved **both** of the previous generation's walls at once:
- **Parallelizable** — no step depends sequentially on the previous step during training
  (this is *why* Transformers can be trained efficiently on modern GPU hardware at all —
  the parallelism this architecture enables is a huge part of why it won).
- **Direct long-range connections** — self-attention lets every position look directly at
  every other position in one step, with no information-decay problem from repeated
  compression through a hidden state.

**Every model in this repo — `tiny_llm.py`'s `TinyGPT` and the pretrained `TinyLlama`
model fine-tuned in `fine_tuning/`** — is a Transformer in this exact lineage. The full
mechanism is [Chapter 10](10_transformer_architecture.md); this chapter is about the path
that led here.

### Generation 5: Scaling up, and the GPT lineage (2018-2022)

The Transformer paper originally described an encoder-decoder architecture for
translation. OpenAI's GPT (Generative Pre-trained Transformer) line took a specific,
consequential simplification: **decoder-only**, trained purely on next-token prediction
over huge amounts of unlabeled text — no translation pairs, no labels, just "predict the
next word" at massive scale. This is the exact same objective
[`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py) uses, just at a vastly
smaller scale (153M parameters here vs. GPT-3's 175B).

The key discovery across GPT-1 (2018) → GPT-2 (2019) → GPT-3 (2020) wasn't a new
architecture each time — it was that the *same* architecture, given more parameters, more
data, and more compute, kept getting better in a surprisingly predictable way. This
observation became formalized as **scaling laws** (covered in depth in
[Chapter 14](14_scaling_laws_and_model_sizing.md)) — a genuinely important discovery in its
own right, not just an engineering footnote.

### Generation 6: From "predicts text" to "follows instructions" (2022-present)

A raw, next-token-trained GPT-3-class model is good at *continuing* text, but not
naturally good at *following instructions* or having a back-and-forth conversation — if
you type "Explain photosynthesis," a pure pretraining-objective model might just continue
your sentence ("...to a five-year-old, in under 100 words, using...") rather than
answering it. The gap between "can predict text well" and "is actually useful as an
assistant" was closed by a second training stage on top of pretraining — instruction
tuning, RLHF, and later DPO (all covered in depth in [Part 3](00_roadmap.md#part-3--fine-tuning-adapting-an-existing-model)).
This is exactly the shape of what
[`fine_tuning/tinyllama-1.1b-lora/`](../../fine_tuning/tinyllama-1.1b-lora/) does at small
scale — take an already-pretrained model and adapt it toward conversational,
instruction-following behavior.

## Grounded in This Repo's Code

Both models in this repo sit at different points in this same lineage:

- [`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py) is a **Generation 5**
  model — a small, decoder-only, next-token-prediction Transformer, built and trained
  from scratch, the same architectural family as GPT-2.
- `TinyLlama-1.1B-Chat`, fine-tuned in
  [`fine_tuning/tinyllama-1.1b-lora/`](../../fine_tuning/tinyllama-1.1b-lora/), is a
  **Generation 6** artifact — a pretrained decoder-only Transformer that's already been
  through instruction tuning (the "-Chat" in its name), which this repo's LoRA script
  then adapts further onto a specific dataset.

## Deep-Dive: Why This History Matters for Understanding Design Decisions Today

Every architectural choice in a modern LLM can be traced back to which historical wall it
was solving:

- **Why self-attention, not recurrence** → solves the sequential-processing bottleneck
  from Generation 2.
- **Why "decoder-only" and not the original encoder-decoder** → GPT's specific bet that a
  single, simpler decoder-only design, trained purely on next-token prediction at scale,
  would generalize better than a more complex, task-specific architecture — a bet that
  proved right.
- **Why pretraining, then a separate fine-tuning stage** → next-token prediction on raw
  text (Generation 5) produces genuine language competence but not instruction-following
  behavior — that gap is what Generation 6's fine-tuning stage exists to close.

## Try It Yourself

- Read [`tiny_llm.py`'s `generate()` function](../../from_scratch/custom-gpt-153m/tiny_llm.py)
  (around line 338) and confirm for yourself: it does nothing but repeatedly predict one
  next token and append it — the exact Generation 5 next-token-prediction objective, with
  no instruction-following training on top. Compare that to a prompt/response exchange
  with `fine_tuning/tinyllama-1.1b-lora`'s served model, which *has* had that additional
  training — the behavioral difference you'll observe is the entire content of
  Generation 6.

## Common Misconceptions

- **"Transformers replaced RNNs because they're 'smarter.'"** Not quite — the actual,
  specific win was parallelizability and direct long-range connections, both concrete
  engineering properties, not a vague notion of intelligence.
- **"GPT-3 was a new architecture."** It wasn't — GPT-1, GPT-2, and GPT-3 share
  essentially the same decoder-only Transformer architecture; the difference between them
  is almost entirely scale (parameters, data, compute), which is exactly what made
  scaling laws a discoverable, surprising pattern in the first place.
- **"A base (pretrained-only) model and a chat model are the same thing with different
  names."** They're genuinely different artifacts — a base model has only ever seen the
  next-token-prediction objective; a chat model has been through an additional training
  stage specifically to make it follow instructions and converse. `tiny_llm.py` produces
  the former; the `-Chat` suffix on `TinyLlama-1.1B-Chat-v1.0` signals the latter.

## Practice Questions

1. Name the specific engineering wall each of the first three generations (n-grams, RNNs/
   LSTMs, attention-as-add-on) hit, in a way that explains *why* the next generation was
   necessary, not just that it existed.
2. Why is parallelizability during training specifically tied to the rise of GPU-scale
   compute for language models — what would an RNN-based approach not be able to take
   advantage of, even with unlimited GPUs?
3. What's the actual difference between a "base model" and a "chat model," and where does
   that difference get introduced in the training pipeline?

## Key Terms

- **N-gram model**: predicts the next word from counts of short, fixed-length word
  sequences seen in training text.
- **RNN / LSTM**: neural networks that process sequences one step at a time, carrying a
  hidden state forward; LSTMs add gating to better preserve long-range information.
- **Attention**: a mechanism letting a model directly weigh and combine information from
  any other position in a sequence, not just a compressed summary.
- **Transformer**: the architecture built entirely around self-attention (no recurrence),
  introduced in 2017, that underlies essentially all modern LLMs.
- **Decoder-only**: a Transformer variant using only the decoder half of the original
  encoder-decoder design, trained purely on next-token prediction — the GPT lineage's
  choice, and the architecture this repo's `tiny_llm.py` implements.
- **Scaling laws**: the empirical, predictable relationship between model size, data,
  compute, and resulting performance — see [Chapter 14](14_scaling_laws_and_model_sizing.md).
- **Instruction tuning / RLHF / DPO**: post-pretraining stages that shape a model toward
  following instructions and conversing usefully — see [Part 3](00_roadmap.md#part-3--fine-tuning-adapting-an-existing-model).
