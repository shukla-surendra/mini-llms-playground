# The NLP Architecture Landscape

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 0 — Deep Learning
Foundations. [Chapter 7](07_history_how_we_got_here.md) told the *chronological* story of
how architectures evolved; this chapter is a *taxonomy* — the families of architecture
NLP has used, side by side, so you can place any model you encounter (this repo's
included) into the right category and know what that category implies.

## In Plain English

Before settling on the Transformer, NLP tried several fundamentally different shapes for
processing text. Even now, "Transformer" isn't one single architecture — it splits into
three genuinely different sub-families depending on how text flows through it, each
suited to different tasks. Knowing which family a model belongs to tells you a lot about
what it can and can't do well, before you've read a single line of its code.

## The First-Principles Explanation

### Pre-neural and early-neural approaches, briefly

- **Bag-of-words / TF-IDF** — represent a document as word counts (or count-based
  weights), with *no* notion of word order at all. "Dog bites man" and "man bites dog"
  produce the identical representation. Simple, fast, and still genuinely useful for some
  classification tasks (spam detection, topic tagging), but structurally incapable of
  anything requiring word order or context.
- **CNNs for text** — apply a sliding filter (a small learned pattern-detector) across a
  sequence of token embeddings, good at picking up local patterns (a few words at a time)
  but with a limited, fixed receptive field per layer — capturing long-range dependencies
  requires stacking many layers, similar in spirit to the RNN long-range problem from
  [Chapter 7](07_history_how_we_got_here.md#generation-2-networks-with-memory-rnns-and-lstms-2000s-2014),
  though for a different underlying reason.
- **RNNs / LSTMs** — already covered in depth in
  [Chapter 7](07_history_how_we_got_here.md#generation-2-networks-with-memory-rnns-and-lstms-2000s-2014);
  process sequentially, one token at a time, carrying a hidden state forward.

### The Transformer's three architectural sub-families

This is the taxonomy worth knowing precisely — every Transformer-based model you'll
encounter is one of these three, and the difference is *not* cosmetic:

```
ENCODER-ONLY (e.g., BERT)
  Input:  full sentence, all at once
  Attention: BIDIRECTIONAL — every position can see every other position,
             including ones that come "after" it (no causal mask)
  Trained on: masked language modeling (hide some tokens, predict them
              using context from BOTH directions) — a different objective
              from next-token prediction
  Good at: understanding/classification tasks — sentiment analysis,
           named entity recognition, question answering (extractive)
  NOT naturally good at: generating new text left-to-right

DECODER-ONLY (e.g., GPT, TinyLlama, and THIS REPO'S TinyGPT)
  Input:  a sequence, processed left-to-right
  Attention: CAUSAL (masked) — a position can only see itself and earlier
             positions, per Chapter 10's causal-mask mechanism
  Trained on: next-token prediction (Chapter 8)
  Good at: open-ended text generation, conversation, and — via prompting
           at scale — a surprisingly wide range of tasks without being
           explicitly trained for each one
  NOT naturally good at: tasks needing full bidirectional context in one
                          pass (though large decoder-only models often
                          compensate well via scale and prompting)

ENCODER-DECODER (e.g., the original 2017 Transformer paper, T5)
  Input:  TWO sequences — a source (read bidirectionally, by the encoder)
          and a target being generated (causally, by the decoder, which
          also attends back to the encoder's output)
  Trained on: sequence-to-sequence tasks — translation, summarization
  Good at: tasks with a clear "input sequence -> output sequence" shape
  NOT naturally the current default for open-ended chat/generation,
  though still genuinely used for translation/summarization-shaped tasks
```

## Grounded in This Repo's Code

Both models in this repo are **decoder-only**, and the code makes this an explicit,
checkable architectural fact, not just a label:

```python
causal_mask = torch.triu(
    torch.full((seq_len, seq_len), float("-inf"), device=x.device),
    diagonal=1,
)   # CausalSelfAttention.forward(), tiny_llm.py, line 240
```

This one mechanism — covered fully in
[Chapter 10](10_transformer_architecture.md#the-causal-mask-the-detail-that-makes-this-gpt-like-not-bidirectional)
— is *the* line separating decoder-only from encoder-only. If this causal mask were
removed, `TinyGPT` would become architecturally bidirectional (an encoder), but it would
also no longer make sense to train it on next-token prediction — position `t` could see
position `t+1`'s answer directly, defeating the entire objective. **The architecture and
the training objective are a matched pair, not independent choices** — this is the
concrete mechanism behind why encoder-only and decoder-only models use genuinely
different training objectives (masked language modeling vs. next-token prediction), not
just different attention patterns.

`TinyLlama-1.1B-Chat`, fine-tuned in
[`fine_tuning/tinyllama-1.1b-lora/`](../../fine_tuning/tinyllama-1.1b-lora/), is *also*
decoder-only — the same family, just pretrained by a different team at much larger scale
and already instruction-tuned
([Chapter 7](07_history_how_we_got_here.md#generation-6-from-predicts-text-to-follows-instructions-2022-present)).
Nothing in this repo touches encoder-only or encoder-decoder architectures — worth naming
explicitly so you know that's a genuine gap in what this repo teaches hands-on, not an
oversight to be confused with "the only architecture that exists."

## Deep-Dive: Why Decoder-Only Won the Current LLM Era

This is worth understanding as a real historical bet, not an inevitability:

- **One objective, one architecture, remarkable generality** — next-token prediction on
  raw text requires no labeled data at all (unlike masked language modeling's specific
  masking scheme, or sequence-to-sequence's paired input/output data), making it far
  easier to scale to enormous, largely unlabeled datasets.
- **Generation is the harder direction, and decoder-only is built for it natively** — an
  encoder-only model isn't naturally suited to generating open-ended text at all; a
  decoder-only model can be *used* for understanding-flavored tasks too, via careful
  prompting, even though it wasn't explicitly trained with that objective — a real,
  empirically observed flexibility advantage.
- **Simplicity at scale** — encoder-decoder architectures have more moving parts (two
  separate stacks, cross-attention between them); decoder-only's single, simpler stack
  turned out to scale efficiently to very large sizes, which mattered enormously once
  scaling ([Chapter 14](00_roadmap.md#part-2--pretraining-building-a-model-from-zero))
  became the dominant lever for capability improvements.

Encoder-only and encoder-decoder aren't obsolete — BERT-family models remain a strong,
efficient choice for pure classification tasks, and T5-family encoder-decoder models
remain genuinely used for translation and summarization — but the current wave of
general-purpose conversational LLMs (including everything in this repo) is decoder-only,
for the specific, real reasons above.

## Try It Yourself

- In `tiny_llm.py`, imagine (don't need to actually run this destructively) removing the
  `causal_mask` argument from the `self.attn(...)` call in `CausalSelfAttention.forward`.
  Predict, using this chapter's taxonomy: what architecture family would this become, and
  would the existing `masked_next_token_loss` training objective still make sense against
  it? (It wouldn't — this is the exact mismatch the deep-dive names.)
- Look up any model you've heard of (BERT, T5, GPT-4, Claude, LLaMA) and classify it into
  one of this chapter's three families before checking — building the habit of asking
  "which architecture family is this" as a first orienting question about any new model.

## Common Misconceptions

- **"BERT and GPT are basically the same kind of model with different names."** They're
  architecturally and objective-wise different — bidirectional vs. causal attention,
  masked-token prediction vs. next-token prediction, understanding-oriented vs.
  generation-oriented by design.
- **"Encoder-decoder architectures are obsolete now that decoder-only LLMs exist."**
  Not accurate — they remain a strong, still-used choice for translation and
  summarization specifically, tasks with a clean input-sequence-to-output-sequence shape.
- **"Decoder-only models can't do understanding/classification tasks well."** Large
  decoder-only models handle these tasks surprisingly well via prompting, even without
  the bidirectional-attention advantage encoder-only models have by design — an
  empirical result, not something the architecture guarantees in principle.

## Practice Questions

1. Why can't a causal (decoder-only) attention mask be trained on a masked-language-
   modeling objective the way an encoder-only model is — what would go wrong?
2. Name the three Transformer sub-families and, for each, one real task it's naturally
   well-suited to and one it's not.
3. This repo's `TinyGPT` and `TinyLlama` are both decoder-only. What would have to change
   in `tiny_llm.py` — architecturally, not just in configuration — to turn it into an
   encoder-only model instead?

## Key Terms

- **Bag-of-words / TF-IDF**: word-count-based text representations with no notion of
  order.
- **Bidirectional attention**: attention allowed to look at both earlier and later
  positions (encoder-only models).
- **Causal (masked) attention**: attention restricted to earlier positions only
  (decoder-only models) — see [Chapter 10](10_transformer_architecture.md).
- **Masked language modeling**: the training objective of hiding some tokens and
  predicting them from bidirectional context (BERT-family).
- **Sequence-to-sequence (seq2seq)**: tasks and architectures mapping one full input
  sequence to one full output sequence (translation, summarization).
- **Encoder-only / decoder-only / encoder-decoder**: the three Transformer architecture
  families, distinguished by attention direction and training objective.
