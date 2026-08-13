# The Pretraining Objective & Why Data Dominates

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 2 — Pretraining: Building a
Model From Zero. Builds on [Chapter 10](10_transformer_architecture.md)'s architecture and
[Chapter 3](03_how_neural_networks_learn.md)'s training loop — this chapter is about *what
task* the network is trained to solve, which turns out to be a genuinely separate design
decision from the architecture itself. Grounded in three real, working implementations in
[`from_scratch/tinystories-gpt-6m/`](../../from_scratch/tinystories-gpt-6m/): causal LM
(`model.py`/`train.py`), masked LM (`model_mlm.py`/`train_mlm.py`), and contrastive
learning (`model_contrastive.py`/`train_contrastive.py`).

## In Plain English

"Pretraining" means teaching a model something useful using only raw, unlabeled data — no
human needed to write "correct answers" for each example. The task the model is actually
given to practice on (predict the next word, fill in a blanked-out word, recognize that two
noisy versions of the same sentence are "the same") is the **pretraining objective** — and
different objectives, on the *same* raw text, produce models with different strengths, need
different attention patterns, and are chosen for different downstream purposes. There isn't
one "correct" pretraining objective; there are several real ones, each a genuine trade-off.

## The First-Principles Explanation

### What makes an objective "self-supervised"

None of the three objectives below need a human to label anything. The "labels" are
derived automatically from the raw text itself — the next real token, the original token
under a mask, or the fact that two views came from the same source sequence. This is what
makes pretraining scalable to enormous unlabeled text corpora in the first place — the
entire reason pretraining works at internet scale is that supervision is manufactured from
the data, not collected from humans.

### Objective 1: Causal (next-token) language modeling

Predict token `t+1` from tokens `0..t`, one token at a time, left to right. Requires
**causal attention** — position `t` must never see position `t+1` or later, or the "task"
becomes trivially easy (copy the answer that's already visible) and the model learns
nothing useful. This is what GPT-family models train on, and it's the objective this
project's main `model.py`/`train.py` implements — see
[`../../from_scratch/tinystories-gpt-6m/docs/TRAINING.md`](../../from_scratch/tinystories-gpt-6m/docs/TRAINING.md)
for real training numbers. Its natural strength: the objective *is* the deployment task for
a generative model — a model trained to predict the next token is, by construction, already
doing the thing you want at inference time (autoregressive generation).

### Objective 2: Masked language modeling (BERT-style)

Randomly hide ~15% of tokens, predict the original token at each hidden position using
**bidirectional attention** — every position can see every other position, before and
after. This requires an architecture change, not just a loss-function change: bidirectional
attention would make causal LM's task trivial (as noted above), and causal attention would
make masked LM's task impossible for right-context information. This project's
`model_mlm.py` implements this — see
[`../../from_scratch/tinystories-gpt-6m/docs/MASKED_LM.md`](../../from_scratch/tinystories-gpt-6m/docs/MASKED_LM.md)
for the exact 80/10/10 masking policy and real training numbers. Its natural strength:
every position gets to use full-sentence context, which tends to produce strong
*representations* for understanding tasks (classification, retrieval) — at the cost of not
being naturally generative, since there's no well-defined "next token" step at inference
time the way there is for causal LM.

### Objective 3: Contrastive self-supervised learning

Given two different "views" of the same underlying input, train the model to produce
similar representations for that pair, and dissimilar representations for unrelated inputs
— no masking or next-token prediction at all, just a similarity judgment. The "two views"
can come from many places: data augmentation (cropping/rotating an image, back-translating
a sentence), or — the technique this project's `model_contrastive.py` uses (SimCSE, Gao et
al. 2021) — simply passing the *same* input through the same model twice, letting dropout's
randomness be the only difference between the two passes. See
[`../../from_scratch/tinystories-gpt-6m/docs/CONTRASTIVE_LEARNING.md`](../../from_scratch/tinystories-gpt-6m/docs/CONTRASTIVE_LEARNING.md)
for the InfoNCE loss mechanics and real numbers. Its natural strength: directly optimizes
for a similarity-embedding space, which is exactly what retrieval, search, and
clustering-style production use cases need — closer to the actual downstream task than
either causal or masked LM's token-prediction framing.

### Why "data dominates" — the deeper point behind all three

Whichever objective is chosen, the ceiling on what pretraining can teach is set by the
data, not the objective — an objective can only manufacture supervision *from information
already present in the corpus*. A model can't learn a fact, a style, or a capability the
training data never demonstrated, no matter how cleverly the objective extracts signal from
what *is* there. This is why, in practice, decisions about pretraining data composition,
scale, and quality tend to matter as much as (often more than) the specific objective
chosen — a theme Chapter 14 ("Scaling Laws," planned — see [Chapter 0](00_roadmap.md)) picks up in
more depth.

## Grounded in This Repo's Code

All three objectives train on **the exact same tokenized data and tokenizer**
(`from_scratch/tinystories-gpt-6m/data/`, prepared once by `prepare_dataset.py`) — a
deliberate design choice in this project specifically so the *only* variable across the
three training runs is the objective itself, not the underlying data:

```python
# model.py — causal: GPTBlock(..., causal=True), naive/SDPA attention with is_causal mask
# model_mlm.py — bidirectional: GPTBlock(..., causal=False), same block class, mask off
# model_contrastive.py — reuses TinyStoriesGPT (causal, unchanged) as an encoder,
#                         pools the last token, no architecture change needed at all
```

Notice that objective 3 required *zero* architecture changes to the backbone — this is a
real, working illustration of exactly how modern causal-LM-based embedding models (E5,
LLM2Vec) are actually built in production: take an existing pretrained decoder, don't
touch its architecture, add a small pooling+projection head, and pretrain that head (and
optionally fine-tune the backbone) with a contrastive objective.

## Deep-Dive: Why Bidirectional Attention Can't Just Be "Turned On" for Causal LM

A natural question: since `model_mlm.py` shows bidirectional attention working fine, why
not just train the causal-LM model with bidirectional attention too, to get "the best of
both"? Because the *objective* and the *attention pattern* aren't independently
choosable — they're coupled. Causal LM's supervision signal is "predict what comes next";
if attention can see what comes next, the correct answer is sitting directly in the input,
and gradient descent will happily learn to just copy it, producing a model that's
completely unable to generate text (there's no "next" to look at when actually generating,
token by token, at inference time). Bidirectional attention is only usable with an
objective, like masked LM, where the *thing being predicted* is deliberately hidden from
the model's own input — otherwise the task degenerates instead of teaching anything.

## Try It Yourself

- Train a few hundred steps of all three objectives in this project
  (`make train`, `make train-mlm`, `make train-contrastive`, each with `STEPS` set low for
  a quick run) and compare what each `logs/*.csv` eval history actually measures — notice
  that none of the three loss curves are on a directly comparable scale, for reasons each
  objective's own doc explains.
- Read `apply_bert_masking` in `model_mlm.py` and `info_nce_loss` in `model_contrastive.py`
  side by side — both are manufacturing supervision from unlabeled text, using genuinely
  different mechanisms, on the same underlying data.

## Common Misconceptions

- **"Masked LM is just causal LM with a different masking pattern."** The masking pattern
  difference is real, but the deeper difference is architectural: masked LM requires
  bidirectional attention, a structural change to what each position is even allowed to
  see, not a superficial loss-function swap.
- **"Contrastive learning needs paired/augmented data you have to collect."** SimCSE's
  insight (used in this project) is that it doesn't — dropout noise from two forward
  passes of the *same* input is enough to manufacture a usable positive pair, no augmented
  dataset required.
- **"A more clever objective can make up for weak or small training data."** As the "why
  data dominates" section argues, an objective can only extract supervision from
  information the data actually contains — it can't manufacture facts, style, or
  capability the corpus never demonstrated.

## Practice Questions

1. Why does causal LM require causal attention specifically, and what would go wrong
   (mechanically, not just "it would be worse") if bidirectional attention were used with
   the same next-token objective?
2. Explain, in your own words, why `model_contrastive.py` didn't need any change to the
   Transformer block architecture at all, while `model_mlm.py` did.
3. Two production use cases: (a) a chatbot that generates free-form responses, (b) a
   semantic search feature that finds similar support tickets. Which of the three
   objectives in this chapter is the most natural fit for each, and why?

## Key Terms

- **Self-supervised learning**: training signal manufactured automatically from unlabeled
  data (the next real token, the original token under a mask, same-source-pair identity),
  not collected from human annotators.
- **Causal (autoregressive) language modeling**: predict the next token from only
  left-context; requires causal attention.
- **Masked language modeling**: predict deliberately hidden tokens using full
  bidirectional context; requires bidirectional attention.
- **Contrastive learning / InfoNCE**: train representations to be similar for positive
  pairs (same underlying source) and dissimilar for negatives (everything else in the
  batch), via a softmax-based loss over similarity scores.
