# Data Preparation Strategies for Pretraining

Part of the [LLM Engineering Curriculum](00_roadmap.md). Numbered 34 to avoid renumbering
already-written chapters (same reason Part 2B/3B were appended after the original catalog
— see the roadmap's "Reading order"), but belongs **right after
[Chapter 12](12_the_pretraining_objective_and_why_data_dominates.md)** in reading order.
Chapter 12 argues *why* data dominates over objective/architecture; this chapter is the
practical follow-up — *what to actually do* to a raw corpus before training on it, and what
breaks, mechanistically, when each step is skipped. Grounded in
[`from_scratch/custom-gpt-50m/`](../../from_scratch/custom-gpt-50m/)'s real pipeline
(`src/gpt/data/`) and a real diagnostic artifact: a QA report pulled from that project's own
step-306,799 checkpoint, used throughout as "here is what this failure mode actually looks
like in raw model output," not a hypothetical.

## In Plain English

A next-token predictor learns exactly the statistical patterns present in its training
stream, in proportion to how often it sees them, shaped by however that stream is packed
and segmented. Nearly every data-prep decision is one of five levers on that single fact:
**what** goes in, **how much** noise is mixed in, **how often** each source is seen relative
to the others, **how** the stream is chunked/segmented, and **what atomic units** it's
broken into before training even starts. Get any one of these wrong and the model doesn't
fail loudly — it just quietly learns the wrong proportions, or the wrong boundaries, and
that shows up later as a specific, traceable behavior in generation.

## The First-Principles Explanation

### Lever 1 — Source selection: what counts as signal at all

Public conversational corpora, scraped/extracted domain text, and synthetic
model-generated data are not interchangeable inputs — they carry different diversity
profiles even at equal token count. Synthetic data (model-generated, like
`HuggingFaceH4/ultrachat_200k` and `HuggingFaceTB/smoltalk`) is cheap and internally
consistent, but risks **homogenization**: the same templated phrasing gets reproduced far
more densely than a real population of human writers would ever produce it, because it all
traces back to one generating model's habits.

### Lever 2 — Filtering/cleaning: removing what shouldn't count as signal

Two different filtering strategies, catching different failure classes:

- **Heuristic filters** — length thresholds, printable/ASCII ratio, redaction-placeholder
  rejection. Cheap, deterministic, catches outright garbage. This is what
  `custom-gpt-50m`'s own `is_quality_text()` does
  ([`src/gpt/data/prepare.py:65`](../../from_scratch/custom-gpt-50m/src/gpt/data/prepare.py#L65)) —
  `min_chars` (turn length floor) and `min_ascii_ratio=0.995` are the two live thresholds,
  applied per turn inside `extract_turns_conversation`/`extract_turns_instruction`.
- **Model-based/classifier filters** — a small trained classifier scores "does this read
  like high-quality prose" (e.g. the fastText quality classifier GPT-3 and Llama's
  pretraining pipelines used, trained with Wikipedia-like text as positive examples).
  Heuristics catch garbage; classifiers catch text that's clean but low-quality, which
  heuristics structurally can't see.

### Lever 3 — Deduplication: three granularities, three different failure modes

A document repeated many times isn't "more signal" — it's memorization risk, and at small
model scale it directly inflates repetitive-phrasing patterns the model then reproduces
under low-entropy decoding (greedy, low temperature).

| Granularity | Mechanism | Catches |
|---|---|---|
| Exact dedup | hash whole documents | literal copies |
| Near-dedup | MinHash/LSH over shingles | templated-but-not-identical text (the synthetic-data risk from Lever 1) |
| Substring dedup | suffix array over the whole corpus (used by GPT-3/Llama papers) | repeated *spans* inside otherwise-different documents |

`custom-gpt-50m`'s own `audit.py` ([`audit.py:42`](../../from_scratch/custom-gpt-50m/src/gpt/data/audit.py#L42))
only implements `overlap_rate()` — exact-line **train/test leakage** detection. There is no
within-train duplicate check at any of the three granularities above. That's a real,
current gap, not a stylistic choice — see "Grounded in This Repo's Code" below for what it
looks like in practice.

### Lever 4 — Mixture weighting vs. capping: a distinction worth being precise about

Whatever fraction of raw documents a source contributes *is* its training frequency,
unless something deliberately overrides it. Two different mechanisms get confused often
enough to be worth separating explicitly:

- **A cap** (a ceiling) prevents one large source from drowning out the others.
  `custom-gpt-50m` has this: `max_per_dataset=100_000`
  ([`prepare.py:331`](../../from_scratch/custom-gpt-50m/src/gpt/data/prepare.py#L331)).
- **A weight** (oversampling/downsampling) deliberately changes a source's *effective*
  frequency relative to its raw size — e.g., sampling a small, high-value source 3x so it's
  seen more often than its scrape volume alone would produce. `custom-gpt-50m` does **not**
  have this. GSM8K enters the mix at exactly its raw count (16,860 conversations, ~6.8% of
  the corpus's 249,589 total chat conversations) — nobody chose that ratio for pedagogical
  reasons, it's just what the source happened to contain. A cap stops a source from being
  *too big*; only a weight can make an undersized-but-important source *bigger than its
  natural share*.

### Lever 5 — Chunking, packing, and document-boundary masking

Two separate decisions, often conflated:

1. **Chunk size** — how big a unit an extra (non-chat) document gets split into before
   entering the corpus. `custom-gpt-50m` chunks at 1024 tokens with 100-token overlap,
   deliberately matched to its own `context_length=1024`, so a random training window is
   more likely to see one coherent document rather than fragments of several unrelated
   short ones — and the 100-token overlap means a window landing near a chunk seam still
   has real lead-in context on both sides, not a truncated sentence.
2. **Attention behavior across packed documents** — when multiple documents are
   concatenated into one training stream (as they are here, joined by GPT-2's real
   `<|endoftext|>` special token, not a soft separator like `"\n\n"`), does the model's
   causal attention actually treat that boundary as a hard reset, or can token *N* still
   attend across it into an unrelated document that happens to precede it in the packed
   window?

The second question matters more than it sounds like it should, and Lever 5 is where "why
does the training pipeline data-prep decision, not just a training-loop detail" argument
lives: **whether attention resets at a document boundary is set at data/packing-prep time**
(via a segment id carried alongside the tokens), even though it's *enforced* inside the
attention computation. A pipeline that just concatenates token ids with a separator token
and hands the flat stream to `get_batch()` has already made the decision — implicitly,
usually unintentionally — that boundaries are a soft statistical cue, not a hard structural
one.

### Lever 6 — Tokenization: the atomic units the model can build patterns from at all

Standard BPE (GPT-2's `tiktoken` `r50k_base`, what `custom-gpt-50m` uses) merges frequent
character sequences — including digits — based on corpus-wide frequency, not on preserving
place value. The same three-digit number can tokenize as one merged piece in one context
and three separate digit tokens in another, purely as an artifact of what else was nearby
during BPE training. This is a documented, *separate* cause of poor arithmetic in small
language models, independent of how much math data is in the mix or how far through
training the model is. Some math-focused setups (Minerva, several GSM8K-tuned small models)
switch to digit-by-digit tokenization specifically to remove this source of error — a real
lever, but an expensive one to retrofit, since it changes the embedding table shape and
invalidates any existing checkpoint (see
[Chapter 27](27_checkpointing_and_resuming_training.md) on why a shape mismatch can't be
silently resumed).

### Lever 7 — Format and structure: base pretraining and instruction-tuning are prepared for genuinely different objectives

Base pretraining — what `custom-gpt-50m` does — computes loss on *every* token in the
stream, including the `User:` turn, not just the `Assistant:` reply. Instruction-tuning
(SFT, [Chapter 18](18_instruction_tuning_and_sft.md)) is a structurally different data-prep
job: a much smaller, heavily curated dataset where loss is **masked** so gradient only
flows through the response tokens. This is the actual mechanism behind "why doesn't my base
model refuse harmful requests or reliably follow format constraints" — it isn't a corpus
*mixture* problem fixable by adding more of the right conversations; the base objective
never specifically rewards "stop generating after the assistant's turn" or "decline this."
That behavior has to be taught by a separate, explicitly loss-masked training stage.

### Lever 8 — Split and leakage prevention

Split before near-dedup and you leak near-duplicates across train/test without the exact-
match leakage check ever catching it. Split each source independently, then pool (what
`custom-gpt-50m` does — see `DATASET.md`'s "Pooling and splitting" step), rather than
splitting the already-pooled whole, so one source's internal ordering can't dump entirely
into a single split.

### A frame that sits above all eight levers: data-to-parameter scaling

Chinchilla's finding — roughly 20 training tokens per parameter is compute-optimal — isn't
a *prep* strategy so much as a sanity check on whether prep is even the current bottleneck.
`custom-gpt-50m` has 51,475,968 parameters and 280,330,103 train tokens: at ~5.4
tokens/parameter, it's meaningfully below the ~20:1 compute-optimal ratio, which means (at
this parameter count, this far into a fixed-length run) the higher-leverage move is likely
more steps on the existing corpus, not more raw text — a separate axis from every lever
above, worth checking before assuming "the data must be the problem."

## Grounded in This Repo's Code

Every failure mode above has a real, observed instance in
[`custom-gpt-50m`'s own step-306,799 QA report](../../from_scratch/custom-gpt-50m/reports/):

- **Lever 5 (no document-boundary masking)** is the direct, mechanistic explanation for
  answers that drift mid-generation into `<|endoftext|>User: ...` and start a new,
  unrelated conversation turn — seen repeatedly across the report (e.g. the "why is the sky
  blue" answer sliding into "I'm looking for a book I need to read a list of my favorite
  books"). `get_batch()`
  ([`dataset.py:269`](../../from_scratch/custom-gpt-50m/src/gpt/data/dataset.py#L269))
  samples a random 1024-token window from the flat token stream with plain causal
  attention — it has no concept of where a `<|endoftext|>` seam falls inside that window,
  so the model was trained, many times, to predict tokens after a boundary using full
  attention over an unrelated document before it.
- **Lever 4 (cap, not weight)** is consistent with the report's near-total GSM8K-style
  arithmetic failure — a source that's ~6.8% of chat volume purely by scrape luck gets
  exactly that much training exposure, with no mechanism to boost it further.
- **Lever 3 (no within-train dedup)** is consistent with the report's repetition-under-
  greedy-decoding behavior and templated phrasing ("Trello, Trello, or Trello," "the
  'Projection' feature... the 'Projection' feature...") — patterns synthetic sources like
  UltraChat/SmolTalk are prone to reproducing at higher-than-natural density.
- **Lever 7 (base, not instruction-tuned)** is the explicit, stated design in
  [`DATASETS.md`](../../from_scratch/custom-gpt-50m/docs/DATASETS.md#L120): *"Training does
  not treat this as chat... There is no chat template and no per-turn loss masking."* The
  report's weak refusal/format-constraint behavior is exactly what that design predicts,
  not a bug.

## Deep-Dive: Why a Document Separator Token Alone Doesn't Do What It Looks Like It Does

It's tempting to think reserving a real special token for document boundaries (rather than
a plain `"\n\n"`) *is* the fix for cross-document bleed-through — and it is a real
improvement over a soft separator, since GPT-2's own pretraining already primed that token's
embedding as a meaningful boundary cue. But it's a **statistical** cue, learned the same way
every other pattern is learned: by how often the model saw it predict "unrelated content
follows." It is not a **structural** constraint the way causal masking is. Causal masking
makes it *architecturally impossible* for position `t` to attend to position `t+1` — there's
no amount of counter-evidence in the data that changes that. A special token by itself only
makes cross-boundary attention *statistically discouraged*, and an undertrained model (30%
through a 1M-step run, as `custom-gpt-50m`'s checkpoint was) hasn't yet learned that
discouragement reliably. The structural fix — a segment id that hard-masks attention across
a packed sequence's document boundaries — doesn't have this dependency on training progress
at all; it's correct from step zero.

## Try It Yourself

- Read `overlap_rate()` in
  [`audit.py`](../../from_scratch/custom-gpt-50m/src/gpt/data/audit.py) and extend it with a
  same-granularity check *within* the train split (not just train-vs-test) — even a cheap
  exact-hash-per-conversation count would surface Lever 3's current blind spot.
- Compute each of the 7 registered chat sources' actual share of `data/train.txt` (by
  conversation count, then again by token count) and compare it to what you'd *choose* if
  you were deliberately weighting for capability coverage rather than accepting raw scrape
  volume — this makes Lever 4's "cap vs. weight" distinction concrete rather than abstract.
- Pick any answer in `reports/qa_report_50m_step306799.html` that drifts into
  `<|endoftext|>User: ...` mid-generation, and trace back which two documents in
  `data/train.txt` most plausibly sit on either side of that seam in some training window —
  this is Lever 5's failure mode from the inside.

## Common Misconceptions

- **"More data always helps."** Only up to the compute-optimal ratio for the parameter
  count in use (the Chinchilla frame above) — past that point, more *steps* on existing data
  usually has higher leverage than more raw text.
- **"A reserved document-separator token is enough on its own, without attention masking."**
  It's a strong statistical cue, not a structural guarantee — see the Deep-Dive above. Only
  hard attention masking at pack time removes the dependency on the model having learned the
  cue reliably.
- **"Whatever fraction a source is of the raw scrape is the fraction it *should* be of
  training."** That's an accident of what happened to be easy to collect, not a deliberate
  choice about what capability the model needs — Lever 4's cap-vs-weight distinction exists
  specifically because these are not the same thing.
- **"A base model trained on `User:`/`Assistant:`-formatted text is already an instruction-
  following assistant."** It's a next-token predictor over that format, which is a
  genuinely different thing from a model *trained to stop, refuse, or follow constraints* —
  that requires the loss-masked SFT stage in [Chapter 18](18_instruction_tuning_and_sft.md),
  not just more of the same base-format data.

## Practice Questions

1. A corpus has a document separator token, but training windows are sampled from a flat,
   unmasked token stream. Explain, mechanically (not just "it'll be worse"), why a
   generation can still drift across an unrelated document boundary even after the model has
   seen millions of examples of that separator token.
2. Two datasets: Dataset A is 200,000 synthetic conversations from one generating model;
   Dataset B is 15,000 hand-written, single-turn instruction/response pairs from human
   annotators. Equal token count aside, what does each actually contribute to corpus
   diversity, and what's the specific risk of Dataset A dominating the mixture by raw
   volume?
3. A pipeline has a `max_per_dataset` cap but no oversampling weight. Explain why this means
   a small, high-value source (e.g., a math dataset) can never be trained on more heavily
   than its raw scrape size, no matter how important it is to final model capability.
4. Why does fixing Lever 5 (document-boundary masking) not also fix Lever 7 (lack of
   refusal/format-following behavior)? What would actually be required to fix Lever 7?

## Key Terms

- **Exact / near / substring deduplication**: three granularities of duplicate detection —
  identical documents, templated-but-not-identical documents (MinHash/LSH), and repeated
  spans inside otherwise-different documents (suffix array).
- **Mixture capping vs. mixture weighting**: a cap limits a source's maximum contribution; a
  weight deliberately raises or lowers a source's effective training frequency relative to
  its raw size. Capping alone cannot boost an undersized source above its natural share.
- **Document-boundary (segment) masking**: a hard attention-level constraint, set at
  pack/data-prep time, that prevents a packed sequence's tokens from attending across a
  document boundary — structurally different from, and stronger than, a learned statistical
  cue like a reserved separator token.
- **BPE digit fragmentation**: byte-pair-encoding's tendency to tokenize the same numeric
  sequence differently depending on surrounding context, since merges are chosen by
  corpus-wide frequency rather than to preserve place value — a documented contributor to
  poor arithmetic in small language models.
- **Compute-optimal token-to-parameter ratio (Chinchilla)**: roughly 20 training tokens per
  parameter as the point past which more raw data has diminishing marginal value relative to
  more training steps, for a fixed parameter count.
