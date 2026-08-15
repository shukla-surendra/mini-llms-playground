# Data Preparation Guideline: Maximum Impact Per Parameter

Written while planning a pivot from this project's current goal (proving the pipeline
mechanics on a broad, 5-source general-chat corpus — see the top-level
[`README.md`](../README.md)) toward a **domain-specialized** model: same ~10M-parameter
budget, aimed at being genuinely good within one narrow domain instead of mediocre
across a broad one — the same trade-off
[`tinystories-gpt-6m`](../../tinystories-gpt-6m/) already demonstrates working at this
scale (see that project's [`README.md`](../../tinystories-gpt-6m/README.md#why-this-is-a-different-project-from-custom-gpt-153m-not-a-smaller-copy-of-it)
comparison table).

Ranked by actual leverage, not by pipeline order — get the top items right before
spending effort further down. The domain itself is a placeholder below (`<DOMAIN>`) —
fill in the concrete choice, source list, and numbers once it's picked; everything else
here applies regardless of which domain is chosen.

## 1. Source selection: depth and correctness over breadth

For a knowledge domain specifically, a small model can't fact-check or hedge — it
reproduces whatever's in the training data, confidently, whether it's right or wrong.
Prioritize a small number of **authoritative, internally consistent** sources over
scraping broadly. One well-vetted source beats five mismatched ones; conflicting facts
across sources actively hurt, since the model can't learn "it depends" — it averages
toward incoherence instead. See
[Chapter 12 — The Pretraining Objective & Why Data Dominates](../../../docs/llm-engineering/12_the_pretraining_objective_and_why_data_dominates.md)
for why the data ceiling, not the objective or architecture, is what ultimately caps
model quality.

- `<DOMAIN>`: TBD
- Candidate sources: TBD

## 2. Deduplicate before anything else

Near-duplicate examples (common in scraped/aggregated domain data) waste training steps
re-teaching the same pattern and inflate apparent corpus size without adding real
coverage. Dedup at the conversation/document level (exact or near-exact) **before** the
quality filters below — filtering first just wastes filter effort on rows that would be
discarded as duplicates anyway.

## 3. A domain-fit tokenizer, not GPT-2's off-the-shelf one

The single highest-leverage change available. This project's own numbers already prove
why: at the `10m` preset, `token_embedding` accounts for **8,041,120 of 9,979,040 total
parameters — 80.6%** (`make config`'s breakdown), sized for GPT-2's full 50,257-token
general-English vocabulary. A domain-fit custom BPE vocabulary — the same approach
[`tinystories-gpt-6m`](../../tinystories-gpt-6m/docs/DATASET_AND_TOKENIZER.md) uses —
frees most of that budget for actual reasoning capacity instead of an oversized lookup
table for tokens the domain will barely use.

**How to size it, not guess it**: train the tokenizer at a few candidate vocab sizes
(e.g. 2K / 4K / 8K) on the actual domain corpus, and check average tokens-per-example
and byte-fallback rate at each. Pick the smallest vocab where that rate stops
meaningfully improving — undersized vocab wastes capacity on excessive byte-level
fallback, oversized vocab wastes capacity the same way GPT-2's 50,257 does now. See
[Chapter 9 — Tokenization](../../../docs/llm-engineering/09_tokenization.md) for the
underlying mechanism.

- Candidate vocab sizes to benchmark: TBD
- Chosen size + justification: TBD

## 4. A strong document-boundary token, from the start

Reserve a real special token in the tokenizer's own vocabulary (train it in from the
start, the way `tinystories-gpt-6m` does with `<|endoftext|>`) rather than retrofitting
one later. This project's current general-chat corpus uses a plain `"\n\n"` separator
between conversations — a documented, currently-unfixed weak spot (see
[`TRAINING_QA.md`](TRAINING_QA.md#does-arranging-data-in-a-particular-way-increase-model-performance))
— worth designing correctly from the start here rather than repeating it.

## 5. Consistent example structure

Pick one input/output format (e.g. a fixed `Question: ... Answer: ...` shape) and
normalize every source into it. Heterogeneous formatting (some examples Q&A, some prose,
some bullet lists) forces the model to spend capacity learning format-switching instead
of domain content. Databricks Dolly 15k's clean, single-format design (per
[`DATASETS.md`](DATASETS.md)) is the right model to copy — this project's current
5-source blend, by contrast, mixes conversation-schema and instruction-schema sources
together.

- Chosen format: TBD

### What "consistent structure" looks like, concretely

Same underlying fact, three different shapes — this is exactly what to normalize away
before writing `train.txt`, not a hypothetical:

```
# shape 1 (Q&A-labeled)
Q: What is the boiling point of water at sea level?
A: 100°C (212°F) at standard atmospheric pressure.

# shape 2 (prose)
Water boils at 100°C when at sea level under normal atmospheric conditions...

# shape 3 (bullet list)
- Boiling point: 100°C
- Pressure: 1 atm
```

Trained on a mix of all three, the model has to spend real parameter budget just learning
"these are all the same kind of thing" — capacity a ~10M-parameter model can't spare —
before it can even get to learning the fact itself. Pick one shape (this project's own
pipeline already normalizes every source into `"User: ...\nAssistant: ..."` via
`turns_to_text()` in `src/gpt/data/prepare.py` — a reasonable default to keep unless the
domain gives a specific reason not to) and make every example match it exactly.

**Why this matters more than it sounds like**: training is next-token prediction over
literal text ([Chapter 3](../../../docs/llm-engineering/03_how_neural_networks_learn.md))
— the model has no concept of "this is a question" beyond the literal token sequence that
precedes an answer in training. A consistent prefix (`"Question: "`, `"User: "`, whatever
is chosen) becomes a strong, low-entropy signal the model can lock onto cheaply: seeing it
reliably predicts "answer-shaped content follows." Scatter that same underlying intent
across three different phrasings and the signal fragments — the model has to learn three
separate patterns instead of reinforcing one.

**This also means the training format and the inference-time prompt format must match
exactly, not just be internally consistent with each other.** This project's models are
raw next-token predictors with no chat template (see the top-level
[`README.md`](../README.md)'s "Training objective: raw, not instruction-tuned" section) —
there is no mechanism that infers "the user wants a Q&A-style answer" from intent alone.
If every training example is phrased `"User: ...\nAssistant: ..."`, prompting the trained
model with `"Q: ..."` at inference time is a different, unseen prefix — the completion
pattern the model actually learned won't reliably trigger. Whatever format is chosen here
is also the required prompt format for every future `make infer`/`make serve` call against
this model.

### Does the data need to be rearranged?

Depends on *what* — this project's `TRAINING_QA.md` already answers this precisely
against the actual training code, not in the abstract (see
[`TRAINING_QA.md`'s full answer](TRAINING_QA.md#does-arranging-data-in-a-particular-way-increase-model-performance)).
Short version, and what it means for a new domain corpus specifically:

- **Physical order in the file: doesn't matter.** `get_batch()` samples a random
  `context_length`-token window every step, not a sequential walk — reordering rows in
  `train.txt` changes nothing about training. Don't spend effort curating file order.
- **Interleaving sub-topics before writing: matters, and follows the same principle as
  item 6 below.** If the domain has several sub-topics, don't write all of sub-topic A
  followed by all of sub-topic B — shuffle them together first, the same way
  `build_corpus()` already does across this project's five general-chat sources via
  `random.shuffle()`. Un-shuffled blocks mean *which* sub-topic a random window lands in
  is no longer independent of file position, which has real consequences even under
  random sampling (see [Chapter 28](../../../docs/llm-engineering/28_catastrophic_forgetting_and_continual_training.md)).
- **How example boundaries are marked: matters, and is item 4 above.** A soft separator
  like `"\n\n"` is a weak "unrelated content starts here" signal; a real reserved token is
  a strong one. Get this right from the start for a new domain tokenizer rather than
  inheriting the current corpus's known weak spot.

## 6. Sub-topic balance within the domain

Even a narrow domain has sub-areas. Count examples per sub-topic before finalizing the
corpus — an accidental 80/20 skew means the model is effectively narrower than intended,
and rare sub-topics may barely get learned at all. Once sources are chosen, tabulate
counts per sub-topic here before building `train.txt`.

## 7. Quality audit gate, quantified — not eyeballed

Before committing to a long run, extend this project's existing `make audit` gate
(`src/gpt/data/audit.py`) with:

- ASCII/encoding-noise ratio (already present)
- Empty/truncated-example rate (already present)
- **Duplicate rate** (new — per item 2 above, not currently checked)
- **Per-sub-topic example counts** (new — per item 6 above)

## 8. Size the corpus to the training budget honestly

A narrow domain will almost certainly be a **smaller** corpus than the current
173M-token general mix. Once the real token count is known, recompute the epoch math
using [`TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md)'s formula
(`steps × batch_size × context_length ÷ train_tokens`), and expect to watch the
train/test gap more closely than the current run does — small corpus + small vocab +
narrow domain is a combination that **overfits faster**, not slower. See
[Chapter 15 — Evaluating a Model While It's Still Training](../../../docs/llm-engineering/15_evaluating_a_model_while_training.md)
for the stopping-rule framework to apply once real numbers are in.

- Corpus token count: TBD
- Recomputed epoch math at current `steps`: TBD

## The one-way door worth naming explicitly

A new tokenizer means the current checkpoint (`checkpoints/10m/`, 250K+ steps on the
general-chat corpus) **cannot be resumed or reused** — different vocab, different
embedding-table shape, incompatible from the ground up
([Chapter 27](../../../docs/llm-engineering/27_checkpointing_and_resuming_training.md)'s
self-describing-checkpoint mechanism refuses a shape mismatch rather than silently
corrupting it). Pivoting to a domain-specialized model via a custom tokenizer is a
genuine restart from step 0 under a new label, not a continuation — worth confirming
deliberately before starting data prep, not discovering after.
