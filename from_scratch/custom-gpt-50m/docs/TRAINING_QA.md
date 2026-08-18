# Training Q&A

A running log of specific questions asked while training this project's model, each
answered against this project's actual code and numbers rather than generic advice.
Where the general mechanism is already covered elsewhere, the entry links out instead of
repeating it — see [`TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md) for the step-count/LR
mechanism and the [LLM Engineering Curriculum](../../../docs/llm-engineering/00_roadmap.md)
for first-principles treatments.

## Does training for multiple epochs help?

Yes, up to a point — more passes over a fixed corpus keeps teaching the model something
*until* train loss keeps falling while test loss plateaus or rises (overfitting), which
is why `logs/train_eval_history_10m.csv` tracks `test_loss`/`best_test_loss` separately
from training loss in the first place.

The more useful question for this project specifically is *how many* epochs the
configured run actually represents, and whether that's enough — not whether epochs help
in the abstract. `TrainConfig.steps = 1_000_000` at `batch_size=1`,
`context_length=512` processes `512,000,000` tokens total; against this corpus's
`173,706,682` train tokens, that's **≈2.95 epochs** for the full configured run. See
[`TRAINING_SCHEDULE.md`'s three-question framework](TRAINING_SCHEDULE.md#is-a-longer-run-still-worth-it-a-three-question-framework)
for how to judge, from the live log, whether that budget is enough or a plateau is real
rather than an artifact of where you are on the LR schedule — and
[Chapter 15 — Evaluating a Model While It's Still Training](../../../docs/llm-engineering/15_evaluating_a_model_while_training.md)
for the general four-signal stopping-rule version of the same idea.

## Does arranging data in a particular way increase model performance?

Depends on *what* is being arranged — checked against this project's actual data
pipeline (`src/gpt/data/prepare.py`, `src/gpt/data/dataset.py`), not generic advice:

- **Storage order in `train.txt`: no effect.** `get_batch()` draws a uniformly random
  `context_length`-token window every single training step
  (`ix = torch.randint(0, max_start, (batch_size,))`), not a sequential walk through the
  file. Whatever order conversations physically sit in on disk is invisible to training —
  reordering the file changes nothing.
- **Shuffling the five sources together before writing: matters, and this project
  already does it correctly.** `build_corpus()` calls `random.shuffle(all_conversations)`
  across the full UltraChat + OASST1 + Dolly + SmolTalk + LMSYS pool *before* the
  train/test split, so no single source dominates any stretch of the corpus. This is the
  "joint/shuffled" arrangement
  [Chapter 28](../../../docs/llm-engineering/28_catastrophic_forgetting_and_continual_training.md)
  describes as avoiding catastrophic forgetting — the alternative (all of UltraChat, then
  all of OASST1, ...) would have real consequences even under random-window sampling,
  since which source a window happens to land in would no longer be independent of
  position in the file.
- **Conversation-boundary marking: a real, currently weak spot.** Conversations are
  joined with a plain `"\n\n"` (`build_corpus`'s `"\n\n".join(turns_to_text(t) for t in
  train_rows)`) — not GPT-2's own `<|endoftext|>` special token, even though
  `encode_raw`'s `tokenizer.encode(text, disallowed_special=())` already permits special
  tokens to pass through. Because windows are randomly sampled, a meaningful fraction of
  every batch straddles two completely unrelated conversations with only a blank line
  between them — a far weaker "this is unrelated, start fresh" signal than the token
  GPT-2's own pretraining already primed the embedding space to recognize. This is the
  one arrangement choice here that's plausibly costing real quality, since it directly
  affects what every random window's next-token targets actually mean at the seam.

**Net**: reordering the file → no effect; mixing sources before writing → already
correct; how document boundaries are marked → the one lever worth revisiting.

## What does train/test loss actually represent, and what should I expect at 100%?

Answered against this project's own numbers (`logs/train_eval_history_50m.csv`), not
generic advice — the mechanism first, then what it means for *this* run specifically.

**Mechanism**: at every token position, the model outputs a probability distribution
over all `vocab_size=50,257` tokens. Cross-entropy loss = `-log(p)`, where `p` is the
probability the model assigned to whatever token *actually* came next in the real text.
It's a direct measure of how surprised the model was, averaged over every token in the
eval set — nothing more. Working it backward:

- **Step 0** (untrained): `train_loss=10.96, test_loss=10.97` — matches the theoretical
  random-guessing floor, `ln(50,257) ≈ 10.82`, almost exactly. `p ≈ 1/50,257`.
- **Current** (~step 838K): `test_loss ≈ 2.73` → `p = e^-2.73 ≈ 0.065`. The model now
  assigns ~6.5% probability, on average, to the actual next token.
- **Perplexity** (`e^loss`) turns this into an intuitive number: "as if choosing
  uniformly among N options." Perplexity 15.3 (current) vs. 50,257 (random) vs. 1
  (perfect). Going from ~50,257-way to ~15-way is a real, large improvement — most of it
  happened in roughly the first half of training (see below), which is the normal shape
  of these curves, not a sign anything's off.

**Why loss doesn't track factual correctness well**: the average is over *every* token,
and most tokens (spaces, punctuation, "the"/"a"/"of", the predictable back-half of a
word already started by BPE) are easy and get near-zero loss quickly — they dominate
the token count. The tokens that actually encode a *fact* ("Paris" vs. any other city
right after "The capital of France is") are a small minority by count, so a model can
drive the aggregate way down by mastering grammar/formatting while still being wrong on
the comparatively rare fact-bearing tokens without moving the average much. This is
directly why `reports/qa_report_*.html` keeps showing fluent, well-formatted,
confidently *wrong* answers even as loss keeps improving — the metric was never
measuring "is this true," only "was the model surprised."

**A real trap this project's own log fell into**: `data/train.bin`/`test.bin` was
rebuilt on 2026-08-16 between step 482,400 and 483,200 (confirmed: file mtime lines up
with the eval-log timestamp gap exactly). `best_test_loss` (2.567175) was recorded at
482,400 — right before the swap — so every eval since has been compared against a
*different* test set than the one that produced that record. Chasing "beat the
recorded best" past that point is chasing a number measured on a benchmark that no
longer exists; it looked like a 300K+-step plateau but was a measurement artifact from
comparing across the swap, not the model failing to improve. **The trend that actually
means something is a windowed average computed entirely *after* the swap** (all evals
on the same, current test set):

```
step 482,400-561,600   test_avg = 2.8604
step 562,400-641,600   test_avg = 2.8143
step 642,400-721,600   test_avg = 2.7517
step 722,400-801,600   test_avg = 2.7276
```

**Extrapolating to 100% (step 1,000,000)**: the size of each window's improvement is
visibly shrinking (Δ -0.046, -0.063, -0.024) — the expected diminishing-returns shape
for an LM loss curve, not a red flag. With 162,400 steps left as of the last window
above, a reasonable estimate is **test loss ≈ 2.60-2.70** at completion (perplexity
≈ 13.5-14.9) — a real but modest further narrowing, not a second wave of the early
gains. If a bigger quality jump is wanted after this run finishes, the more effective
lever is changing *what* the model trains on next (see "Does arranging data..." above)
— specifically, a short continuation phase on a chat-only rebuild of the corpus
(`make data-public` without `--extra-jsonl`, dropping Wikipedia/Books/Repo-domain) to
counteract the extra-document dilution that same section documents, rather than
expecting the final stretch of this same run to still be full of gains. Not yet built —
this project has no separate fine-tuning phase today, see `Makefile`/CLI for what
actually exists before assuming otherwise.

## How do we decide how many tokens a given model size needs?

**In one sentence, no jargon**: think of parameters as a student's number of brain
cells (capacity) and tokens as pages of study material (data). A big brain given too
few pages wastes its own potential — it had room to learn far more than it was shown.
A small brain buried in an enormous pile of pages can only hold so much — past a
point, more pages stop helping, because the brain itself is the bottleneck, not the
supply of material. The question this answers is: **for a given brain size, how many
pages is the sweet spot** — enough to fill its capacity, not so many that the excess
is wasted on a brain too small to use it?

**The actual rule — Chinchilla scaling**: for a *fixed compute budget*, loss is
minimized at roughly **20 tokens of training data per model parameter**. This isn't a
guess — it comes from the paper *"Training Compute-Optimal Large Language Models"*
(Hoffmann et al., 2022, the "Chinchilla" paper), which trained hundreds of
model-size/data-size combinations at matched compute cost and fit a curve to where
loss came out lowest. `20:1` is where that curve bottoms out.

**Why 20:1 specifically, mechanically**: training compute cost ≈ `6 × N × D` FLOPs
(`N` = params, `D` = tokens — the same "6N" approximation `benchmark.py`'s own
`flops_per_token()` uses for MFU calculations elsewhere in this project). For a fixed
FLOP budget, every FLOP spent making the model bigger (`N`) is a FLOP not spent
showing it more data (`D`), and vice versa. Too far toward big-`N`/small-`D`: the
model has unused capacity — Chinchilla's actual finding was that many contemporary
(2022-era) large models were *undertrained* relative to their size, i.e. off this
ratio in that direction. Too far toward small-`N`/huge-`D`: the model saturates and
extra tokens buy less and less — the same decelerating-returns shape measured
directly on this project's own loss curve (see the entry above) shows up here too,
just as a function of data instead of steps.

**This project's own numbers against that ratio** (`params` are each model's *exact*
count, not rounded — via `gpt-config`/`resolve_model_config`):

| Model | Params | 20x (Chinchilla-optimal) | 16x (this project's actual convention) |
|---|---|---|---|
| 50m | 51,475,968 | 1.03B tokens | 0.82B tokens |
| 153m | 152,791,296 | 3.06B tokens | 2.46B tokens (the real configured run — see `custom-gpt-153m/docs/GPU_TRAINING.md`) |
| 350m (E=1024,L=24, hypothetical) | 354,823,168 | 7.10B tokens | 5.68B tokens |

**Why 16x, not 20x**: this project deliberately runs a bit under Chinchilla-optimal on
every size, not by accident — `GPU_TRAINING.md`'s own reasoning for 153m is "a little
under Chinchilla-optimal... to leave headroom inside 24h." Trading a slightly
sub-optimal data ratio for a training budget that actually fits an available GPU-hour
or dollar budget is a normal, deliberate engineering call, not a mistake — 16x is
still close enough to the 20x optimum that the loss cost is small, per the same
"repetition/data trade-offs cost little near the optimum, get expensive far from it"
shape as the epoch-count guidance elsewhere in this doc.

**The actual decision procedure, either direction**:
1. **Model size is fixed, need the token target**: `tokens_needed ≈ 20 × params` (or
   16x if trading a little optimality for a tighter compute/cost budget on purpose).
2. **Token supply is fixed, need the model size it supports**: `max_params ≈
   tokens_available / 16-20`. This is exactly what the sibling `custom-gpt-153m`
   project's [`../../custom-gpt-153m/DATASET.md`](../../custom-gpt-153m/DATASET.md)
   `Fresh tokens collected -> Do this` lookup table encodes directly (that table lives
   there, not in this project's own `DATASET.md`, which covers a different
   question — chat-vs-extra-document composition, see "Does arranging data..." above),
   and what `gpt-benchmark`'s own `--sweep-batch` output's "what fits in a fixed
   budget" table computes live for whatever GPU-hours are actually available.

**One nuance worth knowing before treating 20:1 as a universal law**: it minimizes
*training* compute for a target loss — it says nothing about *inference* cost. A
model that will be served/run a huge number of times afterward is often worth
training on tokens *well past* 20:1 on purpose, deliberately accepting less
training-compute efficiency for a smaller, cheaper-to-run model at a given capability
level (the reasoning behind why models like Llama were trained on token counts
substantially beyond their own Chinchilla-optimal point). Not directly relevant to
this project's current runs (all sitting near 16-20x already), but worth knowing
20:1 answers one specific question — minimize training compute — not "the" universally
correct ratio for every goal.

**Units, precisely — the two easy ways to misread "20x"**:
1. **It's tokens, not raw text size.** `20 × params` gives a *token* count, not a byte
   count — BPE tokens average roughly 4 characters each in English, so the
   corresponding raw-text size is several times larger than the token number itself.
   Confusing the two would make a real corpus look ~4x smaller than it needs to be, or
   ~4x more sufficient than it actually is, depending on which direction the mistake
   runs.
2. **It's parameter count, not checkpoint file size.** A model's `.pt` file is bigger
   than its raw parameter count would suggest — `checkpoint.py` saves
   `model_state_dict` (params, 4 bytes each in fp32) *plus* `optimizer_state_dict`
   (AdamW's two per-parameter moment buffers, another 8 bytes/param), so a 51.5M-param
   model's checkpoint is well over 500MB on disk despite `51.5M × 4 bytes ≈ 206MB` of
   parameters alone. Scaling the token target off checkpoint-file-size instead of the
   actual parameter count (`gpt-config`'s reported number, not `ls -la` on the `.pt`)
   would overshoot by roughly 3x.

So, precisely: `tokens_needed ≈ 20 × param_count` — param count as `gpt-config` reports
it, tokens as the tokenizer counts them (`.bin.json`'s `"tokens"` field) — not a loose
"20x more data" read against file sizes on disk.

**What actually happens if the available corpus has *more* than the 16-20x target —
is that a problem?** No. Worked example: the sibling `custom-gpt-153m` project's
pretraining corpus (`1,185,172,323` tokens, per its own `data/train.bin.json` +
`data/test.bin.json`) is `1.15x` this model's 20x-Chinchilla target and `1.44x` its
16x target — genuinely more than 50m needs, not a rounding coincidence.

The mechanism that makes excess harmless: training runs for a **fixed step budget**
(`steps × batch_size × ctx_len`, unrelated to how big the corpus on disk happens to
be), and `get_batch()` draws random windows from wherever the corpus is — it never
"has to" consume the whole file. A bigger-than-needed corpus just means the run
completes *less than one full pass* over it, which is strictly fine; there's no
mechanism by which unread tokens sitting on disk affect the model. **Excess data is
neutral surplus, not harmful** — it doesn't get "seen too much" (that's the opposite
problem, too little data forcing many repeats) and it doesn't confuse or dilute
anything just by existing unused.

**If anything, excess capacity is the favorable direction to be in**, not a neutral
one: this project's own real corpus (0.278B unique train tokens, `~3.68` epochs to
hit its 1.024B-token budget) is on the *other* side of 16-20x, relying on repetition
to make up the gap. Repetition is the thing worth watching (this doc's "Does training
for multiple epochs help?" entry above: cheap up to ~4 epochs, decaying toward
worthless by ~16) — having more unique data than the ratio calls for is protective
against ever needing to lean on repetition that hard, not a risk of its own.

**The one real caveat**: this is about tokens *sitting unused in reserve*, not tokens
*actually mixed into training*. Excess quantity from a mismatched source is harmless
only as long as it stays unused — actually mixing a much larger, differently-shaped
corpus in (raw pretraining prose into a chat-trained model's data, the scenario this
sizing check was originally run against) reintroduces the *content-composition*
problem this doc's "Does arranging data..." entry covers, which is a completely
separate question from whether there's *enough* of it. Size being more than
sufficient never overrides format being wrong for the job.
