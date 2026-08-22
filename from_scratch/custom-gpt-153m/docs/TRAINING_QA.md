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

## How do I read the live training monitor output — what does each field mean, what should I actually watch, and what's a "good" final value?

A real line from this project's own `tqdm` progress bar, captured mid-run:

```
training: 73%|███████▎| 93535/127933 [3:26:52<1:31:53, 6.24step/s,
batch_loss=4.4065, epoch1_eta_h=117.2, est_epoch=0.131, eta_h=6.5,
lr=1.03e-04, test_loss=4.7551, test_ppl=116.2, total_h=17.64, train_loss=4.6036]
```

Every field, traced to the exact line in `src/gpt/training/trainer.py` that produces it:

| Field | What it is | Updated | Notes |
|---|---|---|---|
| `batch_loss` | Cross-entropy loss of the **single micro-batch** just processed | every step | The noisiest number on the line — one `context_length`-token window, `batch_size=1`. Expect it to bounce around a lot; it is not the trend signal. |
| `train_loss` / `test_loss` | Cross-entropy loss **averaged over `eval_batches` random windows**, from `train_tokens`/`test_tokens` respectively | every `eval_interval` steps (held constant on the postfix between evals) | `estimate_loss()` (`trainer.py:138`) — the actual quality signal. `test_loss` is what `best_test_loss`/`checkpoints/153m/best.pt` gate on. |
| `test_ppl` | `e ** test_loss` — see [Chapter 29 — Perplexity](../../../docs/llm-engineering/29_perplexity_understanding_and_interpreting_it.md) for the full mechanism | same cadence as `test_loss` | Same information as `test_loss`, different scale — read together, not as two separate signals. |
| `lr` | Current optimizer learning rate, from the warmup→cosine-decay schedule | every optimizer update | See [`TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md) — where you are on this curve changes how to interpret every other field. |
| `est_epoch` | `processed_tokens / len(train_tokens)` — fraction of one full pass over the training corpus | every step | An estimate, not an exact count — see [Chapter 15](../../../docs/llm-engineering/15_evaluating_a_model_while_training.md#what-an-epoch-is-and-why-its-an-approximation-here) for why random-window sampling makes this approximate. |
| `total_h` | Wall-clock GPU-hours elapsed so far this run | every step | Plain elapsed time, resume-aware (`elapsed()` accounts for prior sessions). |
| `eta_h` | Hours remaining **to finish the configured `steps` budget**, at the current steps/hour rate | every step | This is the number that answers "when does this specific run stop." |
| `epoch1_eta_h` | Hours remaining **to reach one full epoch's worth of tokens processed**, at the current token rate | every step | A different milestone than `eta_h` — "when would this hit 1.0 epochs if it kept going," independent of the configured step budget. Don't confuse the two ETAs; they answer different questions. |

### Why test data exists at all — the mechanism, not just the name

`test_loss` isn't computed on data the model trained on. `prepare.py`'s `build_corpus()`
shuffles every source together once, then does a **90/10 split before any tokenization or
training happens** (`split_idx = int(len(all_conversations) * train_ratio)`,
`train_rows`/`test_rows`), writing separate `data/train.txt`/`data/test.txt` — the 10% test
slice is set aside once, up front, and no gradient update ever sees it. This is what makes
`test_loss` meaningful: `train_loss` tells you how well the model fits data it has
literally seen and adjusted its weights toward; `test_loss` tells you how well it predicts
data it has never been updated on — the actual proxy for "does this generalize," not just
"has this memorized its inputs." See [Chapter 4's train/test diagnostic table](../../../docs/llm-engineering/04_hyperparameter_tuning.md#using-train_loss-vs-test_loss-as-your-tuning-feedback-signal)
for how to read the *pair* together (healthy vs. overfitting vs. underfitting).

### What to actually watch, in order of signal quality

1. **Ignore `batch_loss` for trend-reading.** It's one window; it exists so you can see the
   loop is alive, not to judge progress by.
2. **Watch `test_loss`/`test_ppl` over multi-thousand-step windows, not point to point.**
   A single eval-to-eval tick is sampling noise from `eval_batches` random windows — not
   evidence of anything by itself. Concretely, in this exact run: an earlier reading at
   step ~76,000 showed `test_ppl` around 162–178; by step 93,535 it had dropped to `116.2` —
   then a later reading came back up to `121.5`. Read correctly, that's the same story as
   [Chapter 15](../../../docs/llm-engineering/15_evaluating_a_model_while_training.md)'s
   signal 2: `116.2 → 121.5` is a normal upward tick inside a real downward trend
   (`178 → 116.2` over that same stretch), not a reversal — `best_test_loss` (the running
   minimum, not the raw per-eval column) is the number that actually matters, precisely
   because it's immune to this kind of single-sample noise.
3. **Watch `lr` alongside the loss trend, not in isolation.** A stall while `lr` is still
   near its peak (as here — `1.03e-04` is early in a schedule with a much lower `min_lr`
   floor) is not evidence of convergence; most of the schedule's fine-tuning phase hasn't
   run yet. See [`TRAINING_SCHEDULE.md`'s three-question framework](TRAINING_SCHEDULE.md#is-a-longer-run-still-worth-it-a-three-question-framework).
4. **Watch `est_epoch`.** At `0.131`, the model has seen roughly 13% of the corpus once —
   nowhere near the regime where more data stops helping.

### What should the final value actually be?

There isn't a fixed target number to compare against — perplexity depends on corpus
difficulty, vocabulary size, and model capacity, so quoting a bare number without that
context would be misleading. The real stopping condition is the same three-question
framework already documented in
[`TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md#is-a-longer-run-still-worth-it-a-three-question-framework):
`best_test_loss` (not a single noisy reading) has genuinely stopped improving for a
sustained stretch, **and** `lr` has meaningfully left the near-peak region, **and**
`est_epoch` is no longer small. None of those three hold yet at step 93,535/127,933 — `lr`
is still elevated, `est_epoch` is `0.131`, and the `162 → 116.2` drop just two evaluations
apart shows the curve is still moving, not flattened. "Done" here means finishing this
run's configured step budget (`eta_h`) and re-reading these same three checks at that
point — not chasing a specific perplexity number in isolation.
