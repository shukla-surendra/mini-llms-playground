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
