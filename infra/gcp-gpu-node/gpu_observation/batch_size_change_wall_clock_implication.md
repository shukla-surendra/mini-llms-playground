# Why "more tokens/sec" didn't mean "finishes sooner" — the step-target trap

A real correction made mid-session, worth documenting precisely because it's a
genuine, non-obvious gotcha in this codebase, not a mistake to just quietly fix.

## The setup

Switched `custom-gpt-50m`'s GCP run from `batch_size=1` (steps/sec ~35, MFU 9.1%) to
`batch_size=4, grad_accum=8` (steps/sec ~13.5, MFU 14.1%) — same effective batch
(32), matching the already-benchmarked, more-efficient config from
`training_sop.md`. The efficiency gain is real: **+55% tokens/sec (35,600 →
55,300)**. The mistake was assuming that automatically meant "finishes sooner" — it
doesn't, and the reason is specific to how this trainer defines its stopping point.

## The mechanism

`trainer.py`'s loop: `for step in range(start_step, train_cfg.steps)` — training
stops purely on **step count** (default target 1,000,000), never on token count
directly. But the actual data processed per step is `batch_size × context_length`,
so the *effective* total-token budget is `train_cfg.steps × batch_size × ctx_len` —
**a quantity that changes when batch_size changes, even though `train_cfg.steps`
(the literal stopping condition) does not.**

## The real numbers, computed at the moment of the switch (step 371,530)

| | Wall-clock to reach step 1,000,000 | Tokens seen by then |
|---|---|---|
| Staying at batch=1 (~35 steps/sec) | ~5.0 hours | 1.024B (the originally-printed "budget") |
| Switched to batch=4 (~13.5 steps/sec) | ~12.9 hours | 4.096B (4x more) |

**Reaching the same step-number finish line now takes ~2.6x longer in wall-clock**,
not less — because each step now represents 4x more real work, and the ~55%
per-token efficiency gain doesn't come close to offsetting a 4x increase in
per-step workload.

## What "+55% tokens/sec" actually answers, vs. what it doesn't

- **Correct claim**: for a *fixed number of hours* of rental, batch=4 trains the
  model on more real data than batch=1 would in the same hours. This is genuinely
  true and was the original, real motivation for the change.
- **Incorrect implied claim** (the mistake): that this also means the run's own
  defined finish line (step 1,000,000) arrives sooner. It doesn't — it arrives
  later, because that finish line now requires more total data to reach.

These are two different questions ("how much does it learn per hour" vs. "when does
it stop") that this codebase's `steps`-based stopping condition quietly couples
together whenever batch size changes mid-run.

## The actual choice this creates

1. **Leave `GPT_STEPS` at 1,000,000** — accept ~12.9 more hours, get 4.10B total
   tokens by completion instead of the originally-planned 1.024B (a real, deliberate
   scope change, not free — but consistent with this project's own "excess data is
   neutral, not harmful" finding elsewhere in `docs/TRAINING_QA.md`-equivalent
   reasoning, if the extra GPU-hours are acceptable).
2. **Lower `GPT_STEPS`** to preserve the *original* 1.024B-token target — set it to
   roughly `current_step + remaining_steps_at_new_batch / 4` — and the run both
   finishes faster in wall-clock *and* keeps the originally-intended data budget.

Neither option is "wrong" — the point of documenting this is that the choice has to
be made deliberately, not discovered after the fact by watching the ETA field climb
instead of fall right after what looked like a pure speed improvement.
