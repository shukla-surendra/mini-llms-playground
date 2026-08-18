# What are batch_size and grad_accum_steps? (plain-language version)

## The core idea

Training = show the model some examples → measure how wrong it was → nudge its
internal numbers to be a little less wrong → repeat, millions of times.

**A "batch" is just: how many examples get looked at before calculating one nudge.**

## Why not nudge after every single example, or after the whole dataset?

- **One example at a time**: cheap per step, but each nudge is based on a single
  example's opinion — jittery, easily thrown off by one weird/unusual example.
- **The entire dataset at once**: very stable nudges, but you'd need to hold billions
  of tokens in memory simultaneously and would only get to nudge the model once
  after seeing literally everything. Not physically possible.

The practical middle ground: gather a handful of examples (a **batch**), average
their opinions, nudge once. **`batch_size` = how many examples are in that handful.**

## What gradient accumulation actually solves

Say the goal is a nudge based on **32 examples'** averaged opinion (stable, reliable)
— but the GPU can only physically hold **4 examples** in memory at once.

**The workaround**: process 4 examples, note their opinion but *don't nudge yet* —
keep a running tally. Process another 4, add to the tally. Repeat **8 times** (4×8=32),
*then* actually nudge the model using the full 32-example tally.

**`grad_accum_steps` = how many of these silent "tally, don't act yet" rounds happen
before one real nudge.**

## The change made this session

| | `batch_size` | `grad_accum_steps` | Effective vote size (`batch × accum`) |
|---|---|---|---|
| Before | 1 | 32 | 1 × 32 = **32** |
| After | 4 | 8 | 4 × 8 = **32** |

**The vote size never changed — the model learns identically, nudge for nudge.**
Only *how the work is physically chunked* changed: fewer, bigger handoffs to the GPU
instead of more, smaller ones.

## Why identical learning + different chunking = faster

Every handoff of work to the GPU carries a small fixed setup/dispatch cost before the
real math starts. Chunking as 1-at-a-time means paying that setup cost 32 separate
times to build one nudge. Chunking as 4-at-a-time pays it only 8 times for the exact
same nudge — and the GPU's thousands of parallel cores also have more real work per
handoff to chew on simultaneously, instead of mostly sitting idle waiting on the next
tiny delivery. See `observation_2026-08-18_vram_bandwidth_cores.md` for the measured
effect: MFU (real compute efficiency) went from 9.1% to 14.1% from this change alone.

## One-line summary

`batch_size` = how many examples per physical GPU step.
`grad_accum_steps` = how many physical steps get silently tallied before one real
model update. Their product is what actually determines how "informed" each nudge
is — everything else is just about how efficiently that work gets organized on the
hardware.
