# Why is a rented GPU so much faster than my Mac? (plain-language version)

Same facts as `local_mps_vs_gcp_l4_same_config.md` and the other observation docs in
this folder — this one just explains them without jargon, using analogies. The real
numbers are still real; only the language is simplified.

## The core idea: training is millions of tiny, identical sums, done over and over

Training this model is, underneath everything, mostly one operation repeated an
enormous number of times: **multiply two numbers, add the result to a running
total.** That's it. A transformer's "attention" and "layers" are just that one
operation, done millions of times per second, over and over.

- A **CPU** is like a small team of brilliant generalists — great at complicated,
  branching decisions, but there's only a handful of them.
- A **GPU** is like an army of thousands of simple workers who are only good at one
  thing — multiply-and-add — but there are *thousands* of them working at once.

For "do the same simple math millions of times," the army wins by a landslide. That's
the whole reason GPUs exist for AI training.

## So why isn't your Mac's GPU already an "army"?

It is — just a smaller, more general-purpose one. Think of the difference like this:

- **Your Mac's GPU** is a handyman. It's built to be reasonably good at lots of
  different jobs — video playback, game graphics, some AI work — because a laptop
  has to do everything. It shares its workshop (memory) with the rest of the
  computer, and it doesn't have any tools built *specifically* for AI math.
- **The rented NVIDIA L4** is a specialist factory built for one purpose. It has
  **240 machines (called Tensor Cores) built to do exactly the multiply-and-add
  operation AI training needs, and nothing else** — much faster at that one job than
  a general-purpose tool would be. It also has its **own private, fast warehouse**
  (24GB of dedicated memory, never shared with anything else) instead of sharing
  supplies with the rest of the building the way your Mac's unified memory does.

Neither is "better" in general — a handyman is more useful for everyday laptop life.
But for the one specific job of "repeat this multiplication millions of times,"
the specialist factory wins, and it's not close.

## Why the gap gets even bigger with tiny batches

Imagine a delivery truck that has to stop and get a "go-ahead" from a checkpoint
guard before every single delivery. If you send one tiny package at a time (this is
what "`batch_size=1`" means — training on one example at a time), the truck spends
more time waiting at the checkpoint than actually driving. If you load up a bigger
truck with 4 packages per trip ("`batch_size=4`"), you check in with the guard 4x
less often for the same amount of cargo — much less time wasted waiting around.

Your Mac's "checkpoint guard" (the software layer that hands work to the GPU,
called Metal/MPS) is slower to say "go ahead" than the rented GPU's equivalent
(called CUDA — a system NVIDIA has been refining specifically for this exact job for
over 15 years). At small batch sizes, where you're constantly stopping at the
checkpoint, that difference matters most. That's exactly why, measured this
session, the gap was a real **~6x** at the smallest batch size — bigger than a
rougher earlier estimate of ~2.4x had suggested.

## What we actually measured, in plain terms

- Same model, same data, same batch size, on both machines: **the rented GPU trained
  about 6 times faster.**
- Switching the rented GPU to send bigger "truckloads" per trip (`batch=4` instead
  of `batch=1`) pushed real throughput up by another ~55%, landing at roughly
  **8x faster than your Mac overall**, once both the hardware advantage and the
  bigger-batch advantage are combined.
- The GPU itself still isn't "fully used" even now — real usage of its total power
  (a number called MFU) is only about 14%, because this particular model is small
  enough that even a "cheap" cloud GPU has more army than it can keep busy. That's
  not wasted money exactly — it's more like renting a whole factory floor when your
  order only fills part of it; the alternative (your Mac) doesn't have a factory at
  all.

## Should you always use the rented GPU, then?

For anything that takes more than an hour or two, yes — clearly. For a quick
5-minute experiment, your Mac is free and good enough. The real tradeoffs:

| | Your Mac | Rented GPU |
|---|---|---|
| Cost per hour | $0 | ~$0.70 (this specific GPU) |
| Speed | Baseline | ~6-8x faster, measured |
| Ties up your laptop | Yes — needs to stay awake, plugged in | No — runs independently, you can close your laptop |
| Setup effort | None, just run it | Real one-time effort (renting the machine, moving data over) |

For this actual training run: the whole remaining ~13 hours on the rented GPU costs
under $10. Running the same amount of real progress on your Mac would take days and
keep your laptop unusable for other work that whole time. That's the real-world
"why bother renting a GPU" answer — it's not really about the GPU being fancy, it's
about **time being worth more than $0.70/hour** once a task is big enough.
