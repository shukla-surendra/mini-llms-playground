# Distributed Training: DDP, FSDP, and the Parallelism Landscape

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 2B — Training at Scale (see
[Chapter 25](25_efficient_attention_flash_and_sdpa.md)'s header for why this and that
chapter are appended after the original numbered catalog). Builds on
[Chapter 13](13_the_training_loop_mechanism_by_mechanism.md)'s training loop — this
chapter is about running that same loop across *multiple devices at once*, and the
different strategies for doing so.

## In Plain English

A model and its training data can outgrow what a single GPU (or machine) can hold or
compute in reasonable time. Distributed training splits the work across multiple
devices — but "splits the work" can mean genuinely different things: give every device a
full copy of the model and split the *data* (data parallelism), or split the *model itself*
across devices so no single device ever needs the whole thing in memory (model/sharded
parallelism). DDP is the classic instance of the first; FSDP is a specific, popular
instance of the second.

## The First-Principles Explanation

### Data parallelism (DDP): same model everywhere, different data everywhere

Every device (rank) holds a complete replica of the model. Each step, every rank processes
a different mini-batch, computes its own gradients locally, and then all ranks
**all-reduce** (average) their gradients before applying the optimizer step — so every
replica ends up applying the identical update and stays in sync, without ever directly
exchanging model weights after the initial broadcast. The cost: memory-wise, this is the
*most* expensive strategy per device, since every single device needs enough memory to
hold the full model, its full gradients, and its full optimizer state — DDP doesn't help
at all if the model itself is the thing that doesn't fit.

### Sharded parallelism (FSDP): no device ever holds the whole model

Fully Sharded Data Parallel takes a different approach: parameters, gradients, and
optimizer state are all split (sharded) across ranks from the start — each rank
permanently holds only `1/world_size` of each. When a given layer's forward or backward
pass actually needs its full, unsharded parameters, FSDP performs an **all-gather**
(temporarily reassembling the full parameter tensor from every rank's shard), uses it, then
immediately discards everything except this rank's own shard again. This is strictly more
communication than DDP (an all-gather per layer, not one all-reduce per step) — the
trade FSDP makes is more network traffic in exchange for a peak per-device memory
footprint that doesn't scale with the model's full size, which is what makes training
models too large for any single device's memory possible at all.

### Where these fit in the broader parallelism landscape

DDP and FSDP are both **data**-parallel strategies at their core (different ranks process
different data) — the distinction between them is *how much of the model* each rank
carries. Two other real strategies split the *model itself* differently, worth knowing by
name even without an implementation in this project:

- **Tensor parallelism**: split individual large operations (e.g., one big matrix
  multiply) across devices, each computing a slice of the same operation in parallel —
  requires fast interconnect (like NVLink) since it communicates *within* every forward
  pass, not just once per step.
- **Pipeline parallelism**: assign different *layers* to different devices (device 0 gets
  layers 1-10, device 1 gets layers 11-20, etc.), passing activations between them as data
  flows through the model — trades some device idle time (a "pipeline bubble" while later
  stages wait for earlier ones) for lower communication *volume* than tensor parallelism.

Real large-model training often combines several of these simultaneously (data + tensor +
pipeline parallelism together) — each addressing a different bottleneck (data throughput,
single-operation size, total model depth) rather than one strategy being strictly better
than another.

## Grounded in This Repo's Code

[`from_scratch/tinystories-gpt-6m/train_ddp.py`](../../from_scratch/tinystories-gpt-6m/train_ddp.py)
and
[`train_fsdp.py`](../../from_scratch/tinystories-gpt-6m/train_fsdp.py)
implement both strategies on the exact same model and data this project's `train.py`
already trains — see
[`docs/DISTRIBUTED_TRAINING.md`](../../from_scratch/tinystories-gpt-6m/docs/DISTRIBUTED_TRAINING.md)
for two real, concrete pieces of evidence that the mechanisms are actually doing what they
claim: a genuine `NotImplementedError`/`AttributeError` this project hit trying to run
`torchrun` and (separately) plain `FSDP(model)` on this specific hardware, and a real
side-by-side run showing DDP's `num_parameters()` reporting the full model per rank
against FSDP's reporting exactly half.

## Deep-Dive: Why All-Reduce Keeps DDP's Replicas In Sync Without Ever Re-Syncing Weights Directly

A question worth sitting with: DDP broadcasts weights once, at construction — how do the
replicas not drift apart over hundreds of training steps if weights are never
re-synchronized directly again? Because **all-reduce averages gradients, not weights**,
every single step, before the optimizer step is applied. If every rank starts a step with
identical weights, and every rank then applies the *same* (averaged) gradient update via
the *same* optimizer, every rank necessarily ends the step with identical weights again —
by induction, this holds for every step of training, so the replicas can never drift as
long as the gradient averaging genuinely happens before every single optimizer step. This
is also exactly why DDP requires the gradient all-reduce to be synchronous (block until
every rank has contributed) rather than approximate or best-effort — any rank silently
missing the sync even once would permanently desynchronize that replica from the others.

## Try It Yourself

- Run `python train_ddp.py` and `python train_fsdp.py` in this project back to back (same
  `STEPS`/`BATCH_SIZE`), and compare each run's printed `[model] N parameters` line — a
  direct, concrete confirmation of full-replication vs. sharding, not just a claim.
- Read `docs/DISTRIBUTED_TRAINING.md`'s two "real bug" sections in full — both are genuine
  environment-specific issues (a DNS-resolution hang, an MPS-accelerator misdetection)
  encountered building this project, a real instance of the gap between "the API looks
  simple in a tutorial" and "getting it running on the hardware actually in front of you."

## Common Misconceptions

- **"FSDP is just a faster version of DDP."** It's not faster in general — it's a
  different memory/communication trade, usually *slower* per step (more communication)
  specifically in exchange for fitting models DDP couldn't hold at all. Choosing between
  them is about whether the model fits in per-device memory under DDP, not raw speed.
- **"More distributed processes always means faster training."** Only true when the
  hardware and interconnect can actually support the added communication cheaply (real
  GPUs with fast interconnect) — see this project's own numbers in
  `DISTRIBUTED_TRAINING.md`, where adding CPU processes over `gloo` was slightly *slower*
  than single-process training, for a model this small.
- **"Data parallelism and model parallelism are competing choices — pick one."** Real
  large-scale training frequently combines multiple strategies (data + tensor + pipeline)
  at once, each solving a different bottleneck simultaneously.

## Practice Questions

1. Explain precisely why DDP's replicas stay synchronized for an entire training run
   despite weights only being broadcast once, at construction time.
2. A model's optimizer state alone is larger than a single GPU's memory. Would DDP alone
   solve this problem? Would FSDP? Explain the mechanism-level reason for each answer.
3. Why does FSDP typically involve more network communication than DDP, and under what
   circumstances would you accept that cost anyway?

## Key Terms

- **All-reduce**: a collective operation combining (e.g., averaging) a tensor across every
  rank in a process group, with every rank ending up with the same combined result — the
  mechanism DDP uses to keep gradients (and therefore weights) synchronized.
- **All-gather**: a collective operation where every rank contributes its own shard and
  every rank receives the full, reassembled tensor — the mechanism FSDP uses to
  temporarily reconstruct full parameters before a layer needs them.
- **Sharding**: splitting a tensor (parameters, gradients, or optimizer state) into pieces
  distributed across ranks, so no single rank permanently holds the whole thing.
- **Tensor / pipeline parallelism**: model-parallel strategies splitting, respectively, a
  single large operation or entire layers across devices — distinct from FSDP's
  parameter-sharding approach, though real systems often combine several strategies.
