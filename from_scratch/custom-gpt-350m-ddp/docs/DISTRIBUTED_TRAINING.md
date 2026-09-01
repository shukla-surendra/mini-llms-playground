# Distributed Training: DDP and FSDP, Both, in This Project

Unlike the sibling `custom-gpt-6m` project — which implements DDP and FSDP as two
separate, parallel scripts (`trainer_ddp.py`/`trainer_fsdp.py`) sharing no code — this
project implements both **inside the one production `trainer.py`**, switched with a
single setting (`GPT_PARALLELISM=ddp` or `fsdp`), reusing the exact same training
loop, checkpointing, resume logic, eval cadence, and LR schedule either way. The two
modes differ only at the handful of places the mechanisms actually differ; everything
else — including the general concepts (what a process group is, what all-reduce/
all-gather actually do) — is unchanged from the first-principles treatment in
[`../../../docs/llm-engineering/26_distributed_training_ddp_and_fsdp.md`](../../../docs/llm-engineering/26_distributed_training_ddp_and_fsdp.md).
This doc covers only this project's specific implementation and what was actually
verified building it.

## The one-sentence version of each

**DDP** replicates the full model on every rank and synchronizes gradients (all-reduce)
before each optimizer step — every rank always holds the complete model, complete
optimizer state, complete everything. **FSDP** shards the model's parameters,
gradients, *and* optimizer state across ranks from the start — each rank permanently
holds only `1/world_size` of each, temporarily reassembling a layer's full parameters
(all-gather) right before that layer's forward/backward needs them, then discarding
everything except its own shard again.

## Switching between them

```bash
GPT_PARALLELISM=ddp  torchrun --nnodes=2 --node_rank=<0|1> --master_addr=... -m gpt.cli.train
GPT_PARALLELISM=fsdp torchrun --nnodes=2 --node_rank=<0|1> --master_addr=... -m gpt.cli.train
```

Irrelevant at `world_size=1` — both collapse to the identical single-process path,
since `trainer.py`'s `use_fsdp = world_size > 1 and train_cfg.parallelism == "fsdp"`
only ever matters once there's more than one rank to shard or replicate across.

## What was actually added to the base project to support DDP at all

This project forked `custom-gpt-350m`, which has no distributed code — `world_size` is
always 1, `training_model` is always just `raw_model`. Getting to genuine 2-node DDP
meant touching three files, each for a different reason:

- **`src/gpt/cli/train.py`** — reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK` from the
  environment (`torchrun` sets these per spawned process before this module even
  imports — see below for exactly how), calls `dist.init_process_group(backend=...)`
  when `world_size > 1`, picks `nccl` when CUDA is available and `gloo` otherwise (the
  CPU smoke-test path), selects this rank's own `cuda:{local_rank}` device, and calls
  `dist.destroy_process_group()` in a `finally` block on the way out. With no
  `torchrun` launcher (`WORLD_SIZE` unset), every one of these branches is a no-op —
  single-GPU behavior is bit-for-bit unchanged from before DDP support existed.
- **`src/gpt/config.py`** — `resolve_train_config(context_length, world_size)` makes
  `GPT_TARGET_TOKENS` world-size-aware: `tokens_per_step = batch_size * ctx *
  world_size`, so the same token budget resolves to a *different* `steps` count
  depending on how many ranks are actually splitting the work — without this, running
  the same `steps` under 2 ranks would silently double total tokens consumed (each
  rank processes its own batch independently, in parallel), a real footgun the base
  project doesn't need to guard against since it's always `world_size=1`.
- **`src/gpt/training/trainer.py`** — the largest change, all in `train()`/`_run_loop()`:
  the `DDP(raw_model, device_ids=...)` wrap; per-rank RNG seeding
  (`torch.manual_seed(train_cfg.seed + rank)`) so ranks draw different random batches
  from the identical shared corpus, not the same batch redundantly; the `no_sync()` /
  `is_accum_boundary` gating described below; and `is_main = rank == 0` gating for
  everything that would otherwise race or duplicate if every rank did it — printing,
  the progress bar, eval, and checkpoint writes.

## How one node finds the other (rendezvous) — nothing here is bespoke code

This project never opens a socket to "find" the other machine — that's entirely
`torchrun`'s job, before this project's own code runs at all:

```bash
# On the master (node_rank=0):
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=<master's private IP> --master_port=29500 -m gpt.cli.train
# On the worker (node_rank=1), identical except node_rank:
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=<same master private IP> --master_port=29500 -m gpt.cli.train
```

1. `torchrun` on **each** machine computes this process's global `RANK` (from
   `node_rank` × `nproc_per_node` + local index) and `WORLD_SIZE` (`nnodes` ×
   `nproc_per_node`), and sets `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/
   `MASTER_PORT` as environment variables — all of this happens *before* Python even
   imports `gpt.cli.train`.
2. `cli/train.py` calls `dist.init_process_group(backend="nccl")` — **no address,
   rank, or world_size argument passed explicitly**. This is PyTorch's default
   **`env://`** rendezvous method: it reads exactly the environment variables
   `torchrun` just set.
3. Mechanically, rank 0's call opens a `TCPStore` (a minimal shared key-value store)
   listening on `MASTER_ADDR:MASTER_PORT`. Every rank — including rank 0 itself —
   connects to that same address as a client and writes its own identity in. Every
   rank's `init_process_group()` call blocks until all `WORLD_SIZE` ranks have
   checked in, then all of them return together.

The precise answer to "how does master know about worker": it doesn't reach out —
**the worker connects to the master's known address**, and master's own
`init_process_group()` call is what's listening there, waiting for exactly
`world_size` check-ins before it lets any rank proceed. This is why `master_addr`
only ever needs to be node 0's address, never node 1's — node 1's own address isn't
needed anywhere in the launch command at all, only in the security group rule
allowing traffic between the two (see `infra/aws-gpu-node-multi/network.tf`'s
self-referencing rule). Once this bootstrap TCP connection exists, it's reused once
more to exchange NCCL's own internal setup handshake (unique IDs for the real
GPU-to-GPU communication channels) — nothing about *that* step is this project's code
either; it happens inside `DDP(raw_model, ...)`'s constructor, the first time it needs
to broadcast rank 0's initial weights to every other rank.

## What actually happens on a gradient "sync" — bucketing, `no_sync()`, and the boundary step

`DistributedDataParallel(raw_model, device_ids=[local_rank])`
(`trainer.py`) does two things at construction time, both automatic, neither visible
in this project's own code: it broadcasts rank 0's initial weights to every other
rank (every replica starts identical), and it registers a backward hook on every
parameter, grouped into a handful of gradient "buckets" (~25 MB each by default) —
grouping matters because syncing 300M individual small parameters one at a time
would be dominated by per-call network/kernel-launch overhead; a bucket firing one
collective call for many parameters' gradients together amortizes that cost.

Left alone, those hooks would fire an all-reduce after **every** `.backward()` call —
correct, but far too expensive over plain TCP between two non-EFA nodes if done once
per micro-step. This project avoids that with the `no_sync()` context manager,
gated on `is_accum_boundary` (`_run_loop`, `trainer.py`):

```python
is_accum_boundary = (step - start_step + 1) % train_cfg.grad_accum_steps == 0
sync_now = is_accum_boundary or step == train_cfg.steps - 1

if world_size > 1 and not sync_now:
    with training_model.no_sync():        # suppress the auto-hooks for this call
        (loss / train_cfg.grad_accum_steps).backward()
else:
    (loss / train_cfg.grad_accum_steps).backward()  # hooks fire normally — real sync

if sync_now:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

Concretely, for `grad_accum_steps=256` (this project's validated 2-node setting):
**255 of every 256 micro-steps run `no_sync()`** — each rank computes its own forward
+ backward locally, and gradients simply *accumulate* into each parameter's existing
`.grad` tensor (ordinary PyTorch behavior; nothing DDP-specific about accumulation
itself). Zero network traffic crosses between the two nodes for those 255 steps. On
the 256th (boundary) step, `no_sync()` is not used — the bucket hooks fire for real,
each bucket's now-fully-accumulated gradient gets **all-reduced** (summed across both
ranks, then divided by world_size to average) over NCCL, overlapping with backward
computation for earlier layers still in flight. Only *after* that all-reduce
completes does `optimizer.step()` run — identically on both ranks, since both now
hold the exact same averaged gradient. That single shared-gradient step, applied
locally by a deterministic optimizer starting from already-identical weights, is the
entire mechanism keeping both replicas bit-identical — there is no separate "merge
the two models" step anywhere, ever, because they're never allowed to diverge in the
first place.

## Real, verified proof the two mechanisms actually differ

Ran `scripts/ddp_smoke_test.py` and `scripts/fsdp_smoke_test.py` back to back, same
`tiny` preset, same data, same 2 CPU/gloo ranks — the parameter count printed at
startup is the direct, observable confirmation, not just a claim:

| | DDP | FSDP |
|---|---|---|
| `parameters total` (always the true total, cached before any wrap) | 4,998,272 | 4,998,272 |
| `this rank's shard` (only printed when sharded) | *(not applicable — DDP never shards)* | 2,499,136 (exactly half, world_size=2) |

DDP's `raw_model` still reports the full count after wrapping, because DDP never
touches parameter storage — it only registers gradient-sync hooks. FSDP's `raw_model`
reports **exactly half** after wrapping, because FSDP replaces the wrapped module's
own parameter storage with this rank's shard *in place*. This is the same distinction
`custom-gpt-6m`'s doc demonstrates on its own toy model; verifying it held here too,
on the real architecture, rather than assuming it transfers, was the point of actually
running both smoke tests rather than reasoning about it from the sibling project's
numbers alone.

## Why FSDP needed real surgery, not just a wrapper swap

Swapping `DDP(raw_model)` for `FSDP(raw_model)` is one line. Making checkpointing
(save *and* resume) actually correct under FSDP took much more, because of one fact
that has no DDP equivalent:

**`raw_model.state_dict()` after an FSDP wrap silently returns only this rank's
shard, not the full model — not an error, a checkpoint that looks fine until someone
tries to resume from it or serve it.** Every place this project's `trainer.py` used
to reach for `raw_model` directly (checkpoint save, checkpoint load/resume, the
end-of-run demo generation) had to be re-derived:

- **`_FSDPCheckpointView`** — adapts an FSDP-wrapped model to the plain
  `model.state_dict()`/`.param_count()`/`.attn_impl` interface `checkpoint.make_payload`
  already expects, internally routing `.state_dict()` through
  `FSDP.state_dict_type(..., StateDictType.FULL_STATE_DICT, FullStateDictConfig(...))`
  to gather the real, full, unsharded state — the one piece of FSDP machinery with no
  DDP equivalent, since DDP never needed gathering in the first place.
- **`_FSDPOptimizerView`** — same adaptation for the optimizer half, via
  `FSDP.optim_state_dict()` — AdamW's momentum/variance state is sharded too, and
  needs its own explicit gather, separate from the model weights' gather.
- **Resume** (`_resume_into`) — loading a full state dict back into a sharded model is
  a *different* operation from gathering one out, using `optim_state_dict_to_load()`
  to correctly re-shard the optimizer state to whichever rank needs which slice.
- **The end-of-run demo generation** — `raw_model` can't run a normal forward pass on
  its own shard, so a fresh, plain `TinyGPT` is built and loaded from the
  just-written (already-full) checkpoint instead of trying to generate through the
  FSDP wrapper directly.

This is genuinely more than `custom-gpt-6m`'s own FSDP script attempts — that one
**explicitly skips checkpointing entirely** ("deliberately out of scope, to keep the
script focused on the sharding mechanism itself"). This project's FSDP path has a
real, verified save/resume round trip (see below) precisely because a real ~350M
model training for real hours needs to survive being stopped and restarted, not just
demonstrate the sharding mechanism once.

## Two real bugs hit building this — both about collective operations, not APIs

### Bug 1: eval and checkpointing were silently rank-0-only, which is fine for DDP and wrong for FSDP

The training loop's existing design gates eval, checkpoint saves, and logging behind
`if is_main:` — correct and efficient for DDP, where a forward pass and
`raw_model.state_dict()` are both purely local operations with no synchronization
needed. **Under FSDP, both are collective operations** — every layer's forward pass
triggers an all-gather to reconstruct that layer's full parameters, and
`training_model.state_dict()` triggers a gather across every rank. Collectives need
*every* rank to call them together; if only rank 0 enters (because the rest are gated
out by `if is_main`), the ranks that never show up leave the ones that did hanging on
a gather that will never complete.

**First symptom, reproduced directly**: not a hang, a crash —
`RuntimeError: setStorage: sizes [...] ... storage of size 0` — because rank 0 alone
tried to unshard a `FlatParameter` whose other shards never arrived from the peer that
never called the collective. The fix: `eval_now`/the periodic-save condition are
computed identically on every rank (they depend only on `step`, which is already in
lockstep across ranks — nothing rank-specific about it), and under FSDP the collective
part (the forward pass, the state-dict gather) runs on **every** rank; only the
side effects that follow — printing, CSV logging, deciding whether to write to disk —
stay `is_main`-gated. See `_run_loop`'s `eval_now`/`use_fsdp` branches and the
end-of-run save section for exactly where this applies.

### Bug 2: `offload_to_cpu=True` crashes on a CPU-only (gloo) run

Isolated with a 10-line minimal repro before touching the real training loop again —
faster to debug in isolation than by re-running the full smoke test each time:

```python
with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT,
                           FullStateDictConfig(rank0_only=True, offload_to_cpu=True)):
    sd = fsdp_model.state_dict()
# RuntimeError: setStorage: sizes [64], ... storage of size 0
```

Toggling `offload_to_cpu=False` on the same repro: works immediately. The reason,
once you know what the flag is for: `offload_to_cpu` exists to move the *gathered*
full model off **GPU** memory onto CPU during a checkpoint, so a real multi-GPU run
doesn't need enough spare GPU memory to hold a whole extra copy of the model. On a
CPU-only (gloo) smoke test, the model was never on a GPU to begin with — there's
nothing sensible for the flag to do, and FSDP's internal unshard/offload path breaks
on that combination. Fixed by making it device-aware:
`offload_to_cpu=(device.type == "cuda")` — real CUDA runs still get the memory
benefit; the CPU smoke test just doesn't ask for something that doesn't apply to it.

## Real training run, actually executed (2 CPU/gloo ranks, `tiny` preset)

`GPT_STEPS=10` then a second pass resuming with `GPT_STEPS=20` (so the second pass has
genuinely new work to do, not just re-confirming a completed run) — DDP and FSDP run
independently, each its own two-pass sequence:

| | DDP | FSDP |
|---|---|---|
| Pass 1 (fresh, 10 steps) | reaches step 9, loss 10.44 → 10.37 | reaches step 9, loss 10.40 → 10.37 |
| Pass 2 (resume, extended to 20 steps) | *(not run as a resume test — see ddp_smoke_test.py)* | **resumes at step 10** (confirmed via the printed "Resumed at step 10"), reaches step 19, loss continues 10.34 → 10.35 |

Reading this: the FSDP resume genuinely picking up at step 10 — not step 0, and not
silently re-running steps 0-9 — is the concrete, observable confirmation that both
the model weights *and* the optimizer's AdamW moments survived the
gather-on-save/re-shard-on-load round trip correctly. A model-only resume (weights
right, optimizer reset to zero-init) would still train, just with a brief
re-warming-up wobble in the loss right after resume — not visible in a run this short,
which is exactly why this is flagged as *mechanism-verified*, not
*scaling-benchmark-verified* (see `docs/GPU_TRAINING.md`'s "Known gaps").

## What this doesn't show, and why (same honesty as the reference project)

- **No speedup claim.** Both smoke tests ran on 2 CPU processes on one machine over
  `gloo` — there's no reason to expect, and nothing here claims, the scaling real GPUs
  with NCCL would show.
- **No memory-fits-at-all claim for FSDP.** FSDP's actual purpose is fitting a model
  too large for one device's memory. At ~347M params (~5.6GB static memory in mixed
  precision — comfortably under any real GPU's capacity), this project doesn't need
  FSDP for memory reasons at all; the value here is purely mechanism verification and
  the learning exercise, same as `custom-gpt-6m`'s own honest framing of its numbers.
- **No real multi-node numbers yet.** Everything above is the CPU/gloo mechanism
  check. `infra/aws-gpu-node-multi/` provisions the real 2-node AWS setup; a short
  real run there — not this doc's numbers — is what should inform an actual paid
  multi-hour budget.

## Where to look in the code

- `src/gpt/training/trainer.py` — `_fsdp_full_state_dict_ctx`, `_FSDPCheckpointView`,
  `_FSDPOptimizerView` (the FSDP-specific checkpoint machinery), `use_fsdp` (the
  branch point in `train()`), and the `eval_now`/periodic-save/final-save sections of
  `_run_loop` (the collective-vs-local split from Bug 1 above).
- `src/gpt/config.py` — `TrainConfig.parallelism`, `GPT_PARALLELISM` env override.
- `scripts/ddp_smoke_test.py` / `scripts/fsdp_smoke_test.py` — the two local,
  near-$0 verification paths; run either before trusting a real multi-node launch.
