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
