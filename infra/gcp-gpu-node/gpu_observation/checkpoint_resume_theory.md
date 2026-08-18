# Checkpoint resume theory — what actually carries over across devices

Answers a specific question: if a checkpoint trains partway on GCP (GPU) and is then
brought back to a MacBook (MPS) to continue, is that a *real* resume — same as if
training had never left the original machine — or does something get lost/reset in
the move? Grounded in `custom-gpt-50m/src/gpt/training/trainer.py` and
`checkpoint.py`, not assumed behavior.

## Short answer

**The actual training state (weights, optimizer momentum, step position, LR
schedule) resumes correctly and completely regardless of which device you move
between.** The one thing that does *not* automatically travel with the checkpoint is
the **batch size / gradient-accumulation configuration** — that's re-read from
environment variables at every launch, not restored from the checkpoint file.

## What IS saved and correctly restored — the real training state

| State | Saved as | Restored via | Device-portable? |
|---|---|---|---|
| Step count | `checkpoint["step"]` (plain int) | `state["start_step"] = step + 1` | Yes — just a number |
| Model weights | `model_state_dict` | `model.load_state_dict(...)` | Yes — see precision note below |
| Optimizer momentum (AdamW's per-parameter moving averages) | `optimizer_state_dict` | `optimizer.load_state_dict(...)` | Yes — same mechanism as weights |
| LR schedule position | **not stored at all** — recomputed | `lr_for_step(step, train_cfg)`, a pure function of the step number | Yes — trivially, since it's derived, not stored |

**Why this is a genuine, mathematically correct resume**: none of the above requires
knowing which device (or which machine) produced the checkpoint. `torch.load(path,
map_location=device)` handles remapping tensors to whatever device you're now
running on — that's the whole mechanism, and it's symmetric in both directions
(GPU→Mac exactly as valid as Mac→GPU, which is what's actually been happening all
session, back and forth, repeatedly, without issue).

### The precision subtlety that makes this work cleanly

GPU training uses `torch.bfloat16` **autocast** — but autocast only casts
*activations* during the forward pass for speed. **The stored parameters themselves
never change dtype; they stay `fp32` the entire time**, on GPU exactly as they would
on Mac (comment in `trainer.py`: *"No GradScaler: bf16 keeps fp32's exponent range...
Weights/grads stay fp32 regardless"*). So a checkpoint saved mid-GPU-training is
already in the exact format Mac's fp32-only MPS path expects — no conversion step,
no precision loss, nothing to reconcile. This is what makes moving a checkpoint
between "the GPU trains in bf16" and "Mac trains in fp32" a non-issue: the file on
disk was fp32 the whole time regardless of which device wrote it.

## What is NOT saved/restored — the real gotcha

**`batch_size` and `grad_accum_steps` are not part of the checkpoint's restored
state.** They're resolved fresh from environment variables (`GPT_BATCH_SIZE`,
`GPT_GRAD_ACCUM`) at the start of every `gpt-train` invocation, via
`resolve_train_config()`. The checkpoint *does* record what batch/accum was in use
at save time (`make_payload`'s "provenance fields") — but that's for your own
reference when inspecting the file, not something the trainer reads back and
applies automatically.

**Practical consequence**: resuming with a plain `uv run gpt-train` (no env vars)
silently falls back to whatever that machine's *own* defaults are — which may not
match what the checkpoint was actually being trained with a moment ago on the other
machine. Concretely, this session: GPU checkpoint saved under `batch=4/accum=8`,
resumed on Mac with no override → Mac would silently use its own default
`batch=1/accum=32` instead, unless `GPT_BATCH_SIZE=4 GPT_GRAD_ACCUM=8` is passed
explicitly again.

This is not "wrong" or unsafe — training remains mathematically valid at any
batch/accum combination that multiplies out to the same or a different effective
batch — it's just **not automatic**, and worth deliberately deciding rather than
assuming it carries over.

## A minor, low-impact quirk worth knowing

```python
is_accum_boundary = (step - start_step + 1) % train_cfg.grad_accum_steps == 0
```
The gradient-accumulation boundary is computed **relative to `start_step`** (the
step you resumed at), not the absolute step count since training began. This means
every resume — regardless of device — effectively restarts the accumulation-window
counter from zero at that point, rather than continuing exactly where the previous
window left off. In practice this affects at most one accumulation window's exact
size right after a resume — negligible for training quality, just worth knowing it's
not perfectly seamless at the sub-step level, only at the step level.

## Summary

| Question | Answer |
|---|---|
| Does it resume from the exact right step? | Yes |
| Are the model's learned weights preserved exactly? | Yes |
| Is Adam's optimizer "memory" preserved? | Yes |
| Does the LR schedule pick up correctly? | Yes |
| Does precision need manual handling across devices? | No — handled automatically and correctly either way |
| Does batch_size/grad_accum come back automatically? | **No — must be re-specified manually if you want to preserve it** |
| Is this "a real resume" in the sense that matters (training continuity)? | **Yes, unambiguously** — the one manual step is a config choice, not a correctness gap |
