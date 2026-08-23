# Checkpointing and Resuming Training

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 2B — Training at Scale
(appended after the original numbered catalog, alongside
[Chapter 25](25_efficient_attention_flash_and_sdpa.md) and
[Chapter 26](26_distributed_training_ddp_and_fsdp.md) — see [Chapter 0](00_roadmap.md)'s
reading-order note). Builds on [Chapter 2](02_parameters_vs_hyperparameters.md)'s
parameters-vs-hyperparameters distinction, which this chapter's central design point
depends on directly.

## In Plain English

A checkpoint is a save file for a training run — but unlike a video game save, it can't
just be "the current numbers." If it only stored the learned weights, resuming would mean
somehow already knowing the exact architecture (how many layers, how wide, what context
length) those weights were trained under, and getting even one of those wrong wouldn't
produce a slightly-different model — it would either crash outright or silently corrupt
what loads. A checkpoint that carries its own architecture description alongside the
weights sidesteps that entirely: it's self-describing, so resuming never requires the
resuming code to already know what it's about to load.

## The First-Principles Explanation

### What a correct checkpoint has to store, and why

Three categories of information, each answering a different failure mode:

1. **Learned state** — the model's `state_dict` (every weight) and the optimizer's own
   state (momentum buffers, per-parameter adaptive learning-rate estimates for AdamW).
   Skipping the optimizer state doesn't break correctness the way skipping architecture
   does, but it does change training dynamics on resume — the optimizer effectively
   restarts its own internal statistics from scratch even though the model weights
   didn't, which is a real, if subtler, discontinuity.
2. **Architecture (hyperparameters)** — `embed_size`, `num_heads`, `num_layers`,
   `context_length`, vocabulary size — exactly the fields [Chapter 2](02_parameters_vs_hyperparameters.md)
   distinguishes from the parameters themselves. This is what makes a checkpoint
   self-describing: the code that loads it can reconstruct the exact model that produced
   it without being told its size out of band.
3. **Progress metadata** — the current step, the best validation loss seen so far,
   cumulative processed tokens/wall-clock time. None of this is needed to *load* the
   model correctly, but all of it is needed to resume training *correctly positioned* —
   without a saved step, a "resumed" run would have no way to know it isn't starting from
   step 0 again.

### Why the architecture check exists, and what it prevents

Because a checkpoint stores its own architecture, the code that resumes training can (and
should) verify the current run's configuration matches what the checkpoint was actually
trained under, before touching `load_state_dict` at all. Skipping this check doesn't fail
loudly in every case — a `state_dict` load with a shape mismatch raises an error, which is
at least visible, but a resume where every *shape* happens to match while the semantic
config differs (an easy trap in adjacent systems — see the tokenizer-mismatch version of
this same problem in [Chapter 28](28_catastrophic_forgetting_and_continual_training.md))
can load "successfully" while silently loading the wrong assumptions.

### Why saves are atomic

A checkpoint save is not a single filesystem operation — writing a multi-hundred-megabyte
file takes real, non-instant time, and a training loop is exactly the kind of long-running
process that gets interrupted (`Ctrl-C`, a preemptible cloud instance reclaimed
mid-write, a crash). If the interrupt lands partway through writing the checkpoint file
directly, the file on disk is truncated — neither the old checkpoint nor a valid new one,
just corrupt. The standard fix: write to a temporary sibling file first, and only rename
it over the real path once the write has fully completed. A rename is atomic at the
filesystem level — a reader either sees the complete old file (rename hasn't happened
yet) or the complete new one (rename has happened), never a partial write, regardless of
exactly when an interrupt lands.

### Why checkpoints move across devices for free

Training on a cloud GPU and resuming on a laptop's CPU/MPS (or vice versa) works because
`state_dict` tensors are just numbers plus shape metadata — loading a checkpoint with an
explicit target device (`map_location=device`) transparently moves every tensor to
wherever it's being loaded, without any manual per-tensor conversion. This is what makes
the "train on a GPU box, sync the checkpoint directory, resume on a laptop" workflow a
matter of copying files, not a code change.

### One file or many? Checkpointing under distributed training

Every checkpoint this repo writes is a single file because every training run in this
repo is single-process — there is only ever one `state_dict` to save. Once training is
distributed ([Chapter 26](26_distributed_training_ddp_and_fsdp.md)), the checkpoint format
has to follow whatever split the parallelism strategy already made:

- **DDP**: every rank holds an identical full replica, so saving from rank 0 alone is
  sufficient — still one file.
- **FSDP/ZeRO, gathered mode**: ranks temporarily all-gather the full parameters onto one
  rank before saving — still one file, but requires that one rank to momentarily hold the
  entire model again, which defeats the purpose of sharding for a model too big to fit
  that way in the first place.
- **FSDP/ZeRO, sharded mode**: each rank saves its own shard directly — one file *per
  rank*, the standard approach for genuinely large models. DeepSpeed's ZeRO checkpoints
  look exactly like this: a directory of per-rank optimizer/model shard files, plus a
  `zero_to_fp32.py` script whose entire job is consolidating those shards back into one
  file for downstream use (serving, or a simple single-process load).
- **Tensor / pipeline parallelism**: each rank naturally owns a distinct slice (of a
  layer, or of the layer stack) — checkpointing is naturally per-rank here too.

One practical trap worth naming explicitly: **Hugging Face's sharded model format**
(`model-00001-of-0000N.safetensors` plus an `index.json` mapping parameter names to shard
files) looks identical in shape to a distributed-training checkpoint, but is usually
unrelated to it — it is a post-hoc split done purely for file-size/download limits,
applied after training regardless of how many GPUs actually trained the model. A model
trained on one GPU can ship as 5 shard files; a model trained on 64 GPUs can ship as 1
file if someone consolidated it first. File count alone does not reveal training topology.

The practical consequence: a simple, single-process load-and-generate script — like this
repo's own `inference.py`/`generate()` calls, which expect one `state_dict` in one file —
only works directly against a *consolidated* single-file checkpoint. A sharded
distributed checkpoint either needs loading back into the same distributed setup, or an
explicit consolidation step first. That step is a real part of the pipeline, not an
afterthought to skip.

## Grounded in This Repo's Code

[`from_scratch/custom-gpt-10m/src/gpt/checkpoint.py`](../../from_scratch/custom-gpt-10m/src/gpt/checkpoint.py)'s
`atomic_save` is exactly the tmp-then-rename pattern:

```python
def atomic_save(payload, path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)   # atomic on the filesystem
```

and `make_payload` is the three-category structure above, made concrete — learned state
(`model_state_dict`, `optimizer_state_dict`), architecture (`embed_size`, `num_heads`,
`num_layers`, `context_length`, `vocab_size`), and progress metadata (`step`,
`best_test_loss`, `processed_tokens`, `total_training_seconds`) all in one payload dict.
Every from-scratch project in this repo follows the same shape:
[`custom-gpt-153m`](../../from_scratch/custom-gpt-153m/) writes the identical fields to
`tiny_llm_checkpoint*.pt`, and
[`custom-gpt-6m`](../../from_scratch/custom-gpt-6m/)'s `train.py` does the same
into `tinystories_gpt_checkpoint*.pt`. Cross-device resume is
[`from_scratch/custom-gpt-10m/docs/MIGRATION.md`](../../from_scratch/custom-gpt-10m/docs/MIGRATION.md)'s
whole subject — copying `checkpoints/<label>/latest.pt` between a cloud GPU host and a
Mac via `rsync`/`scp` and resuming with the same training command on either side, no code
change required, because of exactly the device-portability property described above.

## Deep-Dive: `best` vs. `latest` Are Solving Different Problems, Not Duplicating One

Every project here writes at least two checkpoint files with different update rules:
`latest` is overwritten on every periodic save (and on interrupt), unconditionally —
its job is "never lose more than a few minutes of progress to a crash." `best` is only
overwritten when a new evaluation actually beats the previous best validation loss (see
[Chapter 15](15_evaluating_a_model_while_training.md)) — its job is "never let a
temporarily-worse evaluation contaminate what gets served." These solve genuinely
different failure modes: `latest` protects against losing work; `best` protects against
serving a regression. Collapsing them into a single file would mean picking one job over
the other — either resuming from a possibly-worse point after a bad-luck evaluation, or
losing recent progress on interrupt because the last save happened to not be an
improvement.

## Try It Yourself

- Open a checkpoint file from any project here with
  `torch.load(path, map_location="cpu")` and print its keys — confirm all three
  categories (learned state, architecture, progress metadata) are present in one payload.
- Deliberately change `num_layers` in a project's config, then attempt to resume from an
  existing checkpoint trained under the old value — observe whether the failure is a
  clear, actionable error or a silent shape mismatch, and consider which behavior a
  well-designed resume check should produce.
- Interrupt a training run with `Ctrl-C` partway through a save (if timing allows) and
  confirm the previous `latest.pt` is intact rather than corrupted — the atomic-save
  guarantee in action.

## Common Misconceptions

- **"A checkpoint is just the trained weights."** It's the weights plus enough
  architecture and provenance metadata to reconstruct the exact model and resume
  correctly positioned — the weights alone are not self-sufficient.
- **"Resuming on a different device (GPU → CPU) requires converting the checkpoint
  first."** It doesn't — `map_location` handles this transparently at load time; the
  checkpoint file itself doesn't need to change.
- **"`best` and `latest` checkpoints are redundant — just keep the most recent one."**
  They protect against different failures (lost progress vs. served regression); keeping
  only one gives up one of those protections.

## Practice Questions

1. Why does storing architecture fields inside the checkpoint matter more than it would
   for a simpler save/load setup where the model size never changes run to run?
2. Walk through what happens, step by step, if a training process is killed exactly
   halfway through `torch.save` writing a checkpoint directly to its final path — and how
   the tmp-then-rename pattern changes that outcome.
3. Give a concrete scenario where `latest` and `best` would point at two different steps
   at the same moment in a training run, and explain why that's the correct, intended
   behavior rather than a bug.
4. A model was trained with FSDP across 8 GPUs using sharded-state-dict mode. Explain why
   `torch.load()` on a single one of the resulting shard files would not give you a usable
   model, and what step is needed before this repo's own single-process `generate()`-style
   loading code could use it.

## Key Terms

- **Self-describing checkpoint**: a saved payload that includes its own architecture
  metadata, so the loading code doesn't need to be told the model's shape out of band.
- **Atomic save**: writing to a temporary file and renaming it over the target path, so an
  interrupted write can never leave a corrupted file at the real checkpoint path.
- **`map_location`**: the load-time argument that transparently moves a checkpoint's
  tensors to a target device, making checkpoints portable across CPU/CUDA/MPS.
- **`best` vs. `latest` checkpoint**: two different save policies protecting against two
  different failures — losing recent progress, and serving a temporary regression.
- **Sharded checkpoint**: a distributed-training checkpoint saved as one file per rank
  rather than gathered onto one, avoiding the requirement that any single process ever
  hold the full model — the standard format for models too large to fit on one device,
  and the reason a consolidation step (e.g. DeepSpeed's `zero_to_fp32.py`) is often
  needed before simple single-process inference code can use it.
