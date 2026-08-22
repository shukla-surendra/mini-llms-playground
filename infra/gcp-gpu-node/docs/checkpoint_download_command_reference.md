# Checkpoint Download Command Reference — Pulling Live Training State from the GPU Box

Companion to [`training_sop.md`](training_sop.md) (the full provisioning/teardown SOP)
and [`dcgm_gpu_command_reference.md`](dcgm_gpu_command_reference.md) /
[`nvidia_smi_command_reference.md`](nvidia_smi_command_reference.md) (GPU diagnostics) —
this doc covers the specific, repeated operation of pulling the *current* checkpoint
state down to a local machine **while training keeps running on the remote box**, with
real commands and real output from this project's own `custom-gpt-153m` run.

## Overview: the four-step journey, every time

```
1. Trigger a fresh checkpoint-sync on the box  (don't rely on the 10-min periodic timer alone)
2. Download from GCS to local, via gcloud storage rsync  (NOT raw scp — see "why not scp" below)
3. Verify byte counts match the bucket exactly
4. Verify each file actually loads with torch.load  (byte count matching ≠ proof of a valid file)
```

None of this touches the training process itself — it's a pure read/pull operation, safe
to run repeatedly while `gpt-train` keeps running in its own `tmux` session.

## Step 1: Trigger an Immediate Checkpoint Sync

The box's `checkpoint-sync.service` runs on a periodic timer
(`checkpoint_sync_minutes` in `terraform.tfvars`, default every 10 minutes) — waiting for
that timer means the local download could be up to 10 minutes stale. Trigger it manually
instead, for the freshest possible state:

```bash
ssh -i ~/.ssh/id_ed25519 gpu@<ip> \
  'sudo -u gpu /usr/local/bin/checkpoint-sync.sh --once'
```

**Real output**:

```text
At file:///home/gpu/tiny_llm/from_scratch/custom-gpt-153m/checkpoints/**, worker process 11394 thread 128224980240192 listed 3...
At gs://mini-llm-gpu-llm-training-dev-us-central1/153m/checkpoints/**, worker process 11394 thread 128224980240192 listed 3...
ERROR: Cannot check if the destination bucket is compatible for running parallel composite uploads as the user does not permission to perform GET operation on the bucket. The operation will be performed without parallel composite upload feature and hence might perform relatively slower.
Copying file:///home/gpu/tiny_llm/from_scratch/custom-gpt-153m/checkpoints/153m/best.pt to gs://mini-llm-gpu-llm-training-dev-us-central1/153m/checkpoints/153m/best.pt
Copying file:///home/gpu/tiny_llm/from_scratch/custom-gpt-153m/checkpoints/153m/latest.pt to gs://mini-llm-gpu-llm-training-dev-us-central1/153m/checkpoints/153m/latest.pt
Copying file:///home/gpu/tiny_llm/from_scratch/custom-gpt-153m/checkpoints/153m/serving.pt to gs://mini-llm-gpu-llm-training-dev-us-central1/153m/checkpoints/153m/serving.pt
Average throughput: 252.1MiB/s
2026-08-22T11:27:01+00:00 [ckpt-sync] ok
```

**What this output means**: the `ERROR:` line is benign and expected — it's `gcloud`
warning that the service account lacks `GET` permission needed to check for parallel
composite upload compatibility, so it falls back to a single-stream upload instead
(confirmed by the still-healthy `252.1MiB/s` throughput — not actually a bottleneck in
practice). **The line that actually matters is the last one**: `[ckpt-sync] ok` — the
script's own explicit success marker. Confirm this line is present before trusting the
sync happened; a script that errors *before* reaching this point would leave the bucket
stale with no obvious indication in a casual glance at the output.

**Checking the current training step at the same time, useful for correlating what
you're about to download**:

```bash
ssh -i ~/.ssh/id_ed25519 gpu@<ip> \
  'tail -c 800 /tmp/gpt-train.log | tr "\r" "\n" | tail -2'
```

Real output: `training: 60%|█████▉ | 76443/127933 [2:30:25<2:18:28, 6.20step/s, ...
test_loss=5.1834, ...]` — the `tr "\r" "\n"` converts the tqdm progress bar's
carriage-return-only line updates into real newlines so `tail` can actually isolate the
most recent one; without it, `tail` sees one enormous single "line" and returns the whole
buffered blob instead of just the last update.

## Step 2: Download via `gcloud storage rsync`, Not Raw `scp`

```bash
cd infra/gcp-gpu-node
make download-checkpoints PROJECT_DIR=../../from_scratch/custom-gpt-153m
```

This runs `gcloud storage rsync gs://<bucket>/153m/checkpoints/ checkpoints/ --recursive`
under the hood. **Real output** (progress dots trimmed):

```text
Copying gs://mini-llm-gpu-llm-training-dev-us-central1/153m/checkpoints/153m/best.pt to file://checkpoints/153m/best.pt
Copying gs://mini-llm-gpu-llm-training-dev-us-central1/153m/checkpoints/153m/latest.pt to file://checkpoints/153m/latest.pt
Copying gs://mini-llm-gpu-llm-training-dev-us-central1/153m/checkpoints/153m/serving.pt to file://checkpoints/153m/serving.pt
Average throughput: 16.4MiB/s
[exited with code 0]
```

**Why `gcloud storage rsync` and never raw `scp` for this** — a real, previously-observed
failure mode, not theoretical caution: a direct `scp gpu@<ip>:.../latest.pt ./` of a
~1.8GB checkpoint was once interrupted by a tool's own command timeout mid-transfer, and
**the resulting local file looked complete by byte count but was actually truncated and
failed to `torch.load`** — `scp` doesn't reliably resume or clearly signal partial
failure the way `gcloud storage rsync` does. Going through the bucket (push from the box,
then pull via `rsync`) avoids this entirely, at the cost of one extra hop.

**~5.1-5.5GB over home internet routinely exceeds a 120-180 second interactive command
timeout** — expect this step to move to a background task; wait for its actual completion
signal (exit code 0) rather than assuming it finished when the terminal prompt returns.

## Step 3: Verify Byte Counts Against the Bucket — Don't Trust the "exited with code 0" Alone

```bash
ls -la checkpoints/153m/*.pt
gcloud storage ls -l "gs://<bucket>/153m/checkpoints/153m/*.pt"
```

**Real output, side by side**:

```text
Local:
-rw-r--r-- 1 user staff 1833733471 Aug 22 16:55 checkpoints/153m/best.pt
-rw-r--r-- 1 user staff 1833735627 Aug 22 16:54 checkpoints/153m/latest.pt
-rw-r--r-- 1 user staff 1833736417 Aug 22 16:55 checkpoints/153m/serving.pt

Bucket:
1833733471  gs://.../153m/checkpoints/153m/best.pt
1833735627  gs://.../153m/checkpoints/153m/latest.pt
1833736417  gs://.../153m/checkpoints/153m/serving.pt
```

Every byte count matched exactly — the expected, healthy result. A mismatch here would
mean the transfer was genuinely incomplete, worth catching *before* the more expensive
`torch.load` verification below.

## Step 4: Verify Each File Actually Loads — Byte Count Matching Is Necessary, Not Sufficient

**This is the step that would have caught the historical `scp`-truncation failure above**
— a byte-count match only proves the file is the *right size*, not that its contents are
valid. Load each one for real:

```bash
uv run python3 -c "
import torch
for name in ['latest', 'best', 'serving']:
    ckpt = torch.load(f'checkpoints/153m/{name}.pt', map_location='cpu', weights_only=False)
    print(f\"{name}.pt: step={ckpt['step']}, best_test_loss={ckpt['best_test_loss']:.4f}, \"
          f\"processed_tokens={ckpt['processed_tokens']:,}, \"
          f\"total_training_seconds={ckpt['total_training_seconds']:.0f}\")
"
```

**Real output**:

```text
latest.pt: step=75999, best_test_loss=5.2134, processed_tokens=311,296,000, total_training_seconds=59993
best.pt: step=76000, best_test_loss=5.1834, processed_tokens=311,296,000, total_training_seconds=60009
serving.pt: step=76000, best_test_loss=5.1834, processed_tokens=311,296,000, total_training_seconds=60009
```

**Reading this output**: all three loaded without error — real proof of a valid,
uncorrupted checkpoint, not an assumption. `best.pt`/`serving.pt` sit one step ahead of
`latest.pt` (76,000 vs. 75,999) — expected, not a bug: `best.pt` saves whenever a new
best `test_loss` is found, on its own trigger, independent of `latest.pt`'s own periodic
save cadence, so the two can differ by a handful of steps depending on exactly when the
sync happened to land relative to each save event.

**Why `map_location='cpu'`**: the checkpoint was saved from a CUDA device on the GPU box;
loading it on a Mac with no CUDA available would otherwise raise a device-not-found
error. `map_location='cpu'` remaps every tensor to CPU on load regardless of what device
it was saved from — the same cross-device portability
[already noted in `training_sop.md`'s Phase 8](training_sop.md#phase-8-using-the-trained-model-locally-after-download-checkpoints-pending):
a checkpoint trained on GCP's CUDA loads on a Mac's CPU/MPS with zero conversion step.

**Why `weights_only=False`**: this checkpoint's dict carries more than tensors —
`step`, `best_test_loss`, `processed_tokens`, model hyperparameters (`embed_size`,
`num_layers`, etc.), the tokenizer reference. PyTorch's `weights_only=True` default (as
of newer PyTorch versions, for security — restricting what a loaded pickle can construct)
would reject this richer payload; `weights_only=False` is required here specifically
because this project's checkpoint format is deliberately more than a bare state dict.

## Full Sequence, Copy-Pasteable

```bash
# 1. Fresh sync on the box
ssh -i ~/.ssh/id_ed25519 gpu@<ip> 'sudo -u gpu /usr/local/bin/checkpoint-sync.sh --once'

# 2. Download locally
cd infra/gcp-gpu-node
make download-checkpoints PROJECT_DIR=../../from_scratch/custom-gpt-153m

# 3. Verify sizes
ls -la ../../from_scratch/custom-gpt-153m/checkpoints/153m/*.pt
gcloud storage ls -l "gs://mini-llm-gpu-llm-training-dev-us-central1/153m/checkpoints/153m/*.pt"

# 4. Verify each file actually loads
cd ../../from_scratch/custom-gpt-153m
uv run python3 -c "
import torch
for name in ['latest', 'best', 'serving']:
    ckpt = torch.load(f'checkpoints/153m/{name}.pt', map_location='cpu', weights_only=False)
    print(name, ckpt['step'], ckpt['best_test_loss'])
"
```

## Common Mistakes This Sequence Avoids

| Mistake | What actually happens | Why the steps above prevent it |
|---|---|---|
| Trusting the 10-min periodic sync timer | Downloaded checkpoint can be up to 10 minutes stale | Step 1 forces an immediate sync first |
| Using raw `scp` for the transfer | Can silently truncate on a timeout, producing a corrupted-but-plausible-sized file | Step 2 routes through the bucket via `gcloud storage rsync` instead |
| Trusting `exited with code 0` alone | Doesn't prove file *contents* are valid, only that the command didn't crash | Step 4's `torch.load` is the actual proof |
| Trusting matching byte counts alone | A truncated file can coincidentally still match if corruption happened mid-stream without changing length in some failure modes | Step 4 again — load, don't just measure |
