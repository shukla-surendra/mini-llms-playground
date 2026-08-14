# Training Migration Guide (GPU <-> Mac)

Use this when you train on a cloud GPU for some time, then continue on Mac (MPS/CPU), or switch back.

## What is supported

- Sequential resume across machines: supported.
- Simultaneous combined CPU+GPU training in this project: not supported.

`gpt-train` runs on one device per process (`cuda`, `mps`, or `cpu`) and resumes from checkpoint.

## Files to move

Minimum required:
- `checkpoints/<label>/latest.pt`

Recommended to keep full history:
- `checkpoints/<label>/best.pt`
- `checkpoints/<label>/serving.pt`
- `checkpoints/<label>/final.pt`
- `logs/train_eval_history.csv`

Data and code consistency:
- `data/train.txt`
- `data/test.txt`
- same project code version (`gpt-train`, related scripts)

## Cloud GPU -> Mac (resume)

On Mac, from repo root:

Sync the whole checkpoint directory so the `checkpoints/<label>/` layout is preserved
(`<label>` is the model size, e.g. `10m` — see `make config`):

```bash
REMOTE=user@gpu-host:/path/to/mini-llms-playground/from_scratch/custom-gpt-10m
rsync -avz "$REMOTE/checkpoints/" checkpoints/
rsync -avz "$REMOTE/logs/" logs/ || true
```

If you prefer `scp`, create the directory first — `scp` will not create it for you:

```bash
mkdir -p checkpoints/10m
scp "$REMOTE/checkpoints/10m/latest.pt" checkpoints/10m/
scp "$REMOTE/checkpoints/10m/best.pt" checkpoints/10m/ 2>/dev/null || true
```

Resume:

```bash
make train
```

## Mac -> Cloud GPU (resume)

From Mac repo root:

```bash
REMOTE=user@gpu-host:/path/to/mini-llms-playground/from_scratch/custom-gpt-10m
rsync -avz checkpoints/ "$REMOTE/checkpoints/"
rsync -avz logs/ "$REMOTE/logs/" || true
```

Then on GPU host:

```bash
cd /path/to/mini-llms-playground/from_scratch/custom-gpt-10m
make train
```

## Verification checklist before resuming

- Python deps installed (`uv sync (or: make setup)`)
- Checkpoint exists: `checkpoints/<label>/latest.pt` (`make config` shows the active label)
- `data/train.txt` and `data/test.txt` exist
- Model architecture config in code unchanged for the run:
  - `embed_size`
  - `num_heads`
  - `num_layers`
  - `dropout`
- `context_length` can be larger than dataset; code will derive `effective_context_length`.

Quick checkpoint metadata check:

```bash
python3 - <<'PY'
import torch
ckpt = torch.load("checkpoints/10m/latest.pt", map_location="cpu")
print("step:", ckpt.get("step"))
print("context_length:", ckpt.get("context_length"))
print("embed_size:", ckpt.get("embed_size"))
print("num_layers:", ckpt.get("num_layers"))
print("num_heads:", ckpt.get("num_heads"))
print("grad_accum_steps:", ckpt.get("grad_accum_steps"))
PY
```

## Important notes

- Optimizer state is portable across CPU/CUDA/MPS via `map_location`.
- If you change architecture fields and then resume, `load_state_dict` can fail on shape mismatch.
- If interrupted, `gpt-train` writes `checkpoints/<label>/latest.pt`; always sync that file first.
