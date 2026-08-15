# Training this model on a rented GPU

This project's defaults target **one 24 GB GPU for roughly a day**, not a laptop —
unlike the sibling `custom-gpt-{10m,50m}` projects, whose `batch_size=1` exists for
MPS. Everything below is what those defaults are, why, and the order to do things in
so that billed hours go to training rather than to setup.

## The sizing decision

Chinchilla-optimal training is about **20 tokens per parameter** (`C ≈ 6ND`, `D ≈ 20N`,
so `N = sqrt(C/120)`). Turning a wall-clock budget into a model size:

| Budget @ 25% MFU on an L4 | Optimal params | Tokens needed |
|---|---|---|
| 12 h | ~104M | 2.1B |
| 24 h | ~148M | 3.0B |

Hence **152.8M parameters** (`E=768, L=16, C=1024`) and a **2.46B-token** budget —
16 tokens/param, a little under Chinchilla to leave headroom inside 24 h.

`E=768` is not an arbitrary round number either: [`MODEL_SIZING_GUIDE.md`](MODEL_SIZING_GUIDE.md)
computes the crossover at **E ≈ 697**, where transformer blocks overtake the token
embedding in parameter share. Below it, widening the model mostly buys a bigger lookup
table. `make config` confirms this one sits past it — blocks 74.2%, embedding 25.3%.

**The budget is implied, not declared.** A step is one *micro-batch*, so:

```
tokens = steps x batch_size x context_length = 150,000 x 16 x 1024 = 2.46B
```

Changing `batch_size` without changing `steps` silently rescales the entire run — and
also reshapes the LR schedule, since warmup is `2% of steps`. `gpt-train` prints the
resolved budget at startup for exactly this reason; read that line before walking away.

## Instance choice

| | GPU | bf16 | TF32 | ~$/hr | Verdict |
|---|---|---|---|---|---|
| `g4ad.xlarge` | AMD Radeon Pro V520 | — | — | 0.38 | **Unusable.** RDNA1; ROCm never supported it, so PyTorch finds no GPU and silently trains on 4 vCPUs. |
| `g4dn.xlarge` | NVIDIA T4 (Turing) | ✗ | ✗ | 0.53 | Marginal. No bf16 and no TF32, so `runtime.py`'s `set_float32_matmul_precision("high")` is a no-op and fp16 would need a GradScaler. |
| **`g6.xlarge`** | **NVIDIA L4 (Ada)** | ✓ | ✓ | **0.80** | **Recommended.** Lower bandwidth than the A10G (300 vs 600 GB/s) but the same code path and cheaper. |
| `g5.xlarge` | NVIDIA A10G (Ampere) | ✓ | ✓ | 1.01 | Equivalent; pick on price/availability. |

`precision="auto"` resolves to **bf16 on CUDA** and fp32 everywhere else. bf16 rather
than fp16 deliberately: it keeps fp32's exponent range, so no `GradScaler` and no
silent gradient underflow. Ampere and Ada have it; Turing does not.

## Why the corpus lives in a `.bin`

The obvious implementation — tokenize the corpus into one tensor on the training
device — does not survive this scale, in two independent ways:

* `torch.tensor(ids)` is **int64**. 2.5B tokens = **20 GB**, on a 24 GB card.
* `load_text` builds one Python `str` first — **~10 GB** for 2.5B tokens, on a
  g6.xlarge with **16 GB** of system RAM.

Either OOMs before step 0. So `gpt-tokenize` streams the corpus into a flat **uint16**
file (GPT-2's 50,257-token vocab fits in uint16) and training memmaps it: 2 bytes per
token, ~5 GB on disk for 2.5B tokens, near-zero resident memory because only the
sampled windows page in. It also means tokenization happens **once**, not on every
launch and crash-restart of an hourly-billed machine.

Chunked tokenization is verified to produce a **byte-identical token stream** to
tokenizing the whole file at once. That is not free: cutting a chunk mid-word stops BPE
forming merges across the cut (measured at **+24% tokens** of pure noise on a test
corpus), and cutting through a literal `<|endoftext|>` destroys the document boundary
it exists to provide. `dataset.py` only ever cuts at a separator, carrying an
incomplete tail forward.

## Runbook

Do steps 1-3 **locally**, before renting anything.

```bash
# 1. Sanity-check the config — confirm 152,791,296 params and the token budget
make config

# 2. Put the corpus in place as data/train.txt + data/test.txt, then tokenize once.
#    Verify the printed token count is what you expect BEFORE paying for a GPU.
make tokenize

# 3. Smoke-test the loop on a laptop with env overrides — no code edit needed
GPT_PRESET=tiny GPT_BATCH_SIZE=4 GPT_GRAD_ACCUM=2 GPT_STEPS=40 \
  GPT_EVAL_INTERVAL=20 GPT_SAVE_EVERY=20 uv run gpt-train
```

On the instance:

```bash
# 4. Copy the .bin files up (NOT the .txt — the .bin is half the size and
#    re-tokenizing on a rented GPU is wasted money)
scp data/train.bin data/test.bin  <instance>:.../data/

# 5. Confirm the GPU is what you think it is
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"

# 6. Train. Check the startup banner says bf16 and the expected token budget.
make train
```

## Measure MFU in the first ten minutes

Every hour estimate above assumes **25% MFU**. Verify it rather than trusting it:

```
MFU = (steps_per_sec x 6 x N x batch_size x context_length) / 121e12
```

`steps_per_sec` comes off the tqdm bar; `N = 152,791,296`. If you measure ~15%, cut
`steps` before committing to the run; if ~35%, the full 3.06B-token Chinchilla budget
fits. This one measurement replaces every projection on this page.

## Knobs, without editing code

| Env var | Field | Use |
|---|---|---|
| `GPT_BATCH_SIZE` | `batch_size` | Raise if VRAM allows; **adjust `GPT_STEPS` to match** |
| `GPT_GRAD_ACCUM` | `grad_accum_steps` | Effective batch = batch x accum |
| `GPT_STEPS` | `steps` | The token budget |
| `GPT_LR` / `GPT_MIN_LR` | `lr` / `min_lr` | |
| `GPT_PRECISION` | `precision` | `auto` \| `bf16` \| `fp16` \| `fp32` |
| `GPT_EVAL_INTERVAL` | `eval_interval` | Telemetry only; safe to change between resumes |
| `GPT_SAVE_EVERY` | `save_every_steps` | A checkpoint is ~1.8 GB here |
| `GPT_PRESET` | architecture | `tiny` for smoke tests |

## Known gaps

- **VRAM at `batch_size=16` is estimated, not measured** (~10 GB, of which the
  `batch x seq x 50257` logits tensor is the largest single item). It has only been run
  at `batch_size=1` on MPS locally. Watch `nvidia-smi` on the first real step and drop
  the batch if it is tight.
- **No `torch.compile`.** Likely a further speedup; untested here.
- **Single GPU only.** No DDP.
- **`get_batch` samples windows uniformly at random**, so "epochs" are an estimate
  (`est_epoch`), not a real pass over shuffled data.
