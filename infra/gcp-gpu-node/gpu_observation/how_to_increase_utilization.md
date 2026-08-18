# How to increase GPU utilization — levers, ranked

Follows directly from `observation_2026-08-18_vram_bandwidth_cores.md`'s finding:
this run (`batch_size=1`) sits at ~9.1% MFU, ~3.6% of peak memory bandwidth, ~8.4%
VRAM. This doc is the "what do I actually do about it" companion — every number
below is either already measured this session (`training_sop.md`'s benchmark work)
or clearly marked as untested/estimated.

## Current state, for reference

| Metric | This run (batch=1) | Already-benchmarked (batch=4) |
|---|---|---|
| steps/sec | ~35 | 13.3 |
| tokens/sec | ~35,600 | ~54,500 |
| MFU | ~9.1% | 18.2% |
| VRAM used | 8.4% | ~48% (11GB/23GB) |
| Effective CUDA cores | ~676 / 7,424 | ~1,350 / 7,424 (estimated, scales with MFU) |

The batch=4 column isn't hypothetical — it's a real number from this session's own
`gpt-benchmark --sweep-batch` run. Nothing below needs new benchmarking to confirm
the top lever; it needs relaunching with that config.

## Levers, ranked by expected payoff vs. effort

### 1. Increase batch size — highest payoff, already proven, zero risk

```bash
GPT_BATCH_SIZE=4 GPT_GRAD_ACCUM=8 uv run gpt-train
```
Same effective batch (4×8=32, unchanged from the checkpoint's own training history —
this doesn't change what the model learns, only how the work is chunked per step).
**Already measured**: MFU 9.1% → 18.2%, tokens/sec ~35,600 → ~54,500, a genuine ~53%
wall-clock speedup for the exact same training budget, no extra cost.

**Why not go bigger than 4** — also already measured: batch=8 and batch=16 came back
*flat-to-slightly-worse* than batch=4 in the original sweep (56,350 / 55,308 tok/s
vs. 56,888 tok/s at batch=4). This model is small enough that 4 is roughly the
efficiency ceiling for batch size alone — going to 8/16 mostly just uses more VRAM
without more throughput. Don't chase batch=32+ expecting it to keep helping.

**Risk**: none — checkpoint-compatible, same effective batch, reversible by
relaunching without the env vars.

### 2. `torch.compile(mode="reduce-overhead")` — untested, plausible fit for THIS specific problem

```bash
GPT_COMPILE=1 uv run gpt-train   # only tests default mode today
```
Only default-mode `torch.compile` was tested this session (+2%, not worth it alone).
**`reduce-overhead` mode specifically targets kernel-launch overhead via CUDA
graphs** — a different mechanism than default-mode's operator fusion, and one that
maps directly onto today's finding: at `batch_size=1`, the run is latency-bound
(idle gaps between tiny bursts, per the `dmon` time-series in the observation doc),
which is exactly the kind of overhead CUDA graphs are built to eliminate. Worth a
real test specifically at `batch_size=1` (where launch overhead is proportionally
largest) rather than dismissing compile based on the batch=4-16 test alone.

**Caveat**: requires static shapes / no data-dependent control flow across calls —
verify `get_batch()`'s output shapes are constant before trusting this blindly (this
project's own docs already flag this as the reason it wasn't tried yet).

**Risk**: low — opt-in via env var, falls back to normal execution if compile fails.

### 3. Run a second model concurrently on the same GPU — doesn't speed up THIS run, but uses the idle 90%

Since MFU is only ~9%, the L4 has genuine spare compute capacity most of the time —
not "nothing more to give," just nothing *this specific run* is asking for. A second
training process (e.g. `custom-gpt-10m`, or a second `custom-gpt-50m` variant) via
CUDA's MPS (Multi-Process Service — unrelated to Apple's MPS backend, an NVIDIA
feature for sharing one GPU across processes efficiently) could plausibly get real
extra throughput for near-$0 marginal cost, since there's provably idle capacity to
use. Relevant only if there's a second model actually worth training right now —
not a fix for this run's own speed.

### 4. Everything else already ruled out this session — don't re-try without a new reason

- **Bigger/different GPU** (A100 etc.) — no clear win; GCP prices GPUs roughly
  proportional to bandwidth, so a 5x-bandwidth card costs ~5x more. A wash at best,
  likely worse once fixed per-step overheads (which don't shrink with more
  bandwidth) are accounted for.
- **Removing CPU-GPU sync stalls** (the `loss.item()` progress-bar read) — tested,
  ~0% (noise-level) gain.
- **grad_accum restructuring beyond batch size itself** — already covered by lever
  #1's own sweep; batch=8/16 came back flat-to-worse, not a separate lever.

## Recommended order

**Do #1 now** — it's proven, free, and the single biggest lever available (~53%
faster for zero cost or risk). **Consider #2** as a follow-up experiment once #1 is
running, specifically because today's batch=1 data made a case for it that the
original batch=4-16 test didn't. **#3** only if there's genuinely a second model
worth training in parallel — not a fix to reach for by default.
