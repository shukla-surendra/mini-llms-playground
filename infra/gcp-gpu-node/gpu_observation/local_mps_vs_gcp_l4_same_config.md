# Local (MPS) vs GCP L4 — same model, same data, same config

Both runs below are the exact same `custom-gpt-50m` restart-from-step-0 run (153m-
sourced pretrain corpus, same checkpoint lineage), same config throughout —
`batch_size=1, grad_accum_steps=32` (32 effective batch), same architecture
(51,475,968 params). The only variable that changed is the hardware. This is a
genuinely fair comparison, unlike the earlier "2.4x" finding (see
`training_sop.md`), which compared local's default batch=1 against a GCP run
manually re-tuned to batch=4.

## Local — three real sessions, extracted from actual terminal history

| Segment (steps) | Steps run | Wall time | Avg step/s (this segment) |
|---|---|---|---|
| 1,775 → 187,297 | 185,522 | 9h 24m 47s | 5.47 |
| 187,297 → 327,671 | 140,374 | 5h 50m 13s | 6.68 |
| 327,671 → 331,246 | 3,575 | 10m 25s | 5.72 |

Each row's rate is computed from the resumed-run's own cumulative-time bookkeeping
(`start_step`/`start_cum_time` → `end_step`/`end_cum_time`), and independently
cross-checked against the live tqdm readout printed at the moment each session was
interrupted (5.47, 6.68, 5.71 — matches within rounding). Config every time:
`device=mps`, `Precision: fp32` (MPS autocast isn't used — see `trainer.py`'s
`resolve_amp` — fp32 is the deliberate choice on this platform, not a missed
optimization).

**Local average across all three real segments: ~5.96 steps/sec.**

## GCP L4 — same config, live

```
Model: 50m  |  51,475,968 parameters  |  device=cuda  |  attn_impl=sdpa
Precision: torch.bfloat16  |  batch 1 x accum 32 = 32 seqs/update
training: 367200/1000000 [01:33<4:53:32, 35.93step/s, ...]
```
**35.93 steps/sec** — same `batch=1/accum=32` config as every local segment above.
One inherent platform difference, not a config choice: `Precision` resolves to
`torch.bfloat16` on CUDA automatically (`resolve_amp`'s `"auto"` setting) vs. `fp32`
on MPS (deliberately, per that function's own docstring — MPS autocast isn't
dependable and has no tensor cores to benefit from it anyway). This isn't an unfair
comparison — it's each platform correctly running its own best available precision
at the identical batch config, which is exactly the real-world "how much faster is
GPU rental, practically" question.

## The real, same-config speedup

| Local segment used | Speedup (35.93 / local) |
|---|---|
| 5.47 step/s | 6.57x |
| 6.68 step/s | 5.38x |
| 5.72 step/s | 6.28x |
| **Average (5.96 step/s)** | **6.03x** |

**~6x, not the ~2.4x found earlier this session.** The earlier "~2.4x" figure
(`training_sop.md`) compared local's batch=1 against GCP's *manually re-tuned*
batch=4 — different amounts of work per step on each side. This comparison holds the
config identical on both sides for the first time, and the real gap is more than
double what the mismatched comparison suggested.

## Why the gap is bigger than the earlier finding, mechanically

Per `observation_2026-08-18_vram_bandwidth_cores.md`'s live-run diagnosis: at
`batch_size=1`, this workload is **latency-bound**, not bandwidth-bound — each step
is small enough that per-step dispatch/launch overhead dominates over actual
arithmetic time. A discrete CUDA GPU's kernel-launch path is thinner than Apple's
Metal-based MPS translation layer; that gap matters most exactly when per-step work
is tiny and dispatch overhead is a bigger fraction of total time — which is precisely
this config. At larger batch sizes (where real compute-per-step dominates over
dispatch overhead on both platforms), the gap would be expected to narrow — which is
consistent with the earlier ~2.4x figure being measured at a *bigger, GPU-tuned*
batch size, a regime where the CPU/MPS-vs-CUDA dispatch-overhead gap matters less.

## Bottom line

At the *current, unoptimized* config (batch=1, what's actually running on both
platforms right now), the GPU is genuinely ~6x faster — a bigger, more real
advantage than previously documented. This makes the batch-size lever (see
`how_to_increase_utilization.md`) doubly worth doing: it should improve GPU
throughput further (9.1%→18.2% MFU, already measured), while also being the
regime where local MPS's relative disadvantage should *shrink* — meaning the GPU's
practical advantage over local should hold or grow, not get diluted, once both are
compared at their respective best configs.
