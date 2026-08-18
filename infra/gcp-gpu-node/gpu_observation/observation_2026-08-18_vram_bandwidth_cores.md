# Observation: VRAM, memory bandwidth, and CUDA/Tensor Core utilization

Captured 2026-08-18, live `custom-gpt-50m` run on the L4 (GCP instance `mini-llm-gpu`,
`us-west1-a`), step ~357,199/1,000,000, `batch_size=1`. Raw command output saved
alongside this file in `observation_2026-08-18_vram_bandwidth.txt`. Format: exact
command run, what it returned, then what that actually means — no number below is
asserted without the command that produced it.

## What tooling is actually available here

```bash
ssh gpu@<ip> 'which dcgmi nsys ncu'
```
**Returned nothing** — none of `dcgmi` (Data Center GPU Manager), `nsys` (Nsight
Systems), or `ncu` (Nsight Compute) are installed on this box. Only plain
`nvidia-smi` is available. This matters directly for the third question below:
**true Tensor-Core-specific utilization is not measurable with what's on this VM** —
that requires Nsight Compute kernel profiling. What follows for that question is the
best available *proxy*, computed, not a direct tool reading — flagged clearly where
it applies.

## Command 1 — burst sampling over time, not a single snapshot

```bash
nvidia-smi dmon -c 15 -d 1    # 15 samples, 1 second apart
```
Output (`sm`/`mem` columns are %):
```
pwr  sm  mem  mclk  pclk
 35   0    0  6251  2040
 35   0    0  6251  2040
 35   0    0  6251  2040
 42  66   51  6251  1770
 71  69   53  6251  1650
 71  87   75  6251  1635
 71  85   73  6251  1590
 71  86   73  6251  1620
 72  84   72  6251  1590
 73  84   72  6251  1575
 65   7    0  6251  2025
 35   0    0  6251  2040
 35   0    0  6251  2040
 35   0    0  6251  2040
 44  86   76  6251  1650
```
**Finding**: this confirms the earlier-documented sampling-noise point with a full
time series instead of two isolated points — the GPU cycles between ~0% and
~85% busy roughly every 3-4 seconds. At `batch_size=1`, one training step is small
enough that the GPU finishes it and idles waiting for the next Python-dispatched
batch a meaningful fraction of the time. This is the mechanism, visible directly,
behind why a single `nvidia-smi` read is unreliable.

## Command 2 — the driver's own rolling average (more honest than any single read)

```bash
nvidia-smi -q -d MEMORY,UTILIZATION,CLOCK
```
Relevant section:
```
GPU Utilization Samples:    Duration=14.00s  Samples=71  Max=84%  Min=0%  Avg=58%
Memory Utilization Samples: Duration=14.00s  Samples=71  Max=75%  Min=0%  Avg=49%
FB Memory Usage: Total=23034 MiB  Used=1946 MiB  Free=20618 MiB
```
The driver itself keeps a short rolling buffer (71 samples over the last ~14s) and
reports Max/Min/Avg — this is the right number to quote, not any single instantaneous
read.

---

## Question 1: Current GPU VRAM utilization?

**1,946 MiB used / 23,034 MiB total ≈ 8.4%.**

Directly from Command 2's `FB Memory Usage`. At `batch_size=1` this model uses under
a tenth of the L4's 24GB — enormous headroom. (The earlier benchmark session found
even `batch_size=16` only used ~11GB / ~48% — VRAM has never been the constraint for
this model on this card; the constraint is throughput, covered below.)

## Question 2: How much memory bandwidth is being used?

**Two different numbers answer two different questions here — worth being precise:**

- **nvidia-smi's own "Memory Utilization" = 49% avg.** This is NOT "% of peak GB/s
  achieved" — NVIDIA defines it as *% of time the memory controller was active at
  all*, regardless of how much data moved in that time. A memory controller handling
  many small, spaced-out transactions can show high "utilization" while moving very
  few actual bytes per second.
- **Estimated actual achieved bandwidth ≈ 10.7 GB/s, or ~3.6% of the L4's 300 GB/s
  peak.** Computed, not measured (no bandwidth-counter tool installed): at 34.78
  steps/sec and ~309MB of weight-traffic per step (bf16 weights read on the forward
  pass, roughly 2x that in read/write traffic during backward — a rough estimate,
  not a profiled figure), achieved bandwidth ≈ `309MB × 34.78/sec ≈ 10.7 GB/s`.

**Finding**: the ~49%-busy-by-time vs. ~3.6%-of-peak-bytes gap is itself the clearest
evidence yet for this project's own memory-bandwidth-bound diagnosis at small batch
sizes — the memory controller is "active" nearly half the time, but each activation
moves so little data (tiny batch) that almost none of the card's actual 300 GB/s
capacity gets used. This is a **latency-bound** regime (many small, gap-spaced
transactions), not a throughput-bound one (few large, back-to-back transfers) — a
different flavor of "memory-bound" than the batch=4/8/16 benchmark's finding, which
was closer to genuinely saturating available bandwidth at each larger batch size.

## Question 3: How much CUDA / Tensor Core utilization?

**No direct measurement available — nvidia-smi doesn't separate "CUDA core"
from "Tensor Core" usage, and no profiler (Nsight Compute) is installed to get that
split.** Two honest proxies, not a real answer to "Tensor Core %" specifically:

- **Overall SM (compute core) busy-time**: 58% avg (Command 2's `GPU Utilization
  Samples`) — this is "some kernel was running on some SM," lumping regular CUDA
  cores and Tensor Cores together, and again measuring *time busy*, not *efficiency*.
- **MFU (Model FLOPs Utilization) ≈ 9.1%** — the actual efficiency metric, computed
  via the roofline method already established this session:
  ```
  tok/s = 34.78 steps/sec × 1 (batch) × 1024 (ctx) = 35,615 tok/s
  FLOPs/token ≈ 6 × 51,475,968 params ≈ 308.9M
  achieved = 35,615 × 308.9M ≈ 11.0 TFLOPS
  MFU = 11.0 / 121 TFLOPS (L4 dense bf16 peak) ≈ 9.1%
  ```
  This is the number that actually answers "how much of the card's compute
  capability (which is mostly Tensor Core FLOPs on Ada Lovelace) is being turned
  into useful work" — and it's **lower** than the batch=4-16 benchmark's own
  14.5-18.2% MFU findings, consistent with `batch_size=1` being an even less
  efficient regime than what was already diagnosed as inefficient.

**Finding**: three numbers, three different meanings, easy to conflate —
`58% SM busy-time` (time, not efficiency) vs. `9.1% MFU` (real compute efficiency)
vs. "0% Tensor-Core-specific" (genuinely unmeasured, not zero — just not
instrumented on this box). If a true Tensor-Core-specific number is ever needed,
that requires installing Nsight Compute (`ncu`) and profiling actual kernel
launches — a real, separate task, not a config flag.

### How many actual cores does that correspond to?

Real L4 (AD104) specs, confirmed via NVIDIA's own datasheet (not from memory):
**7,424 CUDA cores, 240 Tensor Cores (4th-gen), 60 RT cores, 58 SMs.**

Applying the 9.1% MFU figure as a throughput-equivalent (not a literal "these
specific cores were on" claim — real scheduling spreads work across cores in short
bursts, it doesn't cleanly light up a fixed 9.1% subset and leave the rest
completely idle):

```
CUDA-core-equivalent throughput:   676 of 7,424  (9.1%)
Tensor-core-equivalent throughput:  22 of   240  (9.1%)
```

This is the best available honest answer to "how many cores are we using" without
Nsight Compute installed — it says "the chip is producing ~9.1% of the useful work
7,424 fully-fed CUDA cores / 240 fully-fed Tensor Cores would produce," not "9.1% of
the cores are physically switched on while the rest are dark." The literal per-core
occupancy answer needs real kernel profiling, not derived from utilization/FLOP
math.

## Summary table

| Question | Answer | Source |
|---|---|---|
| VRAM used | 1,946 / 23,034 MiB (8.4%) | Direct read, `nvidia-smi -q -d MEMORY` |
| Memory "utilization" (time-busy) | 49% avg over 14s | Direct read, driver rolling average |
| Memory bandwidth (actual GB/s) | ~10.7 GB/s (~3.6% of 300GB/s peak) | **Estimated**, not directly measured |
| SM/compute "utilization" (time-busy) | 58% avg over 14s | Direct read, driver rolling average |
| Tensor Core utilization (specific) | Not measurable on this box | No profiler installed |
| MFU (real compute efficiency) | ~9.1% | **Computed** via roofline method, best available proxy |
