# GPU metrics, explained — grounded in a real live run

Every number in this doc is real, captured from the actual `custom-gpt-50m` GCP L4
resume session (2026-08-18, instance `mini-llm-gpu`, `us-west1-a`) via
`nvidia-smi -q` (full detail) — see `snapshot_1_full.txt` / `snapshot_2_full.txt` in
this folder for the raw, unedited dumps these numbers are pulled from. Nothing here
is a textbook example; it's what this specific 51.5M-param model training at
`batch_size=1` actually produced on an L4.

## The single most important thing to understand first

**`nvidia-smi` utilization numbers are instantaneous samples, not averages.**
Two queries **3 seconds apart**, same training run, same steady state:

```
snapshot_1: GPU Utilization = 84 %
snapshot_2: GPU Utilization =  0 %
```

Neither number is wrong. `nvidia-smi` asks the driver "was any kernel executing on
this GPU during the last sampling window (~1ms)?" — it's a snapshot of one instant,
not a rolling average over the last few seconds. At `batch_size=1`, each training
step is so small that the GPU finishes its work and goes idle waiting for the next
Python-dispatched kernel far more often than at a larger batch size — so consecutive
samples can legitimately swing between "fully busy" and "completely idle." **Never
trust a single `nvidia-smi` reading** — sample several times, or watch it live with
`watch -n 1 nvidia-smi`, before concluding anything about real utilization.

## Utilization metrics

```
Utilization
    GPU      : 84 %      <- % of time an SM (compute core) was executing something
    Memory   : 71 %      <- % of time the memory controller was actively transferring data
    Encoder  :  0 %      <- video encoder engine (irrelevant for ML training)
    Decoder  :  0 %      <- video decoder engine (irrelevant for ML training)
    JPEG     :  0 %      <- JPEG codec engine (irrelevant for ML training)
    OFA      :  0 %      <- Optical Flow Accelerator (irrelevant for ML training)
```

**GPU utilization** and **Memory utilization** are the two that matter for training.
Critically: **high GPU utilization does not mean the GPU is being used efficiently**
— it only means *some* kernel was running. A model too small to keep the compute
units fed (exactly this project's own documented finding: 50m pins at ~14-19% MFU
regardless of batch size) can show 80%+ "GPU utilization" while still wasting most
of the GPU's actual arithmetic capacity — because "busy" here means "not idle," not
"doing useful FLOPs at peak rate." See `training_sop.md`'s MFU discussion for the
real efficiency metric this doesn't capture.

## Memory metrics

```
FB Memory Usage
    Total     : 23034 MiB   <- total VRAM on the card (L4 = 24GB nominal, 23034 MiB usable)
    Reserved  :   471 MiB   <- reserved by the driver itself, not available to any process
    Used      :  1946 MiB   <- currently allocated by running processes
    Free      : 20618 MiB   <- available for new allocations
```

1,946 MiB used out of 23,034 MiB total means this run (`batch_size=1`) is using
**under 9% of available VRAM** — enormous headroom. This is exactly why the earlier
benchmark session found a much bigger batch size (`batch=16`) only used ~11GB — this
model's memory footprint is tiny relative to the L4's 24GB; VRAM was never close to
the actual constraint here (throughput was, per the MFU finding above).

```
BAR1 Memory Usage
    Total : 32768 MiB   <- CPU-addressable window into GPU memory (not the same as FB memory)
    Used  :  1947 MiB
```
BAR1 is a separate address space the CPU uses to directly access GPU memory (for
transfers, mapped buffers) — usually not something to worry about for training
unless it's exhausted (rare, matters more for GPU passthrough / multi-tenant setups).

## Power metrics

```
GPU Power Readings
    Average Power Draw       : 61.83 W
    Instantaneous Power Draw : 36.77 W
    Current Power Limit      : 72.00 W   <- the cap this GPU is allowed to draw
    Default Power Limit      : 72.00 W
    Min Power Limit          : 40.00 W
    Max Power Limit          : 72.00 W
```

The L4 is a **72W-capped card** (low-power by GPU standards — it's designed for
inference/edge deployment density, not raw training throughput; compare an A100's
300-400W envelope). "Average" vs "Instantaneous" here shows the same
sampling-noise issue as utilization: instantaneous power swings with whatever the
GPU was doing in that exact moment, while average smooths it. Seeing power draw
sitting near the 72W cap during active steps (69-71W was observed earlier in this
session) is a genuine sign of real, sustained work — a truly idle GPU sits closer to
its 40W minimum.

## Thermal metrics

```
Temperature
    GPU Current Temp          : 76 C
    GPU T.Limit Temp          :  9 C   <- headroom before throttling starts (not absolute)
    GPU Shutdown T.Limit Temp : -5 C   <- headroom before emergency shutdown
    GPU Slowdown T.Limit Temp : -2 C   <- headroom before clock throttling
```

Confusingly, several of these "T.Limit" fields report as a **delta/headroom in °C
below the actual limit**, not an absolute temperature — that's why they show small
or negative numbers here rather than something like "85°C." 76°C sustained under
load is normal and well within safe range for this card; the relevant question is
whether `Clocks Event Reasons -> HW Thermal Slowdown` ever flips to "Active" (it
didn't, in this session's snapshots — confirmed `Not Active` throughout).

## Clock metrics

```
Clocks
    Graphics : 2025 MHz
    SM       : 2025 MHz
    Memory   : 6251 MHz
Max Clocks
    Graphics : 2040 MHz
    SM       : 2040 MHz
    Memory   : 6251 MHz
Performance State: P0
```

`P0` is NVIDIA's highest performance state (P-states run P0=max down to P12=idle) —
confirms the GPU was **not** power-throttled or idle-parked down to a lower clock
during this snapshot. SM clock running at 2025/2040 MHz (99% of max) alongside a
lower observed MFU is more evidence for the earlier bandwidth-bound diagnosis: the
compute cores are running at full clock speed, they just don't have enough
independent work queued to reach high FLOP efficiency at this model's size.

## Clocks Event Reasons — why the GPU isn't at max speed, if it isn't

```
Clocks Event Reasons
    Idle                    : Not Active
    SW Power Cap            : Active       <- currently limited by the power cap
    HW Thermal Slowdown     : Not Active
    HW Power Brake Slowdown : Not Active
    Sync Boost              : Not Active
    SW Thermal Slowdown     : Not Active
```
This section directly tells you *why*, if clocks aren't at maximum: `SW Power Cap`
being `Active` here means the GPU is being held back by its 72W power limit specifically
(not temperature, not a hardware fault) — consistent with a 72W-capped card running
real sustained work. If you ever see real training slow down unexpectedly, this
section is the first place to check for the actual cause.

## PCIe / data-transfer metrics

```
GPU Link Info
    PCIe Generation: Current = 3, Max (device) = 4
    Link Width: Current = 16x, Max = 16x
Tx Throughput: 985888 KB/s   (~963 MB/s, GPU -> host)
Rx Throughput:     585 KB/s   (~0.6 MB/s, host -> GPU)
```
This L4 is running at PCIe Gen3 x16 even though the card supports Gen4 — likely a
platform/motherboard limitation on this particular GCE instance shape, not
something controllable from inside the VM. The Tx/Rx asymmetry (GPU sending far
more than receiving) makes sense mid-training: the GPU is pushing eval outputs/
logging data back to the host far more than it's receiving new input (input tokens
are a tiny fraction of the bandwidth a checkpoint or eval batch consumes).

## ECC / reliability metrics

```
ECC Mode: Current = Enabled
ECC Errors: Volatile/Aggregate, all 0 across SRAM Correctable/Uncorrectable, DRAM
            Correctable/Uncorrectable
Remapped Rows: Correctable/Uncorrectable Error = 0, Pending = No
```
ECC (Error-Correcting Code) memory silently detects and corrects single-bit memory
errors, and flags uncorrectable multi-bit ones. All-zero here across the board is
the healthy, expected state — any nonzero "Uncorrectable" count would mean real
memory corruption risk and justify an instance replacement, not something to train
through.

## Process accounting — confirming who's actually using the GPU

```
Processes
    Process ID   : 4282
    Type         : C                <- "C" = Compute (as opposed to "G" = Graphics)
    Name         : .../custom-gpt-50m/.venv/bin/python
    Used GPU Memory : 1938 MiB
```
This is the ground-truth check that the training process (and only the training
process) is the thing consuming the GPU — useful to confirm nothing else is
accidentally sharing the card, and that the memory-usage number in the top-level
"FB Memory Usage" section is actually explained by a specific, identifiable process
rather than a leak or an orphaned allocation from a previous run.

## Command reference

See `../docs/gpu_sop_guide.md` for the exact commands used to pull these (`nvidia-smi
-q` for the full dump like this doc, `--query-gpu=...` for scriptable CSV subsets,
`watch -n 2 nvidia-smi` for a live view).
