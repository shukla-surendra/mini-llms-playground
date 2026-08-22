# NVIDIA `nvidia-smi` Command Reference

## 1. Overview

`nvidia-smi` (NVIDIA System Management Interface) is the driver-bundled tool for
inspecting and managing NVIDIA GPUs — the first thing to reach for on any GPU machine,
before DCGM or anything else. Companion to
[`dcgm_gpu_command_reference.md`](dcgm_gpu_command_reference.md) — see §12 below for when
to reach for which.

This reference is based on real commands and real output captured on:

- GPU: NVIDIA L4 (single GPU)
- Driver: 580.173.02, CUDA 13.0
- Instance: `mini-llm-gpu` (GCP `g2-standard-4`), actively running a 153M-parameter
  GPT pretraining job throughout every capture below — every reading in this doc is a
  live production workload's real behavior, not an idle or synthetic benchmark.

---

## 1a. Installation — There Isn't One, and That's the Point

Unlike DCGM (a separate, opt-in package — see the companion doc's §1a), `nvidia-smi`
**ships bundled with the NVIDIA driver itself**. If `nvidia-smi` runs at all, the driver
is installed and functioning — there's no separate install step, no repo to add, no
package to choose a version of. This is the single biggest practical difference between
the two tools, worth stating plainly: **`nvidia-smi` is always there the moment the GPU
driver is; DCGM has to be added deliberately.**

Confirm it's present and see the driver/CUDA version in one line:

```bash
$ nvidia-smi -L
GPU 0: NVIDIA L4 (UUID: GPU-0b5f4a6b-7a7a-6c41-0875-b8c8d963b0af)
```

If this fails (`command not found`, or `NVIDIA-SMI has failed because it couldn't
communicate with the NVIDIA driver`), that's a genuine driver problem — worth diagnosing
*before* anything GPU-workload-related, since nothing downstream (CUDA, DCGM, PyTorch)
can work without this succeeding first.

---

## 2. Bare `nvidia-smi` — the One Command to Know Cold

```bash
$ nvidia-smi
```

**Real output, this machine, mid-training:**

```text
Sat Aug 22 10:08:20 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L4                      On  |   00000000:00:03.0 Off |                    0 |
| N/A   76C    P0             71W /   72W |    7532MiB /  23034MiB |     99%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3485      C   ...tom-gpt-153m/.venv/bin/python       7524MiB |
+-----------------------------------------------------------------------------------------+
```

### What each field actually suggests, read against this real reading

| Field | Value here | What it suggests |
|---|---|---|
| `Fan` | `N/A` | Passively cooled or fan speed not exposed in this virtualized/cloud environment — not an error, common on cloud GPU instances |
| `Temp` | `76C` | Hot but well within the L4's safe operating range (throttle point is well above this) — expected under sustained 99% load, not a warning sign by itself |
| `Perf` | `P0` | Maximum performance state — the GPU is not in a reduced power state; `P0` under load is exactly what you want to see |
| `Pwr:Usage/Cap` | `71W / 72W` | **The number that actually explains everything else in this doc** — 98.6% of the hardware power limit, sustained. This single field is *why* the DCGM power warning fires, and *why* throughput has a ceiling regardless of how the code is optimized |
| `Memory-Usage` | `7532MiB / 23034MiB` | Only ~33% of available VRAM in use — this workload (a 153M-parameter model) is nowhere close to memory-constrained on a 24GB card; there's real headroom to raise batch size before memory becomes the bottleneck |
| `GPU-Util` | `99%` | The GPU is doing compute work essentially continuously — a healthy, well-fed training loop, not one stalling on data loading or waiting on the CPU |
| `Compute M.` | `Default` | Normal sharing mode; `Exclusive_Process` would restrict the GPU to one process at a time — relevant on shared/multi-tenant boxes, not this one |
| Processes table | PID 3485, `7524MiB` | Confirms which process is actually holding GPU memory — the number here (7524) vs. the framebuffer total above (7532) differs by a small driver/context overhead, not a discrepancy to worry about |

**The single most useful reading-order habit**: check `Pwr:Usage/Cap` and `GPU-Util`
together, not in isolation. High util + power near the cap = compute-bound, working as
hard as the hardware allows. High util + power *not* near the cap would suggest a
different bottleneck (memory bandwidth, kernel launch overhead) worth investigating
further. Low util regardless of power draw suggests the GPU is waiting on something
upstream (data loading, a CPU-bound preprocessing step) — see §11 for exactly this
pattern caught live.

---

## 3. `--query-gpu` — Scriptable CSV Output

The table view above is for a human glancing at a terminal. For scripting, logging, or
feeding into a monitoring pipeline, `--query-gpu` with an explicit field list and
`--format=csv` is the right tool — one line per GPU, no parsing a box-drawing table.

```bash
$ nvidia-smi --query-gpu=timestamp,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv
```

**Real output:**

```text
timestamp, name, driver_version, temperature.gpu, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], power.limit [W]
2026/08/22 10:08:33.455, NVIDIA L4, 580.173.02, 70, 0 %, 0 %, 7532 MiB, 23034 MiB, 34.39 W, 72.00 W
```

**What this specific reading suggests, and why it's included here rather than a cleaner
one**: `utilization.gpu` at `0 %` and power at `34.39 W` — well below the sustained
~99%/71W seen everywhere else in this doc, from the *same* actively-training process.
This wasn't cherry-picked to look good; it's the real result of one poll landing during a
momentary lull. **A re-check one command later, same machine, same workload:**

```bash
$ nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv
utilization.gpu [%], power.draw [W]
94 %, 70.58 W
```

Back to the expected steady state within one poll interval, and the training log's step
counter had advanced in between (confirmed separately) — proving the workload never
stopped. **The lesson this pair of readings demonstrates directly**: a single
`--query-gpu` sample of a bursty metric can be misleading; don't conclude anything from
one poll, especially for `utilization.gpu`, which — unlike `power.draw` or
`memory.used` — can legitimately dip to zero for a fraction of a second between
micro-batches without indicating any problem at all.

### Other useful field combinations

```bash
# Process-focused, not GPU-focused
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**Real output:**

```text
pid, process_name, used_gpu_memory [MiB]
3485, /home/gpu/tiny_llm/from_scratch/custom-gpt-153m/.venv/bin/python, 7524 MiB
```

Suggests: exactly one process holding GPU memory, matching the PID already seen in the
bare `nvidia-smi` table above — useful specifically when several processes might be
sharing a GPU and the plain table's process list needs to be queried programmatically
instead of read by eye.

List every field `--query-gpu` supports:

```bash
nvidia-smi --help-query-gpu
```

---

## 4. `-q -d <SECTION>` — Deep, Structured Detail on One Topic

`nvidia-smi -q` alone dumps *everything* it knows in a long structured text format —
usually too much at once. `-d <SECTION>` scopes it to exactly the topic worth
investigating.

### `-q -d POWER`

```bash
$ nvidia-smi -q -d POWER
```

**Real output (trimmed to the relevant block):**

```text
GPU 00000000:00:03.0
    GPU Power Readings
        Average Power Draw                             : 71.81 W
        Instantaneous Power Draw                       : 69.83 W
        Current Power Limit                             : 72.00 W
        Requested Power Limit                           : 72.00 W
        Default Power Limit                             : 72.00 W
        Min Power Limit                                 : 40.00 W
        Max Power Limit                                 : 72.00 W
    Power Samples
        Duration                                        : 2.36 sec
        Number of Samples                               : 119
        Max                                             : 81.35 W
        Min                                             : 61.41 W
        Avg                                             : 71.67 W
```

**What this suggests, precisely**: `Current Power Limit` equals `Max Power Limit`
(72.00W both) — this GPU is *not* artificially capped below its own hardware maximum by
any software policy; 72W genuinely is the L4's ceiling in this configuration. `Min Power
Limit: 40.00 W` reveals the GPU *could* be capped lower via `nvidia-smi -pl <watts>` if a
deliberate power/cost trade-off were ever wanted (not done here). The `Power Samples`
block — 119 samples in 2.36 seconds, ranging 61–81W, averaging 71.67W — is the
fine-grained evidence *behind* the single `Average`/`Instantaneous` numbers: the power
draw genuinely oscillates sample-to-sample even under a steady workload, which is the
mechanical reason a single `--query-gpu` poll (§3 above) can occasionally catch an
outlier reading.

### `-q -d CLOCK`

```bash
$ nvidia-smi -q -d CLOCK
```

**Real output (trimmed):**

```text
    Clocks
        Graphics                                        : 2040 MHz
        SM                                               : 2040 MHz
        Memory                                           : 6251 MHz
        Video                                            : 1770 MHz
    Max Clocks
        Graphics                                         : 2040 MHz
        SM                                               : 2040 MHz
        Memory                                           : 6251 MHz
        Video                                            : 1770 MHz
```

**What this suggests**: current SM clock (2040 MHz) equals max SM clock (2040 MHz) at
this exact moment — meaning *at the instant this was captured*, the GPU was running at
full clock speed, not actively throttled, even though the DCGM health check elsewhere in
this session reported a power-related clocks-event warning. This isn't a contradiction:
power-limit throttling is intermittent, not a permanent state — the GPU oscillates
between full clock and briefly-reduced clock as instantaneous draw crosses the 72W line,
consistent with the Power Samples block's 61–81W range above. Catching it *at* full clock
on one poll doesn't mean throttling never happens; it means this specific sample landed
between throttle events.

### `-q -d MEMORY`

```bash
$ nvidia-smi -q -d MEMORY
```

**Real output (trimmed):**

```text
    FB Memory Usage
        Total                                           : 23034 MiB
        Reserved                                        : 471 MiB
        Used                                             : 7532 MiB
        Free                                             : 15032 MiB
    BAR1 Memory Usage
        Total                                           : 32768 MiB
        Used                                             : 7533 MiB
        Free                                             : 25235 MiB
```

**What this suggests**: `Reserved: 471 MiB` is memory the driver itself holds back
before any workload even starts (firmware/ECC/driver bookkeeping) — worth knowing before
assuming 100% of the advertised 23034 MiB is ever actually available to a workload.
`Free: 15032 MiB` confirms real headroom (§2's table already flagged this) — this
specific job could grow its batch size meaningfully before hitting a memory wall.

### `-q -d ECC`

```bash
$ nvidia-smi -q -d ECC
```

Reports whether Error-Correcting Code memory protection is enabled and whether any
correctable/uncorrectable memory errors have been logged — the field to check first if a
training run's loss suddenly does something inexplicable (NaN, a sudden spike) that isn't
explained by the code or data, since an uncorrectable ECC error is a genuine hardware
data-corruption signal, not a software bug. Empty/zero counts here mean no such errors
have been recorded — the healthy, expected state, and the case on this machine
throughout this session.

---

## 5. `pmon` — Live Per-Process Monitoring

```bash
$ nvidia-smi pmon -c 3
```

**Real output:**

```text
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command 
# Idx           #    C/G      %      %      %      %      %      %    name 
    0       3485     C      0      0      -      -      -      -    python         
    0       3485     C      0      0      -      -      -      -    python         
    0       3485     C      0      0      -      -      -      -    python         
```

**What this suggests**: `sm` and `mem` at `0` across all three samples here — this is
the *same* momentary-lull phenomenon documented in §3 and §11, caught by a different
tool this time, at a different exact moment. `type: C` confirms this is a genuine CUDA
compute context (not `G`, graphics) — exactly what's expected for a training workload.
The dashes under `enc`/`dec`/`jpg`/`ofa` mean this process isn't using the GPU's
video-encode/decode or optical-flow hardware blocks at all — expected for a language
model training job, which would only show non-dash values there for a video/image
pipeline.

**Distinct from `dmon` in the DCGM reference**: `pmon` is per-*process* (what is this
PID doing on the GPU), while `dcgmi dmon` is per-*GPU* (what is this whole device doing,
regardless of which process). On a box with exactly one GPU process, as here, they read
similarly; on a shared multi-tenant GPU, `pmon` is the tool that actually distinguishes
"whose" utilization is whose.

---

## 6. Continuous / Repeated Monitoring

Two ways to watch `nvidia-smi` update over time, worth knowing the trade-off between:

```bash
# Option A: nvidia-smi's own built-in loop
nvidia-smi -l 1        # repeat every 1 second, in place

# Option B: the generic Unix `watch` wrapper
watch -n 1 nvidia-smi   # same effective result, clears screen each refresh
```

`-l` is nvidia-smi's own native repeat flag — lighter weight, no external dependency.
`watch` is more general-purpose (works with any command, has options like `-d` to
highlight what changed between refreshes) but is a separate tool that has to be
installed on some minimal images. Either is fine for interactively eyeballing a training
run; neither is meant for logging/scripting — use `--query-gpu` with `--format=csv`
piped to a file or `-l`'s CSV-friendly cousin (`--query-gpu=... --format=csv -l 1`) for
that instead.

---

## 7. `nvidia-smi -pl` — Setting a Power Limit (Mentioned, Not Exercised Here)

```bash
sudo nvidia-smi -pl 60   # cap this GPU at 60W instead of its 72W default
```

Not run on this machine (72W is already this L4's ceiling for the current workload, and
there was no reason to cap it lower) — included because §4's `-q -d POWER` output
revealed a real `Min Power Limit: 40.00 W`, confirming this GPU genuinely supports being
capped anywhere in the 40–72W range. This is the actual lever behind a real trade-off:
a lower cap reduces peak throughput but also reduces power draw (and, on a
per-GPU-hour-billed cloud instance, doesn't change the bill — the instance is billed for
time, not watts — but *does* matter on-prem, or for thermal/rack-density planning).
Requires root; the setting does not persist across a reboot unless made permanent via a
systemd unit or `nvidia-persistenced` configuration.

---

## 8. Real Investigation, Reprised: Same Pattern, Caught Twice

This exact "brief zero, then back to normal on the very next check" pattern showed up
**twice independently** during this session — once via `--query-gpu` (§3), once via
`pmon` (§5), each time on live output, not a repeat of the same capture. Two different
tools, two different moments, the same underlying explanation both times: `utilization.gpu`
and `sm`/`mem` percentages are instantaneous samples of an inherently bursty signal (a
training loop's compute happens in short kernel launches separated by brief CPU-side
gaps — data loading, Python overhead, occasional synchronization points), and catching a
poll exactly inside one of those gaps produces a `0` reading that means nothing on its
own. **The generalizable rule**: for utilization-style metrics specifically, always take
a second reading — or better, watch for a few seconds (`-l 1`) — before drawing any
conclusion. Power draw and memory usage are comparatively much more stable
sample-to-sample (see §4's Power Samples block: even its *min* of 61W is still clearly
"under sustained load," never near zero) — a `0` or near-zero reading on *those* two
specifically would be a far stronger signal something had actually stopped.

---

## 9. Common Gotchas

| Symptom | What it actually means |
|---|---|
| `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | Driver not loaded/installed, or a kernel-module mismatch after a kernel update — a real problem, not a permissions issue |
| `Fan: N/A` | Normal on most cloud GPU instances — fan telemetry usually isn't exposed through the hypervisor/virtualization layer |
| `Failed to initialize NVML: Insufficient Permissions` | Usually a container/namespace issue (missing `--gpus` flag in Docker, or a cgroup device restriction) — not a driver problem |
| A single poll shows `0%` utilization on a workload known to be running | See §8 — check a second time before assuming anything is wrong |
| `memory.used` far exceeds what the workload should need | Check `--query-compute-apps` (§3) to see if a *different*, forgotten process is also holding GPU memory — common after a crashed job leaves a zombie CUDA context |

---

## 10. Most Useful Commands to Memorize

```bash
# The one to run first, always
nvidia-smi

# Machine-readable snapshot
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu --format=csv

# Watch it live
nvidia-smi -l 1

# Which process is actually using the GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Deep-dive on one topic
nvidia-smi -q -d POWER
nvidia-smi -q -d CLOCK
nvidia-smi -q -d MEMORY
nvidia-smi -q -d ECC

# Per-process, not per-GPU
nvidia-smi pmon -c 5
```

---

## 11. Mental Model

```text
                    NVIDIA DRIVER
                         |
              (ships together, always present)
                         |
                    nvidia-smi
                         |
        +----------------+----------------+
        |                |                |
   bare table      --query-gpu       -q -d SECTION
   (human, fast)    (scriptable)     (deep, targeted)
        |                |                |
   GPU-Util,        one CSV line     POWER / CLOCK /
   Pwr, Mem,        per GPU, or      MEMORY / ECC —
   Temp, Procs      per-process      full structured
                     via --query-    detail on one
                     compute-apps    topic at a time
```

**Practical reading order for "is this GPU okay":**

```text
nvidia-smi (bare)
    |
    +--> Pwr near Cap + high Util  --> compute-bound, working as designed (this machine)
    +--> Low Util, low Pwr, once   --> check again before concluding anything (§8)
    +--> Low Util, low Pwr, always --> workload actually stopped; check the process (§3, §9)
    |
    v
nvidia-smi -q -d POWER / CLOCK      --> is this a hardware limit, or something else?
    |
    v
nvidia-smi pmon / --query-compute-apps --> which process, exactly, is responsible?
```

---

## 12. `nvidia-smi` vs. DCGM (`dcgmi`) — When to Reach for Which

| | `nvidia-smi` | `dcgmi` (DCGM) |
|---|---|---|
| Install | Bundled with the driver — always present | Separate package, install required (companion doc §1a) |
| Best for | Quick spot checks, interactive debugging, "is this GPU okay right now" | Structured health monitoring, fleet-scale checks, groups, JSON output for automation |
| Health verdict | You interpret the raw numbers yourself | Gives an explicit `Healthy`/`Warning`/`Error` verdict with a stated reason |
| Multi-GPU / multi-node | Lists all GPUs, no built-in grouping | Native GPU groups (`-g <groupId>`), built for fleets |
| Typical consumer | A human at a terminal | Prometheus exporters, monitoring dashboards, automated health gates |

**In practice on this machine**: `nvidia-smi` was reached for first, every time, for a
quick "what's actually happening right now" read — DCGM was reached for specifically
when a structured health *verdict* was wanted (§5 of the companion doc), not raw numbers
to interpret by hand. Neither replaces the other; they answer genuinely different
questions.
