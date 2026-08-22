# NVIDIA DCGM Command Reference

## 1. Overview

NVIDIA DCGM (Data Center GPU Manager) is used to discover, monitor, and health-check NVIDIA GPUs.
Companion to [`nvidia_smi_command_reference.md`](nvidia_smi_command_reference.md) — that
doc covers the driver-bundled tool that needs no install and is the first thing to check;
this one covers the separately-installed, structured-health-verdict tool. See §12 of the
`nvidia-smi` doc for exactly when to reach for which.

This reference is based on the commands and output observed on a machine with:

- GPU: NVIDIA L4
- GPU count: 1
- DCGM CLI: `dcgmi`

---

## 1a. Installation — What We Actually Did on This Machine

DCGM is **not installed by default** on this machine's image
(`common-cu129-ubuntu-2204-nvidia-580`, a GCP `ml-images` Deep-Learning-style VM image).
Before installing, both of these failed — correctly, not a naming mistake:

```bash
$ systemctl status nvidia-dcgm
Unit nvidia-dcgm.service could not be found.
$ systemctl status dcgm
Unit dcgm.service could not be found.
```

Confirmed nothing was present at all before installing:

```bash
$ which dcgmi nv-hostengine
# (no output — neither binary exists)
$ dpkg -l | grep -i dcgm
# (no output — nothing installed)
$ apt-cache search dcgm
libnvidia-nscq-450 - NVSwitch Configuration and Query library
libnvidia-nscq-565 - NVSwitch Configuration and Query library
... (only unrelated NVSwitch libraries — the real DCGM package isn't in any
     configured repo yet)
```

**Why**: the image ships the NVIDIA driver, CUDA toolkit, and two basic daemons
(`nvidia-persistenced`, running; `nvidia-fabricmanager.service`, present but
`failed` — expected and harmless, that daemon manages NVLink/NVSwitch fabric, which a
single L4 doesn't have) — but DCGM itself is a separate, opt-in NVIDIA product, not
bundled by default.

### Step 1 — Add NVIDIA's CUDA APT repo (not configured on this image)

```bash
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -qq
```

After this, `apt-cache search dcgm` reveals two separate package families:

```text
datacenter-gpu-manager                        - NVIDIA® Datacenter GPU Management Tools   (legacy, version 3.3.9)
datacenter-gpu-manager-4-core                  - CUDA-version agnostic components of DCGM  (current, version 4.x)
datacenter-gpu-manager-4-cuda11 / -cuda12 / -cuda13
datacenter-gpu-manager-4-cuda-all              - Metapackage pulling in all CUDA variants
datacenter-gpu-manager-4-dev
datacenter-gpu-manager-4-multinode(-cuda12/-cuda13)
datacenter-gpu-manager-4-proprietary(-cuda11/-cuda12/-cuda13)
```

### Step 2 — Choose DCGM 4.x, matched to this box's CUDA version, not the legacy 3.3.9 package

This machine runs CUDA 12 (per the image name), so the right package pair is the
`-4-core` + `-4-cuda12` combination — not the `-cuda-all` metapackage (which would
pull in CUDA 11 and 13 binaries this box will never use) and not the older
single-package `datacenter-gpu-manager` (3.3.9, still available in the same repo for
compatibility, but not the current release line):

```bash
sudo apt-get install -y datacenter-gpu-manager-4-core datacenter-gpu-manager-4-cuda12
```

Real install output (trimmed):

```text
Selecting previously unselected package datacenter-gpu-manager-4-cuda12.
Unpacking datacenter-gpu-manager-4-cuda12 (1:4.6.1-1) ...
Selecting previously unselected package datacenter-gpu-manager-4-proprietary.
Unpacking datacenter-gpu-manager-4-proprietary (1:4.6.1-1) ...
Selecting previously unselected package datacenter-gpu-manager-4-proprietary-cuda12.
Unpacking datacenter-gpu-manager-4-proprietary-cuda12 (1:4.6.1-1) ...
Setting up datacenter-gpu-manager-4-core (1:4.6.1-1) ...
Setting up datacenter-gpu-manager-4-proprietary (1:4.6.1-1) ...
Setting up datacenter-gpu-manager-4-cuda12 (1:4.6.1-1) ...
Setting up datacenter-gpu-manager-4-proprietary-cuda12 (1:4.6.1-1) ...
```

**Version actually installed: DCGM 4.6.1.**

### Step 3 — Enable and start the service

```bash
sudo systemctl enable --now nvidia-dcgm
```

Real output — worth reading closely, it does something not obvious from the command
alone:

```text
Created symlink /etc/systemd/system/dcgm.service → /lib/systemd/system/nvidia-dcgm.service.
Created symlink /etc/systemd/system/multi-user.target.wants/nvidia-dcgm.service → /lib/systemd/system/nvidia-dcgm.service.
● nvidia-dcgm.service - NVIDIA DCGM service
     Loaded: loaded (/lib/systemd/system/nvidia-dcgm.service; enabled; vendor preset: enabled)
     Active: active (running) since Sat 2026-08-22 09:44:08 UTC; 3s ago
   Main PID: 7419 (nv-hostengine)
      Tasks: 17 (limit: 19114)
     Memory: 18.2M
        CPU: 78ms
     CGroup: /system.slice/nvidia-dcgm.service
             └─7419 /usr/bin/nv-hostengine -n --service-account nvidia-dcgm

Aug 22 09:44:08 mini-llm-gpu systemd[1]: Started NVIDIA DCGM service.
Aug 22 09:44:08 mini-llm-gpu nv-hostengine[7419]: DCGM initialized
Aug 22 09:44:08 mini-llm-gpu nv-hostengine[7419]: Started host engine version 4.6.1 using port number: 5555
```

**Key finding**: enabling `nvidia-dcgm` automatically creates a `dcgm.service` symlink
alias to it. This means **both of the original failing commands now work**:

```bash
$ systemctl status dcgm --no-pager        # works now, via the symlink
$ systemctl status nvidia-dcgm --no-pager # works now, the real unit
```

### Step 4 — Verify it works, and works with L4 specifically

Confirmed both discovery and live telemetry, immediately after starting the service
— see §2 and §11 below for the exact commands and their real output. Both were run
**while a real training workload was actively running on the GPU** (a 153M-parameter
GPT pretraining run), with zero disruption — this was a pure `apt install` + service
start, no driver change, no reboot.

---

## 2. GPU Discovery

### List GPUs detected by DCGM

```bash
dcgmi discovery -l
```

Useful for confirming that DCGM can see the GPU.

**Real output, this machine** (run with `sudo`, immediately after the install above):

```text
$ sudo dcgmi discovery -l
1 GPU found (Active).
+--------+----------------------------------------------------------------------+
| GPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
| 0      | Name: NVIDIA L4                                                      |
|        | PCI Bus ID: 00000000:00:03.0                                         |
|        | Device UUID: GPU-0b5f4a6b-7a7a-6c41-0875-b8c8d963b0af                |
+--------+----------------------------------------------------------------------+
0 NvSwitches found.
+-----------+
| Switch ID |
+-----------+
+-----------+
0 ConnectX found.
+----------+
| ConnectX |
+----------+
+----------+
0 CPUs found.
+--------+----------------------------------------------------------------------+
| CPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
+--------+----------------------------------------------------------------------+
```

**`0 NvSwitches`, `0 ConnectX`, `0 CPUs found` are all correct, not errors** — this is a
single-GPU cloud VM (GCP `g2-standard-4`, one L4), not a DGX-style multi-GPU node with
NVLink/NVSwitch fabric or InfiniBand (ConnectX) interconnects. On hardware that actually
has those components, they'd be enumerated here instead of showing empty tables.

### What to check

- Number of GPUs found
- GPU state (`Active`)
- GPU model
- PCI Bus ID
- Device UUID

---

# 3. Health Monitoring

DCGM health monitoring works with **health watches**. You must enable watches before running a health check.

## Show health command help

```bash
dcgmi health --help
```

Important options:

| Option | Meaning |
|---|---|
| `-s <flags>` | Set health watches |
| `-f` | Fetch current watch status |
| `-c` | Check health |
| `--clear` | Disable all watches |
| `-g <groupId>` | Use a specific GPU group |
| `-j` | JSON output |
| `-m <seconds>` | Maximum sample cache age |
| `-u <seconds>` | Update interval |

---

## 4. Enable Health Watches

### Enable all health watches

```bash
dcgmi health -s a
```

Expected result:

```text
Health monitor systems set successfully.
```

### Enable specific watches

The available flags are:

| Flag | Monitor |
|---|---|
| `a` | All watches |
| `d` | Driver |
| `i` | InfoROM |
| `m` | Memory |
| `n` | NVLink |
| `p` | PCIe |
| `t` | Thermal and power |
| `x` | ConnectX |

Examples:

```bash
dcgmi health -s d
```

Driver health.

```bash
dcgmi health -s m
```

Memory health.

```bash
dcgmi health -s t
```

Thermal and power health.

```bash
dcgmi health -s p
```

PCIe health.

### Multiple watches

Depending on the DCGM version, multiple watch flags can be supplied together.

For example:

```bash
dcgmi health -s dmtp
```

This enables driver, memory, thermal/power, and PCIe watches.

For the simplest setup, use:

```bash
dcgmi health -s a
```

---

# 5. Check GPU Health

After enabling watches:

```bash
dcgmi health -c
```

This checks for errors and warnings detected by the currently enabled watches.

**Real output, this machine, exact terminal formatting**:

```text
$ dcgmi health -c
+---------------------------+----------------------------------------------------------+
| Health Monitor Report                                                                |
+===========================+==========================================================+
| Overall Health            | Warning                                                  |
| GPU                       |                                                          |
| -> 0                      | Warning                                                  |
|    -> Errors              |                                                          |
|       -> Power system     | Warning                                                  |
|                           | Detected clocks event due to power violation in GPU 0.   |
|                           | Monitor the power conditions. This GPU can still         |
|                           | perform workload.                                        |
+---------------------------+----------------------------------------------------------+
```

Confirmed **reproducible, not a one-off glitch** — ran twice, several minutes apart,
identical result both times. That repeatability is itself diagnostic information: a
transient event would eventually clear on a re-check; this stayed present because the
underlying condition (sustained near-power-limit draw under 100% utilization) was
still true both times.

### Important

A `Warning` does **not necessarily mean the GPU is unusable**.

In the observed output, DCGM specifically reported:

```text
Detected clocks event due to power violation in GPU 0.
This GPU can still perform workload.
```

This means DCGM detected a power-related clock event and recommends monitoring the power conditions.

---

# 6. Fetch Current Health Watch Configuration

```bash
dcgmi health -f
```

This is useful for checking which health watches are currently configured.

---

# 7. Clear Health Watches

To disable all currently monitored health watches:

```bash
dcgmi health --clear
```

After clearing watches, a health check may report:

```text
Health watches not enabled.
Please enable watches.
```

Enable them again with:

```bash
dcgmi health -s a
```

---

# 8. JSON Output

Many DCGM commands support JSON output.

For health:

```bash
dcgmi health -c -j
```

This is useful for scripts, automation, and monitoring systems.

You can also combine JSON with other health operations where supported:

```bash
dcgmi health -f -j
```

---

# 9. GPU Groups

DCGM can operate on GPU groups.

The health command supports:

```bash
dcgmi health -g <groupId> -c
```

Example:

```bash
dcgmi health -g 1 -c
```

This checks the health of GPUs belonging to group `1`.

For a simple single-GPU machine, you may not need to specify a group unless your DCGM setup requires it.

---

# 10. Health Watch Timing

Some health watches require time before their first useful query.

DCGM reports that these watches require approximately **60 seconds** before the first query:

- Memory
- NVLink
- PCIe
- Thermal and power
- ConnectX

Therefore, after:

```bash
dcgmi health -s a
```

wait about 60 seconds before relying on the first health result.

A practical sequence is:

```bash
dcgmi health -s a
sleep 60
dcgmi health -c
```

---

# 11. Live GPU Monitoring

DCGM also provides `dmon` for monitoring GPU metrics.

```bash
dcgmi dmon
```

This is useful when you want to watch GPU activity while a workload is running. Bare
`dcgmi dmon` streams continuously and picks a default metric set — for a bounded,
scriptable sample instead, name the exact field IDs and a sample count with `-e` / `-c`.

**Real output, this machine, run while a 153M-parameter training job was actively
running**:

```text
$ sudo dcgmi dmon -e 203,204,252 -c 5
#Entity         GPUTL             MCUTL             FBUSD
ID
GPU 0           97                64                7532
GPU 0           97                64                7532
GPU 0           100               68                7532
GPU 0           100               67                7532
GPU 0           100               69                7532
```

| Field ID | Column | Meaning |
|---|---|---|
| 203 | `GPUTL` | GPU (SM) utilization, % |
| 204 | `MCUTL` | Memory controller utilization, % |
| 252 | `FBUSD` | Framebuffer (VRAM) used, MiB |

**Cross-checked against `nvidia-smi` at the same moment** — 7532 MiB matched
`nvidia-smi`'s own memory-used reading exactly, and the 97–100% utilization matched
`nvidia-smi`'s utilization figure too. Two independent tools reading the same live
state, in agreement — useful confirmation that DCGM is reporting real values, not a
stale or default sample.

Full field-ID list for `-e` (there are dozens; a few of the most useful beyond the
three above):

| Field ID | Name | Meaning |
|---|---|---|
| 150 | `SMCLK` | SM clock speed, MHz |
| 155 | `POWER` | Power usage, W |
| 156 | `PSTATE` | Performance state |
| 140 | `TMPR` | GPU temperature, °C |
| 251 | `FBFRE` | Framebuffer free, MiB |
| 1002 | `PCITX` / `PCIRX` | PCIe throughput |

List every available field on this DCGM version with:

```bash
dcgmi dmon --list
```

---

# 12. Standard NVIDIA GPU Status

DCGM is not the only useful GPU diagnostic tool.

Use:

```bash
nvidia-smi
```

for a quick overview of:

- GPU utilization
- GPU memory usage
- GPU temperature
- Power usage
- Running GPU processes
- Driver information

For a continuously refreshed view:

```bash
watch -n 1 nvidia-smi
```

This refreshes `nvidia-smi` every second.

---

# 13. Useful Log Monitoring

If a GPU workload writes to a log file:

```bash
tail -n 50 -f /tmp/gpt-train.log
```

This:

1. Shows the last 50 lines.
2. Continues following the file.
3. Displays new lines as they are written.

Stop with:

```text
Ctrl+C
```

### Search errors while following

```bash
tail -f /tmp/gpt-train.log | grep --line-buffered -i "error"
```

### Search errors and warnings

```bash
tail -f /tmp/gpt-train.log | grep --line-buffered -Ei "error|warn"
```

---

# 14. Quick Diagnostic Sequence

For a GPU machine, a useful basic diagnostic sequence is:

```bash
# 1. Check that Linux/NVIDIA sees the GPU
nvidia-smi

# 2. Check that DCGM sees the GPU
dcgmi discovery -l

# 3. Enable all DCGM health watches
dcgmi health -s a

# 4. Check the configured watches
dcgmi health -f

# 5. Wait for watches that need initial samples
sleep 60

# 6. Check GPU health
dcgmi health -c

# 7. Live DCGM monitoring
dcgmi dmon
```

---

# 15. Investigating a Power Warning

If you see:

```text
Power system | Warning
Detected clocks event due to power violation
```

do not immediately assume the GPU is broken.

First check:

```bash
nvidia-smi
```

Then inspect power-related information:

```bash
nvidia-smi -q -d POWER
```

Check:

- Current power usage
- Power limit
- Default power limit
- Maximum power limit
- Power management information

You can also watch the GPU continuously:

```bash
watch -n 1 nvidia-smi
```

And monitor DCGM:

```bash
dcgmi dmon
```

The goal is to determine whether the warning is:

- A historical/transient event
- A recurring power-limit condition
- A workload-induced power constraint
- A hardware/platform power issue

---

# 16. Common Errors

## Error: Health watches not enabled

```text
Error: Health watches not enabled. Please enable watches.
```

Fix:

```bash
dcgmi health -s a
```

Then:

```bash
dcgmi health -c
```

---

## Error: Missing value for `-s`

If you run:

```bash
dcgmi health -s
```

you will get a parse error because `-s` requires a value.

Correct:

```bash
dcgmi health -s a
```

Here:

- `-s` = set watches
- `a` = all watches

---

# 17. Most Useful Commands to Memorize

### Discovery

```bash
dcgmi discovery -l
```

### Health

```bash
dcgmi health -s a
dcgmi health -f
dcgmi health -c
dcgmi health --clear
```

### Live DCGM monitoring

```bash
dcgmi dmon
```

### NVIDIA status

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Power details

```bash
nvidia-smi -q -d POWER
```

### Log monitoring

```bash
tail -n 50 -f /tmp/gpt-train.log
```

### Error filtering

```bash
tail -f /tmp/gpt-train.log | grep --line-buffered -i "error"
```

---

# 18. Mental Model

Think of the tools like this:

```text
                    GPU MACHINE
                         |
          +--------------+--------------+
          |                             |
      NVIDIA Driver                   DCGM
          |                             |
      nvidia-smi                 dcgmi discovery
          |                      dcgmi health
          |                      dcgmi dmon
          |
      Basic GPU status
                                    |
                         Health + monitoring
```

A practical troubleshooting flow is:

```text
GPU problem
    |
    v
nvidia-smi
    |
    +--> GPU visible?
    |       |
    |       +--> No -> investigate driver/hardware
    |
    v
dcgmi discovery -l
    |
    +--> GPU Active?
    |
    v
dcgmi health -s a
    |
    v
dcgmi health -c
    |
    +--> Warning/Error?
    |       |
    |       v
    |   nvidia-smi -q
    |   nvidia-smi -q -d POWER
    |   dcgmi dmon
    |
    v
Monitor workload
```

# 19. Your Current Machine

Based on the output you provided:

```text
GPU: NVIDIA L4
GPU ID: 0
State: Active
PCI Bus ID: 00000000:00:03.0
```

DCGM successfully discovers the GPU.

You also successfully enabled all health watches:

```bash
dcgmi health -s a
```

The current health result is:

```text
Overall Health: Warning
GPU 0: Warning
Reason: Power system
Detected clocks event due to power violation
```

The message explicitly says the GPU **can still perform workload**, but the power condition should be monitored.

The next commands I would run are:

```bash
nvidia-smi
nvidia-smi -q -d POWER
dcgmi dmon
```

These will help determine whether the power warning is currently occurring or was a previous/transient event.

---

# 20. Real Investigation Session — Power Warning, Triaged Live

This is what actually ran, in order, to answer "is this power warning something to worry
about" for real — following the exact checklist §15 already lays out in the abstract, now
with each step's real command and real output, including a false alarm along the way that's
worth knowing how to recognize.

### Step 1 — Cross-check the warning against raw power numbers

```bash
$ nvidia-smi --query-gpu=power.draw,power.limit,clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu --format=csv
power.draw [W], power.limit [W], clocks.current.sm [MHz], clocks.max.sm [MHz], temperature.gpu, utilization.gpu [%]
34.19 W, 72.00 W, 2040 MHz, 2040 MHz, 71, 0 %
```

**This looked like a red flag at first glance**: `utilization.gpu` at `0 %` and power
well under the 72W limit, right after having just been told the GPU was under a power
warning. Read naively, this could mean the workload had stopped.

### Step 2 — Don't trust one sample; check the actual process and the log

```bash
$ ps -ef | grep -i gpt.train | grep -v grep
gpu         3482    3413  0 08:55 pts/0    00:00:00 uv run gpt-train
gpu         3485    3482 88 08:55 pts/0    00:48:12 /home/gpu/.../python .../gpt-train
```

The `88` in the CPU-usage column confirmed the process was actively burning CPU — not
dead, not hung. The tmux pane itself (`tmux capture-pane -t train -p`) came back blank,
which looked alarming in isolation but had a mundane explanation: training's own output
was redirected straight to a log file at launch (`... > /tmp/gpt-train.log 2>&1`), so the
terminal pane was *never* going to show live text — checking the pane was the wrong place
to look, not evidence of a problem.

```bash
$ tail -c 1500 /tmp/gpt-train.log | tr "\r" "\n" | tail -5
training:  38%|███▊      | 48157/127933 [54:59<3:42:47,  5.97step/s, batch_loss=6.0568,
  epoch1_eta_h=209.2, est_epoch=0.067, eta_h=25.0, lr=2.95e-04, test_loss=6.2514,
  test_ppl=518.8, total_h=15.11, train_loss=6.1062]
```

Step count was climbing (32,048 → 48,157 since launch), `test_loss` was falling
(7.166 → 6.251) — genuine, healthy progress. Training was never stopped.

### Step 3 — Take a fresh reading

```bash
$ nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu,temperature.gpu --format=csv
power.draw [W], power.limit [W], utilization.gpu [%], temperature.gpu
70.93 W, 72.00 W, 100 %, 83
```

**This is the real, sustained steady state**: 98.5% of the 72W power limit, 100%
utilization, 83°C. The Step-1 reading was a one-sample artifact — almost certainly caught
during a brief lull (a data-loading gap or an evaluation-phase pause between training
microsteps), not a real dip. This is exactly why the diagnostic flow in §15/§18 says to
*monitor*, not trust a single point-in-time sample, before drawing a conclusion.

### Conclusion

The DCGM "Power system: Warning — clocks event due to power violation" message is an
**accurate, expected, and benign** report of steady-state behavior for this workload on
this GPU: a training job driving 100% utilization will run right up against the L4's
72W power cap, and the driver's clock-throttling response to that is a designed
protection mechanism, not a fault — DCGM's own message says so explicitly ("can still
perform workload"). The practical consequence is a throughput ceiling (this box tops
out around 5.9-6 steps/sec on this model, power-limited rather than compute-limited),
not a reliability concern. **Nothing was fixed because nothing was broken** — the value
of this session was confirming that with real, cross-checked evidence instead of
reacting to the warning label alone.

---

# 21. Lessons for Reading Any DCGM Warning

Generalizing from the session above, into a repeatable checklist for the *next* warning
this reference is used to triage:

1. **A DCGM "Warning" is not automatically an incident.** Read the actual explanatory
   text DCGM prints under the category — it usually states plainly whether the GPU can
   still do useful work.
2. **Never conclude from one sample.** A single `nvidia-smi` snapshot can catch a
   momentary lull that looks alarming in isolation; take a second reading before
   deciding anything, especially for a metric (utilization, power) that's naturally
   bursty rather than constant.
3. **Cross-check DCGM against `nvidia-smi` and the actual workload's own log/process
   state.** Three independent signals agreeing (DCGM's field values, `nvidia-smi`'s
   own numbers, and the workload's log showing forward progress) is much stronger
   evidence than any one of them alone.
4. **Know what "no output" actually means before treating it as a symptom.** A blank
   tmux pane, an empty `tail`, or a quiet log can mean "broken" — or it can mean
   "output was redirected somewhere else entirely," which is a five-second check
   (`ps -ef`, check the actual log path) away from a wrong conclusion.
5. **A recurring, reproducible warning under sustained near-100% utilization is usually
   the GPU's hardware power limit, not a defect** — check `power.draw` against
   `power.limit` before assuming anything is wrong with the card itself.
