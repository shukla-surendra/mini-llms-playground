# GPU node — quick operational reference

Companion to `training_sop.md` (which covers provisioning/teardown) — this doc is
just the commands you actually reach for while a training run is live on the box.
Every command below was run for real during the 2026-08-18 `custom-gpt-50m` resume
session, not written speculatively. IP/instance name below match that session
(`0.0.0.0`, instance `mini-llm-gpu`) — get current values via `make status`
or `terraform output public_ip` if they've changed since.

## Connecting

Direct SSH:
```bash
ssh -i ~/.ssh/id_ed25519 gpu@0.0.0.0
```

Via `make` (reads the IP from Terraform output automatically — use this once the IP
has changed and you don't want to update commands by hand):
```bash
make ssh          # direct
make iap-ssh       # via Identity-Aware Proxy tunnel — works even if allowed_ssh_cidrs
                   # is stale (ISP rotated your IP) or the direct route has issues
```

## tmux — attach / detach

Training runs inside a detached `tmux` session named `train` so it survives your
SSH connection dropping. Once SSH'd in:

**Attach** (reconnect to the live session, see real-time output):
```bash
tmux attach -t train
```

**Detach** (leave it running, return to your own shell) — press, don't type:
```
Ctrl-b   then   d
```
(release `Ctrl-b` first, *then* press `d` — it's two separate keypresses, not held
together). You'll drop back to your normal SSH shell; training keeps running
untouched in the background.

**List sessions** (confirm `train` is actually still alive):
```bash
tmux ls
```

**If you get disconnected without detaching cleanly** (closed terminal, network
drop): just SSH back in and `tmux attach -t train` again — the session survives an
ungraceful client disconnect, that's the entire point of running it inside tmux.

## GPU metrics

**Full snapshot** — utilization, memory, power, clocks, p-state, one line:
```bash
ssh -i ~/.ssh/id_ed25519 gpu@0.0.0.0 \
  'nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,pstate --format=csv'
```

Real example output from the live run:
```
NVIDIA L4, 82 %, 70 %, 1946 MiB, 23034 MiB, 79, 69.11 W, 72.00 W, 1650 MHz, 2040 MHz, 6251 MHz, 6251 MHz, P0
```
Read as: 82% of SMs busy, 70% of memory-bandwidth capacity in use, ~1.9GB/23GB VRAM,
79°C, drawing 69W of a 72W cap (essentially power-limited, not idle), SM clock
1650/2040MHz max, P0 = highest performance state (not throttled down).

**Live-refreshing view** (Ctrl-C to exit) — best run from inside the SSH session
itself, not piped through a one-shot `ssh '...'` command:
```bash
watch -n 2 nvidia-smi
```

**Minimal one-liner**, just the numbers that matter most day-to-day:
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
```

## Training progress

**Latest step/loss/speed line** (the tqdm progress line, `\r`-terminated so it needs
translating to see with `tail`):
```bash
ssh -i ~/.ssh/id_ed25519 gpu@0.0.0.0 \
  'tail -c 2000 /home/gpu/train_stdout.log | tr "\r" "\n" | tail -3'
```

**Confirm the run's actual device/architecture** (the one-time banner printed at
startup, not repeated per-step — grep for it specifically rather than scrolling):
```bash
ssh -i ~/.ssh/id_ed25519 gpu@0.0.0.0 \
  'grep -m1 "Precision:\|Model:" /home/gpu/train_stdout.log'
```

**Follow the log live** (matches what you'd see attached to tmux, but read-only,
doesn't require attaching):
```bash
ssh -i ~/.ssh/id_ed25519 gpu@0.0.0.0 'tail -f /home/gpu/train_stdout.log'
```

## Reading steps/sec honestly

`steps/sec` is **not** directly comparable across runs with different `batch_size`/
`grad_accum_steps` — each "step" does a different amount of real work depending on
batch size. Convert to **tokens/sec** (`steps/sec × batch_size × context_length`)
before comparing two runs' actual throughput. Real example from this session: a
`batch=1` run showing ~36 steps/sec (~36,900 tok/s) is *slower* in real throughput
than a `batch=4` run showing 13.3 steps/sec (~54,500 tok/s) — the smaller-batch run
just completes more (smaller) steps per second, which looks faster at a glance but
isn't.
