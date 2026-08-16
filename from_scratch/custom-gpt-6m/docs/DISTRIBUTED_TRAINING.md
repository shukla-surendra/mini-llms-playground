# Distributed Training: DDP and FSDP

## The mechanism, if you need a refresher

`src/gpt/training/trainer_ddp.py` and `src/gpt/training/trainer_fsdp.py` wrap the exact same causal-LM model and data
`trainer.py` already trains — same `src/gpt/model.py`, same tokenized data — adding only the
distributed-training machinery on top. If the general concept (why distribute training at
all, what a process group is, what all-reduce/all-gather actually do) is unfamiliar, the
first-principles treatment is
[`../../../docs/llm-engineering/26_distributed_training_ddp_and_fsdp.md`](../../../docs/llm-engineering/26_distributed_training_ddp_and_fsdp.md) —
this doc covers only this project's specific implementation and real, observed numbers.

## Hardware honesty, upfront

This machine is Apple Silicon (MPS), no CUDA, no multiple real GPUs. Both scripts run
multiple **CPU processes on one machine**, communicating via the `gloo` backend (the one
`torch.distributed` backend that supports CPU tensors — `nccl`, the production choice,
requires CUDA GPUs). Everything below is a genuine, real execution of the DDP/FSDP
mechanism and API — the gradient sync, the parameter sharding, the process-group
coordination all really happen — but it is **not** a scaling benchmark. Real production
use would be `nccl` + one process per GPU, across GPUs with far more compute and
inter-GPU bandwidth (NVLink) than this Mac's CPU cores communicating over `gloo`. Nothing
below should be read as "here's the speedup DDP/FSDP gives" — it's "here's proof the
mechanism works, run for real."

## Why two launch paths exist, and a real bug this project hit

The standard, documented way to launch either script is `torchrun`:

```bash
torchrun --nproc_per_node=2 -m gpt.training.trainer_ddp
```

**This hung indefinitely in this project's actual development environment** — not a
hypothetical, a real thing that happened while building this. `torchrun`'s elastic-agent
rendezvous does a reverse-DNS lookup as part of its startup handshake
(`[c10d] The IPv6 network addresses of (...ip6.arpa) cannot be retrieved`, repeating
forever), and that lookup never resolved in this specific sandboxed environment, so the
two worker processes were never even spawned. Both scripts therefore default to a second,
tested-working launch path — `gpt-train-ddp` directly, no external launcher — using
`torch.multiprocessing.spawn` internally with an explicit `tcp://127.0.0.1:PORT` init
method, which sets up the identical process group without that DNS lookup:

```python
if "RANK" in os.environ:
    _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), ...)  # torchrun path
else:
    mp.spawn(_worker, args=(world_size, ...), nprocs=world_size, join=True)  # this project's default
```

`torchrun` is expected to work normally as the production launch method on a non-sandboxed
machine — this project just can't rely on it as the *tested* path given what was actually
observed. `make train-ddp` / `make train-fsdp` use the `mp.spawn` path.

## A second real bug: FSDP auto-detects MPS as its accelerator, even for a CPU-only run

`FSDP(model)` alone, on this machine, crashes with
`AttributeError: Custom backend 'mps' not implement 'torch.mps.current_device'` — a real
error hit running this exact line during development. The cause: when FSDP can't
determine a compute device from the model's own parameters (they're all on CPU), it falls
back to `torch._C._get_accelerator()`, which returns `"mps"` on any Apple Silicon Mac
**regardless of what device is actually being used for this run** — and `torch.mps`
doesn't implement the full CUDA-like device-handle interface (`current_device()`) FSDP
expects from an accelerator backend. The fix, in `src/gpt/training/trainer_fsdp.py`:

```python
fsdp_model = FSDP(model, device_id=torch.device("cpu"))
```

Passing `device_id` explicitly skips FSDP's auto-detection entirely. Worth knowing as a
general lesson, not just a one-off fix: FSDP's device-handling code assumes "an
accelerator is present" implies "use it," an assumption that breaks specifically on
Apple Silicon's partial `torch.mps` API surface — this would not happen on a CUDA machine,
where `torch.cuda` fully implements the interface FSDP expects.

## Real training run, actually executed on this project's own MacBook (two CPU processes, gloo)

`STEPS=150`, `BATCH_SIZE=16` (per rank), default model size, same tokenized data as every
other script in this project — DDP and FSDP run back to back for a direct comparison:

| | DDP | FSDP |
|---|---|---|
| `[model]` parameters printed after wrapping | 5,853,184 (**replicated** — full model, every rank) | 2,926,592 (**sharded** — exactly half, this rank's shard) |
| val_loss: step 0 → 50 → 100 → 150 | 8.321 → 5.783 → 5.018 → 4.568 | 8.360 → 5.799 → 5.041 → 4.591 |
| Wall-clock, 150 steps (2 processes) | 68.7s | 69.8s |

**Reading these numbers**: the parameter count line is the most concrete, honest proof
this doc can offer that the two mechanisms are actually doing what they claim — DDP's
`model.num_parameters()` (called on the *unwrapped* module reference after `DDP(model)`)
still reports the full 5,853,184, because DDP doesn't touch parameter storage, it only
adds gradient-synchronization hooks; FSDP's same call reports exactly half, because FSDP
replaces the module's parameter storage with this rank's local shard in place. The two
loss trajectories are nearly identical (both are training the same model on the same data
with the same effective batch size) — that's expected and correct: DDP and FSDP are
different *implementations* of data-parallel training, not different training algorithms,
so they should converge the same way. FSDP's wall-clock is very slightly slower (~1.5%)
here, consistent with the theory in
[Chapter 26](../../../docs/llm-engineering/26_distributed_training_ddp_and_fsdp.md): FSDP
does strictly more communication (per-layer all-gathers) than DDP (one gradient all-reduce
per step) — a real, if small at this scale, instance of the trade FSDP makes.

## What this project's numbers do NOT show, and why

- **No speedup from adding processes.** Both runs used 2 CPU processes on one machine
  communicating over `gloo`; there's no reason to expect (and this doc makes no claim of)
  the near-linear scaling DDP shows across real GPUs — that requires actual parallel
  compute hardware, which this environment doesn't have.
- **No memory benefit from FSDP.** FSDP's entire purpose is fitting a model too large for
  one device's memory; a 5.85M-parameter model already fits trivially anywhere, so this
  demo cannot and does not show FSDP's actual value proposition, only its mechanism.
- **No checkpoint saving for FSDP.** `src/gpt/training/trainer_fsdp.py` intentionally skips checkpointing —
  saving a *sharded* model correctly requires `FSDP.state_dict_type` configuration to
  gather a full state dict (or save shards separately and reassemble later), deliberately
  out of scope here to keep the script focused on the sharding mechanism itself.

## The gotcha: don't compare these loss curves to `TRAINING.md`'s as a distributed-vs-single-GPU speed claim

The loss trajectory above is *slower* in wall-clock terms per step than `src/gpt/training/trainer.py`'s own
single-process MPS runs in [`TRAINING.md`](TRAINING.md) — two CPU processes coordinating
over `gloo`, doing redundant (DDP) or extra-communication (FSDP) work, is genuinely slower
than one process running natively on MPS for a model this small. That is expected and not
a failure of either mechanism — it's exactly why the "hardware honesty" section above
exists: this doc proves the API and algorithm are real and correct, not that distributing
training helped anything at this scale, on this hardware.
