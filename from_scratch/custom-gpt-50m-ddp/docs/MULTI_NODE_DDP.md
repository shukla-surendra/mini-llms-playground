# Running this model's DDP across 2, 3, or more separate machines

Confirmed nothing in this project's code assumes exactly 2 nodes — `RANK`/
`WORLD_SIZE`/`LOCAL_RANK` (`cli/train.py`) and `scripts/ddp_smoke_test.py`'s own
`SMOKE_WORLD_SIZE` env var are both genuinely N-capable, not hardcoded. This doc
is the generalized version of `DATA_AND_TRAINING_SOP.md`'s brief 2-node example.
For *why* any of this works (rendezvous mechanism, what a gradient sync actually
does), see the sibling `custom-gpt-350m-ddp` project's
[`docs/DISTRIBUTED_TRAINING.md`](../../custom-gpt-350m-ddp/docs/DISTRIBUTED_TRAINING.md)
— that explanation is architecture-agnostic; nothing there is specific to the
350M model, and re-deriving it here would just be a copy.

## The launch pattern, for N machines

One machine is arbitrarily "node 0" — not special hardware, just the one every
other node's launch command points at. Every node runs the **same** command
except its own `--node_rank`:

```bash
# Node 0 (the one every other node's --master_addr points at):
torchrun --nnodes=<N> --node_rank=0 --nproc_per_node=1 \
  --master_addr=<node 0's private IP> --master_port=29500 -m gpt.cli.train

# Node 1:
torchrun --nnodes=<N> --node_rank=1 --nproc_per_node=1 \
  --master_addr=<same node 0 private IP> --master_port=29500 -m gpt.cli.train

# Node 2:
torchrun --nnodes=<N> --node_rank=2 --nproc_per_node=1 \
  --master_addr=<same node 0 private IP> --master_port=29500 -m gpt.cli.train

# ... one such command per node, node_rank 0 through N-1, all sharing the
# identical --nnodes and --master_addr/--master_port values.
```

`--nproc_per_node` is separate from `--nnodes`: it's how many processes (≈ GPUs)
*that one machine* runs. A single-GPU-per-box fleet of N machines uses
`--nproc_per_node=1` on every node (`--nnodes=N` total). Two GPUs per box across
N boxes would use `--nproc_per_node=2 --nnodes=N` (total world size = 2N), same
principle as `torchrun --nproc_per_node=2` alone would do on one multi-GPU
machine.

## What actually changes as N grows (and what doesn't)

- **Rendezvous is unchanged.** Still exactly one `master_addr:master_port` —
  every node's process connects to that same address (see the linked
  `DISTRIBUTED_TRAINING.md` for the precise mechanism); `torchrun` computes each
  process's global `RANK` and total `WORLD_SIZE` from `--nnodes`/`--node_rank`/
  `--nproc_per_node` before this project's own code ever runs.
- **The all-reduce gets more expensive, not free.** DDP's gradient sync is a
  collective across *all* `WORLD_SIZE` ranks, once per `grad_accum_steps` window
  — more ranks means more machines that must all finish their local
  forward/backward before that sync can complete, and (depending on network
  topology) more total data moved. This is why `no_sync()` gating matters more,
  not less, as N grows — it's the only thing keeping non-boundary micro-steps
  network-free regardless of how many nodes are in the job.
- **Network requirements scale to a full mesh, not a star.** Every node needs to
  reach node 0's `master_port` for rendezvous, **and** every node needs to reach
  every other node for NCCL's actual gradient traffic once training starts (NCCL
  picks its own communication topology — often a ring — among all ranks, not
  just node-0-to-everyone-else). The simplest correct security-group/firewall
  rule for N nodes is "allow all traffic between every node in this job," not
  just "allow traffic to node 0" — a rule scoped to only node 0 will pass
  rendezvous and then hang or error the first time two non-zero-rank nodes need
  to exchange gradients directly.

## Token budget math, generalized

Same formula as the 2-node case in `DATA_AND_TRAINING_SOP.md`, just with `N`
instead of `2`: **global tokens per step = `batch_size × context_length × N`**.
`GPT_STEPS` is not world-size-aware in this project (unlike
`custom-gpt-350m-ddp`'s `GPT_TARGET_TOKENS` mechanism) — the same `GPT_STEPS`
value under a bigger `N` consumes proportionally more total tokens, since every
additional node processes its own full batch independently and in parallel.
Divide your intended total-token budget by `batch_size × context_length × N` to
get the right `GPT_STEPS` for whatever `N` you're actually running — read
`train()`'s own startup "Budget" print (it reports the real, world-size-scaled
total) to confirm before committing GPU-hours across many machines at once.

## Smoke-test N ranks locally before touching real hardware

`scripts/ddp_smoke_test.py` isn't limited to 2 — `SMOKE_WORLD_SIZE` controls it
directly:

```bash
SMOKE_WORLD_SIZE=4 uv run python scripts/ddp_smoke_test.py   # 4 CPU/gloo ranks, one machine, near-$0
```

This verifies the *mechanism* (rendezvous among N processes, gradient sync,
checkpoint compatibility) scales to more ranks with zero cloud spend, the same
principle as the 2-rank version already used to validate this project's DDP path
— run this before ever provisioning N real machines, not as an afterthought.

## Actually provisioning N real machines

This project has no ready-made Terraform module of its own yet (unlike
`custom-gpt-350m-ddp`, whose `infra/aws-gpu-node-multi/` is currently hardcoded
to exactly 0 or 2 instances — see that module's `variables.tf`'s
`instance_count` validation). Two real options if you want this on real cloud
hardware:

1. **Generalize `infra/aws-gpu-node-multi`** — change `instance_count`'s
   validation from `contains([0, 2], var.instance_count)` to allow any N,
   replace the two named `aws_instance` resources (`gpu_master`/`gpu_worker`)
   with a `count`-based (or `for_each`-based, for named ranks) resource, and
   generalize the bootstrap template's `node_rank`/`master_addr` templating to
   loop over N nodes instead of assuming exactly two. Real, scoped Terraform
   work — not a config-only change.
2. **Provision N boxes by hand** (or via a simpler ad-hoc script) using that
   same module's networking/security-group pattern as a reference (one subnet,
   one AZ, security group allowing all traffic between the job's own instances)
   — faster for a one-off N-node experiment, at the cost of not being a reusable
   module afterward.

Either way, apply the same lesson from `custom-gpt-350m-ddp`'s real deployment
(`docs/RUN_LOG_2026-08-31.md` in that project): verify batch size / memory
headroom on **one** GPU first, and check EC2 capacity/quota for whatever
instance type and count you actually intend to launch — both real, hours-costing
surprises on that run, and nothing about N nodes makes either risk smaller.

## Practical guidance on how large to make N

More nodes only helps while the extra parallel compute outweighs the growing
sync cost — for a model this size (51.48M params, a genuinely small amount of
gradient data per sync compared to `custom-gpt-350m-ddp`'s 347M), the sync
itself is cheap in absolute terms, but non-EFA cloud networking's fixed
per-collective latency doesn't shrink just because the payload is small,
meaning **that fixed overhead becomes a larger fraction of each step's total time
as N grows**, especially with `grad_accum_steps` set low. If you raise N, raise
`grad_accum_steps` too, for the same reason the 350M run needed it: fewer,
bigger syncs are always better amortized than many small ones. Start at N=2
(cheapest real multi-node validation), confirm real measured throughput
(`gpt-benchmark` or a short timed run) before deciding whether N=3+ is actually
worth the added coordination and cost for this specific model size.
