# Operations SOP — 2x L4 multi-GPU training

Maintained doc for this module — update in place across future sessions rather than
writing a parallel one, same convention as the single-GPU sibling module's own
`training_sop.md` (`infra/gcp-gpu-node/docs/training_sop.md`), which this doc
complements rather than duplicates: general GCP concepts, teardown mechanics, IAM/
budget gotchas already discovered there apply here unchanged and aren't re-explained.
This doc only covers what's genuinely different for 2 GPUs.

## Why this is a separate module, not a tfvars override

`machine_type = "g2-standard-24"` alone would be enough to *provision* 2 GPUs — the
single-GPU module's `variables.tf` has no validation restricting it to `g2-standard-4`.
What actually requires a separate module: **the single-GPU module's
`bootstrap.sh.tftpl` never launches training at all** — that's always been a manual
SSH+tmux+`uv run gpt-train` step (Phase 4 there). For 2 GPUs, the launch command itself
changes (`torchrun --nproc_per_node=2 -m gpt.cli.train`, not `uv run gpt-train`), and
that's real enough of a difference in the actual training workflow — not just the
Terraform — to warrant its own module and its own SOP, rather than a footnote on the
original.

## What's identical to the single-GPU module

- Phase 0 (one-time local setup: `gcloud`, auth) — see the original SOP, unchanged.
- Bucket/IAM/network/budget mechanics — same Terraform patterns, just doubled where
  cost-scaled (see `variables.tf`'s `monthly_budget_usd` default: 60 vs 30).
- Teardown (`make down` / `make destroy`), spot-preemption handling, checkpoint-sync
  watchdog — the bootstrap template is otherwise unchanged from the single-GPU module.
- The known IAM gap (deploying service account lacks `iam.roles.create` /
  `compute.instances.setIamPolicy`, so the self-stop binding is `count = 0`, disabled)
  — carried forward unchanged, see the original SOP's Known Issues for the fix.

## What's different

### 1. Machine shape, verified not assumed

`g2-standard-24` = 24 vCPU, 96GB RAM, **2x NVIDIA L4**, confirmed via:
```
gcloud compute machine-types describe g2-standard-24 --zone=us-central1-a \
  --format="value(guestCpus,memoryMb,accelerators)"
# -> 24  98304  {'guestAcceleratorCount': 2, 'guestAcceleratorType': 'nvidia-l4'}
```
Worth verifying explicitly rather than trusting either module's own pricing-table
comments: the single-GPU module's `outputs.tf` mislabels `g2-standard-12` as "2x L4" —
it's actually still 1x L4 (`guestAcceleratorCount: 1`), just with more vCPU/RAM
allocated to that one GPU. Don't copy that error forward.

### 2. GPU quota

`GPUS_ALL_REGIONS` is a global (non-regional) aggregate cap across every GPU the
project requests, independent of instance count. The single-GPU module's session only
ever needed this raised to 1. **Confirm it covers 2 before the first real `apply`** —
`gcloud compute regions describe <region> --format="value(quotas)"` or the Cloud
Console quotas page; request an increase if still at 1.

### 3. Training launch — the actual new piece

Same manual pattern as the single-GPU module (SSH in, `tmux`, launch by hand — no
Makefile target, no bootstrap automation, deliberately, matching that module's own
convention of keeping training launch a visible, supervised step rather than
something that silently starts before the corpus/checkpoint sync is confirmed done):

```bash
make iap-ssh                      # or: make ssh
cd ~/tiny_llm/from_scratch/custom-gpt-50m-ddp
tmux new -s train

nvidia-smi                        # confirm both L4s show up before launching
torchrun --nproc_per_node=2 -m gpt.cli.train
# add GPT_BATCH_SIZE=/GPT_GRAD_ACCUM= only if deliberately changing from the
# checkpoint's original settings — see the single-GPU SOP's same caution.
```

`torchrun` is the right launcher **here** (real hardware, normal networking) even
though the project's own local smoke test had to switch to `mp.spawn` — the DNS-lookup
hang that forced that switch was specific to the sandboxed local dev machine (see
`custom-gpt-50m-ddp/scripts/ddp_smoke_test.py`'s docstring), not something expected to
recur on a real GCP VM with normal DNS.

**Verify both GPUs are actually in use, not just present** — `nvidia-smi` in a second
pane during training should show non-zero utilization on both GPU 0 and GPU 1, not
just GPU 0. If only one shows activity, `torchrun`'s `--nproc_per_node` or the
per-rank device assignment in `cli/train.py`/`trainer.py` needs debugging before
trusting any throughput numbers from the run.

### 4. What to measure, and what to compare it against

This module's entire purpose is a real 1-GPU-vs-2-GPU comparison, not just "does it
run." The single-GPU sibling already has real, measured numbers for 50m on 1x L4
(`infra/gcp-gpu-node/docs/training_sop.md`'s "Speed vs. model size" section):

| | steps/sec | tok/s |
|---|---|---|
| 50m, 1x L4 (measured, single-GPU module) | 13.3 | ~56,900 |
| 50m, 2x L4 (this module — fill in once run) | ? | ? |

Capture the same shape of measurement here once a real run happens: live
training-loop steps/sec over a few minutes at steady state (not just the first-step
number, which includes one-time compilation/warmup), and note whether it's closer to
2x the single-GPU number (near-linear DDP scaling — plausible here since 50m was
already diagnosed as memory-bandwidth-bound on a single L4, and 2 separate GPUs each
have their own independent memory bus, unlike a bigger single GPU that would still
share one bus) or falls short of that (would point to communication overhead — NVLink
absence between L4s specifically, since L4 doesn't support NVLink, meaning
inter-GPU communication goes over PCIe, slower than an NVLink-connected pair like
A100s would get).

## Status

Terraform module created 2026-08-18, not yet applied. `custom-gpt-50m-ddp`'s DDP code
already verified correct via a local CPU/gloo smoke test (2 ranks, 20 steps, loss
decreased, checkpoint saved with clean non-`module.`-prefixed keys, and that checkpoint
verified to load cleanly in the original non-DDP `custom-gpt-50m`'s inference code) —
see that project's `scripts/ddp_smoke_test.py`. Real GCP run — the actual `terraform
init/plan/apply`, the torchrun launch, and the 1x-vs-2x comparison table above — is the
next real step, pending explicit go-ahead before any billing starts (same pattern as
every other real-spend step this session).
