# Training SOP — 2-node DDP, manual runbook

Self-contained. No prior context needed beyond this file. Written after the
2026-08-31 first real run (full story: [`RUN_LOG_2026-08-31.md`](RUN_LOG_2026-08-31.md))
so every command below already has that run's fixes baked in — you should NOT
need to re-discover the `torchrun`/`uv`/OOM issues described there; if you hit
them anyway, something has drifted from this SOP and that log is where to look.

All commands run from your local machine unless a step says "on `master`" / "on
`worker`" (SSH'd in). `cd` to `infra/aws-gpu-node-multi/` for every `terraform`/
`make` command below.

`<ACCOUNT_ID>` below is a placeholder, not something to fill in by hand — get
the real bucket name with `terraform output bucket` (or `aws sts
get-caller-identity --query Account --output text` to fill it in yourself); it
isn't hardcoded here on purpose, since this file may end up in a shared/public
place.

## 0. One-time prerequisites (already done as of 2026-08-31 — verify, don't redo)

- [ ] AWS CLI has working credentials: `aws sts get-caller-identity` prints your
      account, region `us-east-1`.
- [ ] G/VT instance quota ≥ 8 vCPUs in `us-east-1`:
      `aws service-quotas get-service-quota --region us-east-1 --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value' --output text`
      — should print `8` or higher. If not, request the increase in the Service
      Quotas console first; approval can take up to an hour and blocks
      everything below.
- [ ] Corpus + tokenizer already in S3 (uploaded once, reused every run):
      `aws s3 ls s3://mini-llm-gpu-ddp-<ACCOUNT_ID>-us-east-1/350m-ddp/ --recursive --human-readable`
      should show `corpus/train.bin` (~1.9 GiB), `corpus/test.bin` (~19.6 MiB),
      `tokenizer/tokenizer.json` (~2.2 MiB). If missing, from
      `from_scratch/custom-gpt-350m-ddp`: `make -C ../../infra/aws-gpu-node-multi upload-corpus`
      and `upload-tokenizer`.
- [ ] `terraform.tfvars` in `infra/aws-gpu-node-multi/` exists (copy from
      `.example` if not) and already has `batch_size = 4`,
      `grad_accum_steps = 256`, `instance_type = "g5.xlarge"`,
      `subnet_id = "subnet-0ddd28a9cc6a2f624"` (us-east-1c) — these are the
      validated values, not the module's un-overridden defaults.

## 1. Deploy the instances

```bash
cd infra/aws-gpu-node-multi
terraform apply -var instance_count=2 -auto-approve
```

**What to expect**: 2-5 minutes normally. If a node's line says
`Still creating...` past ~3 minutes with the AWS Console showing nothing under
EC2 → Instances (check the **us-east-1** region specifically, top-right region
selector — the CLI's configured default profile region may differ from what
you're viewing in the console), that is very likely
`Server.InsufficientInstanceCapacity`, not a real hang. Confirm directly instead
of waiting on Terraform:

```bash
aws cloudtrail lookup-events --region us-east-1 \
  --lookup-attributes AttributeKey=EventName,AttributeValue=RunInstances \
  --max-results 1 --query 'Events[0].CloudTrailEvent' --output text \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('errorCode'), d.get('errorMessage'))"
```

If it says `InsufficientInstanceCapacity` for `g5.xlarge`: kill the apply
(`Ctrl-C`, safe — no instances exist yet if this is the first attempt), and
either just retry (capacity fluctuates minute to minute — one node landed in
2m15s and the other took 11+ min in the *same* successful apply on 2026-08-31),
or try a different `subnet_id` (any of `us-east-1a/b/c/d` in this account — get
IDs with `aws ec2 describe-subnets --region us-east-1 --filters "Name=vpc-id,Values=<your default vpc id>" --query 'Subnets[].{ID:SubnetId,AZ:AvailabilityZone}' --output table`).

## 2. Get connection info

```bash
terraform output
```

Note `master_public_ip`, `worker_public_ip`, `ssh_master`, `ssh_worker`. Public
IPs are freshly assigned each apply (no Elastic IP) — always re-run this rather
than reusing yesterday's IPs.

## 3. Verify both nodes finished bootstrapping

```bash
make bootstrap-log-master
make bootstrap-log-worker
```

Look for a line like `=== bootstrap done <timestamp> — node_rank=0, launch with: bash ~/launch_ddp.sh ===`
at the end, and no `WARN: ... sync failed` lines above it. Typically finishes
1.5-2 minutes after the instance is `running`.

## 4. Verify GPU and the generated launch script

```bash
make gpu-master   # or: ssh <master> "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"
make gpu-worker
```

Expect `NVIDIA A10G, 23028 MiB`. Then on **each** node:

```bash
cat ~/launch_ddp.sh
```

Confirm it shows (this SOP's whole point — these should already be correct,
with no manual editing needed):
```
export GPT_TARGET_TOKENS=4000000000
export GPT_GRAD_ACCUM=256
export GPT_BATCH_SIZE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$HOME/.local/bin/uv" run torchrun --nnodes=2 --node_rank=<0 or 1> --nproc_per_node=1 \
  --master_addr=172.31.80.10 --master_port=29500 \
  -m gpt.cli.train
```
(`master_addr` will match whatever `master_private_ip` this apply generated —
it's stable across a stop/start of the same instance, but a fresh `apply`
after a full `destroy` can assign a different one; the generated script is
always correct for whatever was actually deployed, this is just a sanity
check, not something to hand-edit.)

If `torchrun`/`uv` command-not-found happens anyway, or an OOM happens at these
settings: something has drifted from the 2026-08-31 fix — see
`RUN_LOG_2026-08-31.md`'s "Every error hit" section before improvising a new
fix; the single-GPU test method there (§ "Finding the actual working
batch_size") is the fast, cheap way to re-diagnose a memory problem without
burning 2-node time.

## 5. Launch training — both nodes, roughly together

On **master**:
```bash
tmux new -s train
bash ~/launch_ddp.sh 2>&1 | tee ~/train.log
```
Detach: `Ctrl-b` then `d`. Then on **worker**, same:
```bash
tmux new -s train
bash ~/launch_ddp.sh 2>&1 | tee ~/train.log
```
Order matters less than timing — start the second one within a minute or two
of the first; DDP's rendezvous will wait for both, but a very long gap risks a
timeout on the first one.

## 6. Monitor

```bash
ssh <master> "tail -f ~/train.log"       # only master prints progress by design (rank 0 only)
ssh <master> "nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader"
ssh <worker> "nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader"
```

**Worker's own `train.log` staying empty is normal and expected** — only rank 0
(master) prints the progress bar, eval, and writes checkpoints; worker only
writes to its log if it errors. Don't mistake worker silence for a hang; check
its `nvidia-smi` utilization (should read ~100%) instead.

**Expected steady-state numbers** (measured 2026-08-31, same instance type/settings):
- ~1.7-1.74 s/step
- ~9,500 tokens/sec combined across both GPUs
- ~21.6 GiB GPU memory during plain steps, spiking to ~22.5 GiB during eval
  (every 500 steps) — a recurring `expandable_segments: memory mapping failed`
  warning during those spikes is non-fatal (see `RUN_LOG_2026-08-31.md`); it's
  a problem only if steps actually stop incrementing afterward.
- At `GPT_TARGET_TOKENS=4000000000`, this throughput projects to **~125 hours
  (~5 days)** wall-clock — decide up front whether that's the intended budget,
  or lower `GPT_TARGET_TOKENS` in `terraform.tfvars` before deploying (it's
  baked into `launch_ddp.sh` at boot, not something you'd edit live on the box).

## 7. Checkpoints

- Only master (`is_main`) writes checkpoints, to `checkpoints/350m/` — worker
  has nothing to sync, this is correct, not a bug.
- Synced to S3 automatically every 15 minutes
  (`s3://mini-llm-gpu-ddp-<ACCOUNT_ID>-us-east-1/350m-ddp/checkpoints/`).
- **Auto-resume on a fresh instance** requires `checkpoints/350m/latest.pt`
  specifically (written only once `save_every_steps=2000` is reached) — `best.pt`
  (written at every eval-loss improvement, including step 0) does **not**
  trigger auto-resume by itself. Check what's actually in S3 before assuming a
  fresh deploy will resume:
  ```bash
  aws s3 ls s3://mini-llm-gpu-ddp-<ACCOUNT_ID>-us-east-1/350m-ddp/checkpoints/350m/
  ```
  If you want to force-clear a stale/irrelevant checkpoint before a fresh run
  (e.g. only `best.pt`/`serving.pt` from a step-0 baseline, no real progress):
  ```bash
  aws s3 rm s3://mini-llm-gpu-ddp-<ACCOUNT_ID>-us-east-1/350m-ddp/checkpoints/350m/ --recursive
  ```

## 8. Pause or stop

**Pause (keep everything, stop billing on compute only)** — mid-run, from
either node or here:
```bash
# on each node: stop training first if you want a clean stop rather than a hard kill
ssh <node> "tmux kill-session -t train; pkill -9 -f 'gpt.cli.train'; pkill -9 -f torchrun; true"
# then, from infra/aws-gpu-node-multi:
terraform apply -var instance_count=0 -auto-approve   # == `make down`
```
This destroys just the 2 EC2 instances. Bucket, IAM, security group, key pair,
and everything in S3 (corpus/tokenizer/checkpoints) all survive — billing drops
to a few cents/month (S3 storage only). To resume later: repeat from Step 1
(`terraform apply -var instance_count=2`) — new instances, new IPs, same S3
data, corpus/tokenizer sync automatically at boot.

**Full teardown (rare — only if abandoning this project's AWS footprint
entirely)**:
```bash
terraform destroy   # == `make destroy` — also removes the bucket, IAM, SG, key pair
```
Do not use this just to pause between runs — it forces a full corpus
re-upload (~2 GiB) next time.

## Quick reference: known-fixed issues (2026-08-31)

| Symptom | Already fixed by | Where |
|---|---|---|
| `RunInstances` silent for 10+ min, nothing in console | N/A — genuine AWS capacity issue, not a bug. Diagnose via CloudTrail (Step 1), retry or switch AZ/subnet. | — |
| `torchrun: command not found` | `launch_ddp.sh` uses `"$HOME/.local/bin/uv" run torchrun`, not bare `torchrun` | `templates/bootstrap.sh.tftpl` |
| `uv: command not found` | Same fix — absolute path to `uv`, not relying on `PATH` | `templates/bootstrap.sh.tftpl` |
| CUDA OOM at `batch_size=16` or `8` | `batch_size=4`, `grad_accum_steps=256`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — all now the module's own defaults | `variables.tf`, `terraform.tfvars` |

If any of these recur, `terraform.tfvars`/`bootstrap.sh.tftpl` has drifted from
what's described here — fix the source, not just the symptom on the live box.
