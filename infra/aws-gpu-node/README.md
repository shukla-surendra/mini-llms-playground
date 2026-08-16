# `aws-gpu-node` — the training box, as code

Terraform for the EC2 GPU instance that [`docs/AWS_RUNBOOK.md`](../../docs/AWS_RUNBOOK.md)
describes clicking together by hand: `g6.xlarge`, Deep Learning Base OSS Nvidia AMI,
100 GB gp3, SSH from your IP only, an IAM instance role for S3, and the cost guards the
runbook warns you to set up "before you start, not after".

Same box, same choices — the difference is that the choices are now written down, applied
identically every time, and destroyable in one command. The runbook still explains *why*
each value is what it is; this module is the executable form of it.

## What it creates

| Resource | Why it is shaped this way |
|---|---|
| `aws_instance` `g6.xlarge` | L4 24 GB with **bf16 + TF32**. The cheaper `g4dn` is Turing: `torch.cuda.is_bf16_supported()` is `False` and `precision="auto"` silently drops to fp32 |
| Root **100 GB gp3**, encrypted | corpus `.bin` (~5 GB) + ~1.8 GB per checkpoint + OS + venv. The 8 GB default dies mid-run |
| `instance_initiated_shutdown_behavior = "stop"` | Makes the runbook's dead-man switch safe: `shutdown -h now` ends compute billing and **keeps** the volume, so a restart resumes from `latest.pt` |
| IAM role + instance profile | S3 read/write scoped to one bucket, plus `ec2:StopInstances` on this project's tag. **No access keys ever land on the box** |
| S3 bucket | Corpus in, checkpoints out. Encrypted, public access blocked, incomplete multipart uploads auto-aborted after 7 days |
| Security group | Port 22 from your `/32` only. The model API port stays **closed** — use `make tunnel` |
| GPU idle watchdog (systemd) | Stops the instance after N minutes of ~0% GPU. This is the guard for the failure mode the runbook keeps flagging: a finished run idling overnight at **$19/day** |
| Checkpoint sync + spot watcher (systemd) | `checkpoints/` → S3 every N minutes, and again inside the ~2-minute Spot interruption notice. This is what turns spot's 60-70% discount from a gamble into a default |
| `aws_budgets_budget` (optional) | Forecast-based email alert, days before the damage |
| cloud-init bootstrap | Installs `uv`, clones the repo, `uv sync`s the chosen project, pulls the corpus from S3 |

## Quickstart

```bash
cd infra/aws-gpu-node
cp terraform.tfvars.example terraform.tfvars   # ships in "cheap mode": spot + idle stop
make init
make spot-price                                # what the GPU hour costs right now
make plan                                      # read this before spending anything
make apply                                     # billing starts here

make upload-corpus                             # token files -> S3 (do this from the Mac)
make ssh
```

On the box, before anything else — the check that catches a wrong instance type:

```bash
make gpu        # expects: NVIDIA L4 True
```

Then follow the runbook from step 10 (`gpt-benchmark`) onward. When you are done:

```bash
make download-checkpoints
make down       # destroys the box; bucket stays. See "Paying for the GPU" below for
                # why this beats `make stop` once the run is actually finished.
```

## The order that matters: tokenize local, train remote

`make upload-corpus` ships `*.bin` and `*.bin.json` only — never `train.txt`, never
`hf_cache/`. Tokenizing costs CPU seconds on the Mac and GPU-hours on an instance billed
by the hour, and the `.bin.json` sidecars are what make a tokenizer mismatch fail loudly
instead of training on ids that index the wrong embedding rows.

Set `corpus_prefix` and the instance pulls the corpus itself at boot, using the instance
role — no credentials, no `scp`, and at in-region S3 speed (seconds, not your home
upload).

## Decisions worth knowing about

**AMI changes are ignored.** `lifecycle { ignore_changes = [ami] }` on the instance means
a newer Deep Learning AMI released next week cannot quietly destroy a box that is 14 hours
into a run when you apply an unrelated change. Rebuilding on the current AMI is an
explicit act: `make replace`.

**Port 22 defaults to your detected public IP.** Leave `allowed_ssh_cidrs` empty and the
module resolves your `/32` the same way the console's "My IP" does. `0.0.0.0/0` is
refused by a precondition unless you set `allow_open_ssh = true` — this box holds a role
that can write to your bucket, so an open SSH port is not just a box-level risk. Home IPs
rotate; re-`apply` when yours does, or use `make ssm` via Session Manager, which needs no
inbound rule at all.

**The AZ is chosen, not assumed.** Not every AZ has G6 capacity. The module asks EC2
which AZs offer the instance type and picks a subnet there, turning a launch-time
`Unsupported` error into a plan-time one.

**No Elastic IP.** The public IP changes on every stop/start, which is mildly annoying;
an idle EIP is billed, which is worse. `terraform refresh && make status` after a start.

**The watchdog watches the GPU, not the CPU.** During training the CPU sits mid-range
(dataloader, logging, eval), so a CPU-based idle alarm would kill live runs. An idle GPU
is the honest signal that the run finished, crashed, or never started. If `nvidia-smi`
cannot be read, it counts as *busy* — unknown must never mean "kill the run". Pause it
for a session with `touch ~/.no-idle-shutdown`.

**Spot is opt-in and refuses a half-safe config.** Spot instances cannot stop on in-guest
shutdown — AWS terminates them, deleting the root volume with your checkpoints on it. So
`use_spot = true` requires `checkpoint_sync_minutes > 0` and a bucket, and the plan fails
otherwise. With those in place a reclaim costs you one sync interval, not the run: a
systemd timer pushes `checkpoints/` to S3 every N minutes, and a second watcher polls the
IMDS interruption endpoint every 5 s and flushes again inside the ~2-minute warning.

**State is local.** Fine for one operator on one Mac. The moment a second machine applies
this, uncomment the S3 backend in `versions.tf` — two divergent local states means two
`g6.xlarge` instances nobody is watching.

## Paying for the GPU and almost nothing else

On a 21-hour run of `custom-gpt-153m`, the GPU is ~98% of the bill. Everything else is
rounding error **while the run is happening** — the leaks are all in the time around it.

| | Rate | 21-hour run |
|---|---|---|
| `g6.xlarge` on-demand | $0.8048/hr | **$16.90** |
| `g6.xlarge` spot | ~60-70% off (`make spot-price`) | **~$5-7** |
| 100 GB gp3 root | $0.08/GB-month | $0.23 |
| Public IPv4 | $0.005/hr | $0.11 |
| S3, ~600 MB corpus | $0.023/GB-month | ~$0.01 |
| IAM, security group, key pair, budgets | free | $0 |

Three rules follow from that table, in order of how much they save:

**1. Buy the GPU hour on spot.** `use_spot = true` is the whole game — it is the only
line item large enough to matter. The module makes it safe rather than merely cheap
(periodic S3 sync + interruption flush), so a reclaim costs one sync interval.

**2. Between runs, `make down`, not `make stop`.** This is the leak people actually pay:
a *stopped* instance bills **$8/month** for its 100 GB EBS volume, indefinitely, while
doing nothing. `make down` destroys the instance and keeps the bucket, IAM role, security
group and key pair — all of which are free — so the bill goes to roughly zero and the
next `make up` is a 3-minute bootstrap, not a re-setup. Use `make stop` only for an
overnight pause in the middle of a run you are resuming tomorrow.

**3. Do not buy hours you can avoid.** The corpus is tokenized on the Mac because
tokenizing on an hourly GPU is money for nothing. `tinystories-gpt-6m` and
`custom-gpt-10m` train on Apple Silicon in minutes — the cheapest GPU hour is the one you
never launch. And `make benchmark` on the instance is still the best-value hour in the
table: it is what stops you discovering 20 hours in that MFU was half what you assumed.

**Two traps worth naming.** A cheaper *per hour* GPU is not a cheaper run: `g4dn.xlarge`
is $0.526/hr but Turing has no bf16, so `precision="auto"` falls back to fp32 and the run
takes long enough to cost *more* than `g6.xlarge` end to end. And the $0.005/hr public
IPv4 charge tempts people into a private subnet — which needs a NAT gateway at $0.045/hr
plus data processing, nine times the charge it was meant to avoid. Keep the public IP.

`terraform destroy` ends everything; the bucket survives unless
`force_destroy_bucket = true`. Sync your checkpoints before either that or `make down`.

## Variables you will actually touch

| Variable | Default | Notes |
|---|---|---|
| `project_subdir` | `from_scratch/custom-gpt-153m` | Which experiment gets `uv sync`'d and treated as the working dir |
| `corpus_prefix` | `""` | S3 prefix pulled into `<project_subdir>/data` at boot; empty skips it |
| `instance_type` | `g6.xlarge` | See the runbook for why not `g4dn`/`g4ad` |
| `idle_shutdown_minutes` | `30` | `0` disables the watchdog |
| `checkpoint_sync_minutes` | `15` | Periodic `checkpoints/` → S3; required by `use_spot` |
| `allowed_ssh_cidrs` | `[]` | Empty = auto-detect your `/32` |
| `use_spot` | `false` | The main cost lever; needs `checkpoint_sync_minutes > 0` + a bucket |
| `monthly_budget_usd` / `budget_alert_email` | `0` / `""` | Both needed to create the budget |

Full list with reasoning in [`variables.tf`](variables.tf).
