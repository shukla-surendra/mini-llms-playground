# `aws-gpu-node` — the training box, as code

Terraform for the EC2 GPU instance that [`docs/AWS_RUNBOOK.md`](../../docs/AWS_RUNBOOK.md)
describes clicking together by hand: `g6.xlarge`, Deep Learning Base OSS Nvidia AMI,
100 GB gp3, SSH from your IP only, an IAM instance role for S3, and the cost guards the
runbook warns you to set up "before you start, not after".

Same box, same choices — the difference is that the choices are now written down, applied
identically every time, and destroyable in one command. The runbook still explains *why*
each value is what it is; this module is the executable form of it.

**Operating it day to day:** this README covers design decisions and cost. For the
full command-by-command sequence — `apply` → upload data → launch → train → monitor
→ manage → teardown — see [`docs/SOP.md`](docs/SOP.md). For the mechanics of
resuming an existing checkpoint specifically (auto-resume, effective-batch matching,
`tmux` disconnect-safety), see [`docs/RESUME_TRAINING.md`](docs/RESUME_TRAINING.md).

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

## Prerequisites (one-time, on the Mac)

Two CLIs, neither of which this repo installs for you.

**Terraform**, via HashiCorp's Homebrew tap:

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
terraform version              # confirm it's on PATH, >= 1.6 (see versions.tf)
```

That pulls the official binary (BSL-licensed, not open-source since v1.6). To stay
fully open-source instead, swap in the drop-in fork and point the Makefile at it:

```bash
brew install opentofu
# then in this dir's Makefile: change `TF ?= terraform` to `TF ?= tofu`
```

**AWS CLI**, needed for every `make` target that isn't a bare `terraform` command
(`upload-corpus`, `download-checkpoints`, `ssh`'s IP lookup, `spot-price`, `stop`/`start`,
`ssm`):

```bash
brew install awscli
aws configure
```

`aws configure` asks for four things — enter them and it writes `~/.aws/credentials`:

| Prompt | What to give it |
|---|---|
| AWS Access Key ID / Secret Access Key | From an **IAM user** with programmatic access — not your root account. Console → IAM → Users → your user → Security credentials → Create access key |
| Default region | Match `region` in `terraform.tfvars` (`us-east-1` by default) — S3↔EC2 transfer is free in-region, billed and slower across regions |
| Default output format | `json` (what the `--query` flags in this Makefile expect) |

Verify it worked:

```bash
aws sts get-caller-identity   # should print your account id and IAM user ARN, not an error
```

This is a one-time setup per machine. Nothing above is specific to this module — it's
the same pair of tools any Terraform-on-AWS project needs.

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

**No `.pem` — the key pair is imported, not AWS-generated.** `aws_key_pair.this` in
`main.tf` runs `public_key = file(pathexpand(var.public_key_path))`: it reads a
**public** key you already have and uploads only that half to AWS. That's different
from the EC2 console's default flow, where AWS generates the pair itself and shows
you the private half exactly once as a `.pem` to download. Here, AWS never holds a
private key at all — there's nothing to download because the private half never
existed on AWS's side. If `public_key_path` doesn't exist yet:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "mini-llm-gpu"
```

The private key stays on this Mac, permanently. To connect:

```bash
make ssh    # ssh -i ~/.ssh/id_ed25519 ubuntu@<public-ip> — same file, wired
            # into $(KEY) in the Makefile and ssh_private_key_path in tfvars
```

There's also a second way in that needs **no key file at all**: `make ssm`, via
Session Manager (the `AmazonSSMManagedInstanceCore` role attachment, `enable_ssm`).
Useful if the key is ever lost, or your home IP rotates and port 22 is blocked
before you get around to re-`apply`-ing.

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

Of the 14 resources this module plans, exactly **three** carry a real, ongoing charge.
Everything else — IAM role/policies/instance-profile, the security group and its
rules, the key pair, SSM sessions — is free, full stop, forever.

| Cost | Billed while | Rate | Note |
|---|---|---|---|
| EC2 compute (`aws_instance`) | instance **running** | $0.8048/hr on-demand, ~60-70% less on spot | this is "the GPU"; everything below is not |
| EBS root volume | instance **exists** — stopped *or* running | $0.08/GB-month → 100 GB ≈ $8/month | bundled inside `aws_instance`, not its own line in the resource list — easy to miss |
| S3 storage | always, while objects exist | $0.023/GB-month | corpus + checkpoint together are usually a few GB → a few cents/month |

Plus one small one that's easy to forget because it's account-wide AWS policy, not
this module's choice: **public IPv4 is $0.005/hr** (~$3.60/month if the instance
exists 24/7). It stops the moment the instance is destroyed, same as EBS.

On a 21-hour on-demand run of `custom-gpt-153m`: **~$17 compute + $0.23 EBS + $0.11
IP + ~$0.01 S3** — compute is ~98% of it. The leaks are all in the time *around* the
run, not during it — which is what the rules below are actually about.

**The budget alarm needs both variables, and silently isn't armed without them.**
`monthly_budget_usd` alone does nothing — `aws_budgets_budget`'s `count` is
`monthly_budget_usd > 0 && budget_alert_email != "" ? 1 : 0`, so setting only the
number (as the example tfvars does) leaves it at `count = 0`: no resource, no alert,
no error telling you so. Check `terraform plan` includes `aws_budgets_budget.monthly`
before assuming it's watching.

Three rules follow from the cost table, in order of how much they save:

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
