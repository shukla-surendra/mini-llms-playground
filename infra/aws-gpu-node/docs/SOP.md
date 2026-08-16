# Operations SOP — from `terraform apply` to teardown

Everything you run, in order, from a clean checkout through training, monitoring,
and turning it back off. This is the "what to type" reference; the "why" for each
choice lives in [`../README.md`](../README.md) (module design, cost, security
group/key-pair decisions) and [`RESUME_TRAINING.md`](RESUME_TRAINING.md) (resume
mechanics and disconnect-safety in full — this SOP includes those commands inline
but doesn't re-derive the reasoning behind them).

Worked example throughout: `custom-gpt-50m`, resuming an existing run, on-demand,
`us-east-1` — i.e. this directory's current `terraform.tfvars`. Substitute your own
project/region/pricing-model where it differs.

All commands assume `cd infra/aws-gpu-node` unless stated otherwise.

---

## Phase 0 — One-time local setup (skip if already done)

```bash
# Terraform + AWS CLI
brew tap hashicorp/tap && brew install hashicorp/tap/terraform
brew install awscli
aws configure                              # access key, region, output=json
aws sts get-caller-identity                # confirms it worked

# SSH key — imported, not AWS-generated; no .pem involved anywhere in this flow
ls ~/.ssh/id_ed25519.pub 2>/dev/null || \
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "mini-llm-gpu"

# terraform.tfvars — copy once, then edit project_subdir/corpus_prefix/checkpoint_prefix
cp terraform.tfvars.example terraform.tfvars   # skip if terraform.tfvars already exists
make init
```

## Phase 1 — Provision, no billing yet

The instance and the bucket are created in the same `apply` by default. Split it:
create the bucket/IAM/security-group/key-pair *first*, so there's somewhere to
upload data to before the box ever boots.

```bash
make plan                  # read it — this is the point to catch a wrong region/type
make down                  # apply -var instance_count=0: everything except the instance
```

**`make down` here is not "turning something off" — nothing exists yet, so there is
nothing to turn off and nothing has been billing.** It runs
`terraform apply -var instance_count=0`, and Terraform reconciles that declaratively
against *whatever currently exists*: right now that's nothing, so it creates the 13
free resources (bucket, IAM role, security group, key pair, …) and simply never
creates the 14th — the EC2 instance itself, since its `count` is driven by
`instance_count`. No instance means no compute billing, same as before you ran
anything.

The reason it's the same `make down` you'll use later (Phase 6/7) to actually stop
paying for a *running* instance is that the command means the same thing both
times — "reconcile to `instance_count = 0`" — it's only the starting point that
differs: nothing → still nothing, versus a running box → destroyed. `terraform plan`
right before it is what tells you which case you're in; run it if you're ever unsure
whether `make down` is about to create or destroy something.

## Phase 2 — Upload data

```bash
# Corpus (always). Ships *.bin + *.bin.json only — never train.txt.
# If data/*.bin.json is missing, rebuild it first (free, on the Mac):
#   cd ../../from_scratch/custom-gpt-50m && uv run gpt-tokenize --force
make upload-corpus PROJECT_DIR=../../from_scratch/custom-gpt-50m

# Checkpoint (only if resuming an existing run — skip for a fresh start)
make upload-checkpoint PROJECT_DIR=../../from_scratch/custom-gpt-50m
```

Confirm before moving on — both should list real objects, not an empty prefix:

```bash
aws s3 ls "s3://$(terraform output -raw bucket)/50m/corpus/"
aws s3 ls "s3://$(terraform output -raw bucket)/50m/checkpoints/" --recursive
```

## Phase 3 — Launch

```bash
make up                    # billing starts here
make bootstrap-log         # tail cloud-init: uv install, repo clone, uv sync,
                            # corpus pull, checkpoint pull — watch it finish clean
```

`bootstrap-log` should end with `=== bootstrap done ===`. If the checkpoint sync
step logs nothing, `checkpoint_prefix` in `terraform.tfvars` doesn't match what
Phase 2 uploaded to — fix before starting training, not after.

```bash
make gpu                   # expect: NVIDIA L4 True — wrong instance type otherwise
```

## Phase 4 — Start (or resume) training

**Never run `gpt-train` directly in the SSH session you're typing in** — closing the
terminal sends `SIGHUP` and kills it mid-step. Use `tmux` (full reasoning and the
disconnect/reconnect walkthrough: [`RESUME_TRAINING.md`](RESUME_TRAINING.md)):

```bash
make ssh
cd ~/tiny_llm/from_scratch/custom-gpt-50m
tmux new -s train
```

Inside the `tmux` session — check what effective batch the checkpoint was trained
under before picking new values (skip this block for a fresh run):

```bash
python3 - <<'PY'
import torch
ckpt = torch.load("checkpoints/50m/latest.pt", map_location="cpu")
print("step:", ckpt.get("step"))
print("batch_size:", ckpt.get("batch_size"), "grad_accum_steps:", ckpt.get("grad_accum_steps"))
PY
```

Then launch, matching the effective batch (`batch_size x grad_accum`) unless you've
deliberately decided to change it:

```bash
GPT_BATCH_SIZE=16 GPT_GRAD_ACCUM=2 uv run gpt-train
```

Confirm the banner: right model/label, `device=cuda`, and — if resuming —
`Resumed from checkpoints/50m/latest.pt at step <N>` with `N` matching what the
metadata check above printed, not `0`.

```
Ctrl-b  d          # detach — training keeps running, you can now close this terminal
```

## Phase 5 — Monitor while it runs

```bash
make ssh
tmux attach -t train        # back to the exact same running session
```

```bash
make sync-log                # (new SSH session) periodic checkpoint-sync / spot-watch journal
```

```bash
make status                  # instance state, type, public IP, AZ
```

If `tmux attach` says no session exists, the run ended — see the troubleshooting
table below rather than assuming it's still fine.

To check if you can push further than the launch batch size without disrupting the
run (safe to do in parallel — the training run keeps going):

```bash
make ssh
cd ~/tiny_llm/from_scratch/custom-gpt-50m
uv run gpt-benchmark --sweep-batch 16,32,48 --warmup-min 2 --measure-min 5
```

## Phase 6 — Day-to-day management

| Situation | Command | Effect |
|---|---|---|
| Pausing overnight, resuming tomorrow on the same box | `make stop` | Compute billing stops; **EBS keeps billing** (~$8/mo/100GB) until you `start` or `down` |
| Resuming a stopped box | `make start` | New public IP — `terraform refresh && make status` after |
| Done with this run, want billing at ~$0 | `make down` | Instance + EBS destroyed; bucket/IAM/SG/key survive, free |
| Launching again after `down` | `make up` | 3-minute bootstrap, pulls whatever's currently in `corpus_prefix`/`checkpoint_prefix` |
| A newer Deep Learning AMI shouldn't silently replace a mid-run box | *(nothing — this is automatic)* | AMI changes are ignored by `lifecycle` until you explicitly `make replace` |
| Home IP rotated, SSH now refused | `make apply` (re-detects `/32`) or `make ssm` | `ssm` needs no inbound rule at all |
| Checking what's actually being billed right now | AWS Console → Billing → Cost Explorer (no simple CLI one-liner — Cost Explorer's API needs a full JSON filter payload, not worth scripting for a spot-check) | Terraform doesn't track spend, only resources — it has no idea what anything actually costs |
| Confirming the budget alarm is actually armed | `terraform plan \| grep aws_budgets_budget` | No line = not armed — `monthly_budget_usd` alone isn't enough, `budget_alert_email` must be set too |

**Always sync checkpoints down before any destructive step:**

```bash
make download-checkpoints PROJECT_DIR=../../from_scratch/custom-gpt-50m
```

Though in practice `checkpoint_sync_minutes` has already been pushing this to S3
throughout the run — this just pulls the Mac's local copy up to date with it.

## Phase 7 — Teardown

```bash
make download-checkpoints PROJECT_DIR=../../from_scratch/custom-gpt-50m
make down                  # or `make destroy` for a full clean slate — see below
```

`make down` is the normal end state between runs — bucket, IAM role, security
group and key pair all stay (all free), and nothing is billing except the few
cents/month of S3 storage.

`make destroy` (`terraform destroy`) additionally removes IAM/SG/key pair — and
**will fail on the bucket** if it still has objects in it and `force_destroy_bucket`
is `false` (the default): S3 refuses to delete a non-empty bucket, so `destroy`
errors out with everything else already gone. Either empty the bucket first
(`aws s3 rm s3://<bucket> --recursive`) or set `force_destroy_bucket = true` in
`terraform.tfvars` before destroying if you genuinely want the corpus and
checkpoint history gone too, not just paused.

## Phase 8 — Using the trained model locally, after `download-checkpoints`

`make download-checkpoints` in Phase 6/7 pulls everything under `checkpoint_prefix`
back into local `checkpoints/50m/` — `latest.pt`, `best.pt`, `serving.pt`, `final.pt`
(whichever exist), mirroring the exact path structure they were uploaded with.

**It's immediately usable — no conversion step.** `checkpoint.py`'s `load_model()`
loads via `map_location`, and every CLI below calls `get_device()` first, which
auto-detects `cuda`/`mps`/`cpu`. A checkpoint trained on the GPU's CUDA loads on the
Mac's MPS (or CPU) with zero code changes — confirmed directly in `cli/infer.py`:

```python
device = get_device()
checkpoint, tokenizer, model = load_model(checkpoint_path, device)
```

```bash
cd from_scratch/custom-gpt-50m   # back on the Mac, from the repo root
```

| Command | What it does |
|---|---|
| `uv run gpt-infer` | Generate text from the model — a prompt, or the held-out test set |
| `uv run gpt-serve` | Run the FastAPI serving endpoint locally |
| `uv run gpt-eval` / `gpt-score` / `gpt-judge` / `gpt-qa-report` | The quality-evaluation pipeline |
| `uv run gpt-train` | Resume training *further*, now on Mac (MPS/CPU) — the reverse of Phase 4, already covered in `docs/MIGRATION.md` |

**`serving.pt` is not a special export format**, even though the name suggests it
might be — it's the exact same checkpoint format as `latest.pt`/`best.pt` (same
`model_state_dict` + `optimizer_state_dict` + metadata), just a copy of *whichever*
checkpoint is picked as "the one to serve," via `resolve_serving_checkpoint()`'s
preference order: best-by-test-loss → latest → final, written automatically at eval
time and again at the end of a run.

**Publishing externally (optional, Hugging Face Hub):**

```bash
uv run python scripts/upload_to_hf.py --repo-id you/your-model
```

Uploads the checkpoint plus the code needed to load it. Separate from everything
above — only needed if the model should be reachable outside this Mac entirely.

---

## Troubleshooting quick reference

| Symptom | Check | Likely cause |
|---|---|---|
| `terraform apply` fails on `aws_key_pair`: no file at `~/.ssh/id_ed25519.pub` | Phase 0 | Key was never generated — run the `ssh-keygen` line |
| `tmux attach -t train` says no session | `tmux ls`, then `dmesg \| tail -50` | Session ended — check for an OOM kill, or the run finished |
| Training banner shows `step 0` on a resume | `checkpoint_prefix` in `terraform.tfvars`; re-check `bootstrap-log`'s checkpoint-sync block | Checkpoint didn't land where `gpt-train` looks |
| `make gpu` doesn't print `True` for bf16 | `terraform.tfvars`' `instance_type` | Wrong instance family (Turing, e.g. `g4dn`, has no bf16) |
| `nvidia-smi` shows ~0% utilization mid-run | scroll up in the attached `tmux` pane | Run already finished, crashed, or the idle watchdog is about to stop the box |
| Instance keeps existing but nothing's training | `make status`, `idle_shutdown_minutes` | The idle watchdog stops the box after N idle minutes — check if it already fired |
| Bill higher than expected between runs | `make status` | A *stopped* (not destroyed) instance still bills ~$8/mo for its EBS volume — `make down` instead |
| `upload-corpus`/`upload-checkpoint` says access denied | `aws sts get-caller-identity`; bucket must exist first | Ran before Phase 1's `make down`, or AWS CLI isn't configured |

## Full command index

```
make init                    Phase 0 — terraform init
make plan                    Any time — see what would change, no side effects
make down                    Phase 1 & 7 — bucket/IAM/SG/key only, or full teardown of the instance
make up                      Phase 3 & 6 — launch the instance
make upload-corpus           Phase 2 — data/*.bin(.json) -> S3
make upload-checkpoint       Phase 2 — checkpoints/ -> S3, for auto-resume
make download-checkpoints    Phase 6 & 7 — S3 -> local checkpoints/
make bootstrap-log           Phase 3 — tail cloud-init boot log
make gpu                     Phase 3 — nvidia-smi + bf16 sanity check
make ssh                     Phase 4+ — SSH in (needs ~/.ssh/id_ed25519)
make ssm                     Phase 6 — Session Manager, no key/port needed
make tunnel                  Optional — forward the model API to localhost:8000
make sync-log                Phase 5 — tail checkpoint-sync / spot-watch journal
make status                  Phase 5 & 6 — instance state/type/IP/AZ
make stop / make start       Phase 6 — pause/resume, EBS keeps billing while stopped
make replace                 Phase 6 — rebuild the instance on the current AMI
make spot-price               Before switching to spot — live price per AZ (see the
                              module's terraform.tfvars comment on spot vs on-demand)
make destroy                  Phase 7 — remove everything Terraform manages
```
