# Operations SOP — from `terraform apply` to teardown (GCP)

The GCP analog of [`infra/aws-gpu-node/docs/SOP.md`](../../aws-gpu-node/docs/SOP.md) —
same phase structure, rewritten against this module's actual commands. The "why" for
each design choice lives in [`../README.md`](../README.md) and
[`GCP_CONCEPTS.md`](GCP_CONCEPTS.md); this page is "what to type," in order.

**This is a living document, not a one-off writeup.** It's updated in place as later
sessions happen (porting `custom-gpt-153m`, a second `custom-gpt-50m` run, changing
GPU type, resolving the open issue below) — add to it, don't fork a parallel doc.
Every command below was actually run and its real output is what's described; nothing
here is speculative. Phases marked **PENDING** below haven't been executed yet in this
project's actual history — don't treat them as verified until their marker is updated.

Worked example throughout: `custom-gpt-50m`, resuming an existing run (~step 677K/1M),
on-demand, `us-central1` — this directory's current `terraform.tfvars`. This module had
**never been applied before this session** (2026-08-17) — several bugs below were found
because of that, not because anything changed under it.

All commands assume `cd infra/gcp-gpu-node` unless stated otherwise.

---

## Known issues (read before re-running any of this)

Found and fixed during first real `apply` of this module — fixes are already in the
`.tf` files, described here so a future session doesn't waste time rediscovering them:

| Issue | Where | Fix |
|---|---|---|
| `terraform.tfvars.example`'s `repo_url` points at `mini-llms-playground.git`, which this repo no longer has (renamed at some point, example never updated) | `terraform.tfvars.example` | Verify with `git remote -v` in the repo root before trusting the example — actual remote at time of writing: `https://github.com/shukla-surendra/tiny_llm.git` |
| `variables.tf`'s default `image_family = "common-cu124-debian-12"` 404s — `ml-images` has moved its published families on | `variables.tf` default, overridden in `terraform.tfvars` | Find the current one: `gcloud compute images list --project ml-images --no-standard-images \| grep cu1`, verify with `gcloud compute images describe-from-family <family> --project ml-images`. Working as of 2026-08-17: `common-cu129-ubuntu-2204-nvidia-580` |
| `budget.tf`'s `billing_account = "billingAccounts/${var.billing_account_id}"` double-prefixes — the provider already prepends `billingAccounts/`, producing a 404 on `.../billingAccounts/billingAccounts/<id>/budgets` | `budget.tf` | Fixed: `billing_account = var.billing_account_id` (bare id) |
| `budget.tf`'s `budget_filter.projects` used `"projects/${var.project_id}"` (the string project id) — Cloud Billing Budgets silently requires the **numeric project number** here, unlike every other project reference in this module | `budget.tf` | Fixed: added `data "google_project" "this"`, use `"projects/${data.google_project.this.number}"` |

**Open, unresolved**: even after both fixes above, `google_billing_budget.monthly`
still fails with a generic `400: Request contains an invalid argument` — reproduced
with the notification-channel linkage entirely removed too (a bare budget: just
`amount` + `budget_filter` + `threshold_rules`), so it's not the notification wiring.
Leading hypothesis: `currency_code = "USD"` (hardcoded in `budget.tf`) doesn't match
this billing account's actual currency — GCP requires an exact match and returns
exactly this unhelpful generic error on mismatch. Couldn't confirm: the deploying
service account (`llm-training-dev-sa`) lacks `billingAccounts.get`, so
`gcloud billing accounts describe XXXXXX-XXXXXX-XXXXXX` 403s. **Disabled for now**
(`monthly_budget_usd = 0` in `terraform.tfvars`, `billing_account_id`/
`budget_alert_email` commented out) rather than guess against a live billing API.

**To re-enable**: confirm the billing account's currency in the Cloud Console
(Billing → your account → Account settings), fix `currency_code` in `budget.tf` if
it's not USD, uncomment the three lines in `terraform.tfvars`, `make plan` / `make
apply`. Until then — **no automated cost guard exists**. See Phase 6's day-to-day
table for the manual-monitoring fallback this makes mandatory, not optional.

**Also found: `GPUS_ALL_REGIONS` quota was 0 project-wide.** A brand-new project's
per-GPU-type quotas (e.g. `NVIDIA_L4_GPUS=1` in `us-central1`, checked with
`gcloud compute regions describe <region> --format=json` and grepping `quotas[]`)
are meaningless until this separate, *global, non-regional* aggregate cap is also
raised — `terraform apply -var instance_count=1` creates the instance resource, GCP
then fails to attach the GPU, and the box self-transitions to STOPPING within
seconds. **This does not bill** (confirmed: `gcloud compute instances list` showed it
mid-STOPPING, `gcloud compute instances delete` cleaned it up, no charge for a
never-`RUNNING` instance) but Terraform's `apply` errors out before recording the
instance in state, so it must be deleted by hand (`gcloud compute instances delete
<name> --zone=<zone> --project=<project> --quiet`) before retrying — otherwise it's
orphaned from Terraform's view even though it exists in GCP.

Fix — request the quota increase via `gcloud`, no Console UI needed (needs the
`cloudquotas.googleapis.com` API enabled first):
```bash
gcloud services enable cloudquotas.googleapis.com --project=<project_id>
gcloud alpha quotas preferences create --quiet \
  --service=compute.googleapis.com \
  --project=<project_id> \
  --quota-id=GPUS-ALL-REGIONS-per-project \
  --preferred-value=1 \
  --email=<your email> \
  --justification="<why>"
  # NOTE: no --dimensions=region=... — this quota is global, not regional; passing
  # a region dimension 400s with "quota is not regional"
```
Check status: `gcloud alpha quotas preferences describe <name-from-create-output> --project=<project_id>`.
For a single L4, this was auto-approved in ~2 seconds on 2026-08-17 — no manual
Google review wait. Don't assume that's guaranteed for every account/quota amount,
but it's worth trying the API path first before assuming a multi-day Console request
is required.

**Also found: `us-central1-a`, `-b`, AND `-c` all STOCKOUT on `g2-standard-4`/L4.**
Three consecutive `apply` attempts, one per zone, each failed with `STOCKOUT` and each
error's "try zone X instead" hint pointed at a zone already tried (circular, not
useful signal — don't trust it). This is genuine regional GPU scarcity, not a config
problem. **Fix**: cross-region fallback. `region` (used only for the *bucket*'s
location, `storage.tf`) and `zone` (used only for the *instance*, `main.tf`) are
independent variables with no validation tying them together — so `zone` can point at
`us-east4-a` while `region` stays `us-central1`, and the already-uploaded bucket
never has to move or be re-uploaded. Cost of doing this: the corpus/checkpoint pull
at boot becomes a small one-time cross-region GCS transfer (~2.3GB, a few cents)
instead of free in-region. `us-east4-a` had capacity on the first try. If it doesn't
next time, `us-west1-a` is the other zone this module's `variables.tf` names as a
fallback.

**Also found: `bootstrap.sh.tftpl` passes `--only-show-errors=false` to `gcloud
storage rsync`, which is not a recognized flag** (`gcloud storage rsync --help` has
no such flag at all) — every rsync call in the template failed outright, silently
(caught by the script's own `|| echo 'WARN: ... failed'`, so bootstrap still reported
"done" while corpus, checkpoint, AND the recurring checkpoint-sync systemd service
were all non-functional). Fixed in the template (flag removed from all 4 occurrences:
initial corpus pull, initial checkpoint pull, the periodic checkpoint-sync script, and
the preempt-watch sync-on-interrupt path) — future launches are unaffected. **If you
are looking at a box that booted BEFORE this fix landed**, the corpus/checkpoint
won't be there and the periodic sync won't be running even though bootstrap logged
"done" — check manually:
```bash
ls ~/tiny_llm/from_scratch/custom-gpt-50m/data/       # should have train.bin etc.
sudo systemctl status checkpoint-sync.service          # should be active/running
grep only-show-errors /usr/local/bin/checkpoint-sync.sh  # non-empty = still broken
```
Manual fix on an already-booted box (what was actually run on 2026-08-17, since the
box was mid-billing and re-launching to pick up the template fix wasn't worth the
extra boot-and-clone cycle):
```bash
gcloud storage rsync "gs://<bucket>/<corpus_prefix>" "$WORK_DIR/data/" --recursive
gcloud storage rsync "gs://<bucket>/<checkpoint_prefix>" "$WORK_DIR/checkpoints/" --recursive
sudo sed -i "s/ --only-show-errors=false//" /usr/local/bin/checkpoint-sync.sh
sudo systemctl restart checkpoint-sync.service
sudo -u <ssh_user> /usr/local/bin/checkpoint-sync.sh --once   # verify: prints "[ckpt-sync] ok"
```

**Also found: the `self_stop` IAM binding fails too, one permission deeper than the
custom-role issue above.** Even the predefined-role fallback (`roles/compute.instanceAdmin.v1`
bound at the instance level, see `iam.tf`) 403s: `Required
'compute.instances.setIamPolicy' permission ... forbidden`. This service account can
create the instance itself but can't grant IAM on it afterward. **Disabled**
(`google_compute_instance_iam_member.self_stop`'s `count` hardcoded to `0` in
`iam.tf`, with the real conditional commented alongside it). Effect: the idle-shutdown
and spot-preemption watchdog *scripts* are still installed and running on the box (they're
just shell scripts, no special IAM needed to run), but when either one decides the
instance should stop, the `gcloud compute instances stop` call inside them will 403 —
**they will detect the condition and fail to act on it**. This makes Phase 6's
manual-monitoring requirement apply here too, not just for the missing budget alert.
Re-enable by restoring the commented `count` line in `iam.tf` once the deploying
principal has `compute.instances.setIamPolicy` (or ask a project Owner to grant it
directly: `gcloud compute instances add-iam-policy-binding <name> --zone=<zone>
--member=serviceAccount:<sa-email> --role=roles/compute.instanceAdmin.v1`).

**Also found: launching straight at a large batch size without measuring first
OOMs.** The checkpoint was trained at `batch_size=1, grad_accum_steps=32` (the Mac/MPS
default) — jumping straight to `GPT_BATCH_SIZE=32 GPT_GRAD_ACCUM=1` on the L4 (same
effective batch, seemingly an obvious win) OOM'd instantly: `Tried to allocate 6.14
GiB` against `22.03 GiB` total with `21.15 GiB` already in use. `gpt-benchmark
--sweep-batch 1,2,4,8,16 --warmup-min 0.2 --measure-min 0.3` (module: 50m, E=512 L=8
ctx=1024) showed why — peak VRAM at batch=16 was only 11.0 GiB (32 would roughly
double that, right at the 22GB ceiling), AND **throughput actually peaks around
batch=4** (56,888 tok/s) with batch=8/16 flat-to-slightly-worse (56,350 / 55,308) —
this model is too small for a bigger batch to help at all, so there was no upside
being chased in the first place. Actually launched at `GPT_BATCH_SIZE=4
GPT_GRAD_ACCUM=8` (effective batch 32, unchanged from the checkpoint, ~29% faster
than the original `batch=1` throughput per the sweep). **Lesson for next time**: run
`gpt-benchmark --sweep-batch` before picking a batch size on a new GPU type, don't
extrapolate from VRAM headroom alone — it cost one OOM'd launch (a few seconds of
wasted billed time, not serious, but avoidable).

## Phase 0 — One-time local setup ([DONE] 2026-08-17)

```bash
brew install --cask google-cloud-sdk   # gcloud CLI — already present this session
gcloud init                             # already authenticated as llm-training-dev-sa
gcloud auth application-default login   # Terraform's own credentials
gcloud services enable compute.googleapis.com storage.googleapis.com \
  iam.googleapis.com cloudbilling.googleapis.com billingbudgets.googleapis.com \
  monitoring.googleapis.com iap.googleapis.com cloudresourcemanager.googleapis.com
  # monitoring + storage + bigquery-family APIs were already enabled on this project
  # from prior (non-GPU) use; compute/cloudbilling/iam/billingbudgets/iap/
  # cloudresourcemanager were enabled fresh this session.

ls ~/.ssh/id_ed25519.pub 2>/dev/null || \
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "mini-llm-gpu"
  # already present this session, no generation needed

cp terraform.tfvars.example terraform.tfvars
```

**`terraform.tfvars` values actually used** (see the file itself for the full set —
this lists only what differs from the example or needed a real value filled in):

```
project_id          = "llm-training-dev"     # confirmed: billing enabled, GPU quota
                                              # sufficient (NVIDIA_L4_GPUS limit=1 in
                                              # us-central1) with NO quota-increase
                                              # request needed
repo_url            = "https://github.com/shukla-surendra/tiny_llm.git"  # see Known issues
image_family        = "common-cu129-ubuntu-2204-nvidia-580"              # see Known issues
monthly_budget_usd  = 0    # see Known issues — budget resource currently broken
use_spot            = false  # on-demand: resuming an already-hours-invested run,
                              # not worth GCP Spot's ~30s preemption risk for this
```

`billing_account_id` for reference (not currently used, budget disabled):
`XXXXXX-XXXXXX-XXXXXX`, found via `gcloud billing accounts list`.

## Phase 1 — Provision, no billing yet ([DONE] 2026-08-17)

```bash
make init
make plan     # read it
make down     # apply -var instance_count=0: everything except the instance
```

Same "not turning something off, nothing exists yet" semantics as the AWS SOP —
`make down` here created the 8 free resources (bucket, service account, IAM binding,
2 firewall rules; budget + notification channel were part of this too before being
disabled, see Known issues) and created **zero** instances, since `instance_count`
only drives the 9th resource.

**Verified after this phase**:
```
$ gcloud compute instances list --project=llm-training-dev
Listed 0 items.
$ gcloud storage du gs://mini-llm-gpu-llm-training-dev-us-central1 --summarize
0            gs://mini-llm-gpu-llm-training-dev-us-central1
```
Zero compute billing, empty bucket (zero storage billing beyond the free tier).

## Phase 2 — Upload data ([DONE] 2026-08-17)

```bash
# Corpus (always). Ships *.bin + *.bin.json only — never train.txt.
make upload-corpus PROJECT_DIR=../../from_scratch/custom-gpt-50m

# Checkpoint (resuming an existing run — this one is)
make upload-checkpoint PROJECT_DIR=../../from_scratch/custom-gpt-50m
```

Local training was stopped cleanly first (`make train-stop` in
`from_scratch/custom-gpt-50m`, 2026-08-17 — confirmed process exited, checkpoint saved
at the moment of stopping, not mid-write) specifically so the checkpoint uploaded here
is the final, settled state of that run, not a race with a still-running trainer.

Confirm before moving on:
```bash
gcloud storage ls "gs://$(terraform output -raw bucket)/50m/corpus/"
gcloud storage ls "gs://$(terraform output -raw bucket)/50m/checkpoints/" --recursive
```

**Verified after this phase** (2026-08-17):
```
gs://mini-llm-gpu-llm-training-dev-us-central1/50m/corpus/{test,train}.bin{,.json}
gs://mini-llm-gpu-llm-training-dev-us-central1/50m/checkpoints/50m/{best,latest,serving}.pt
```
Note the checkpoint object keys land at `50m/checkpoints/50m/*.pt`, not
`50m/checkpoints/*.pt` — `make upload-checkpoint` rsyncs local `checkpoints/` (which
contains a `50m/` subfolder) into the `50m/checkpoints/` bucket prefix, so the local
subfolder name is preserved underneath it. This is correct, not a bug — it round-trips
exactly with `download-checkpoints` and the bootstrap script's pull step (both rsync
the same prefix back into local `checkpoints/`, landing at `checkpoints/50m/latest.pt`
again). Total bucket size: 2.30 GiB.

## Phase 3 — Launch ([DONE] 2026-08-17)

```bash
make up                    # BILLING STARTS THE MOMENT THE INSTANCE EXISTS
make bootstrap-log         # tail cloud-init: uv sync, repo clone, corpus pull,
                            # checkpoint pull — watch it finish clean
make gpu                   # expect: NVIDIA L4, bf16 True
```

`estimated_hourly_usd` (Terraform output) = **0.70** for `g2-standard-4` on-demand.
With the cost guard currently disabled (see Known issues), there is nothing
automatic stopping spend once this step runs — Phase 6's manual-monitoring
requirement is not optional until the budget is fixed.

**What actually happened**: `us-central1-a/-b/-c` all stocked out (see Known issues)
— launched in `us-east4-a` instead, `region` left at `us-central1` for the bucket.
Instance came up in 26s. `nvidia-smi` in the bootstrap log confirmed `NVIDIA L4`,
driver `580.173.02`, CUDA `13.0` before `uv sync` even started. `make bootstrap-log`
itself uses `tail -f` (blocks forever) — on a Mac with no `timeout`/`gtimeout`
installed, use a bounded snapshot instead: `ssh ... 'tail -n 300
/var/log/gpu-node-bootstrap.log'` (no `-f`), repeat until `=== bootstrap done ===`
appears. Corpus/checkpoint sync both failed silently during this bootstrap (the
`--only-show-errors=false` bug, see Known issues) — fixed manually post-boot, see
that section for the exact commands used. `uv sync` pulled ~2.5GB of CUDA wheels
(torch 502MB alone) in ~35s. `make gpu` output: `NVIDIA L4 True`.

## Phase 4 — Start (or resume) training ([DONE] 2026-08-17)

**Never run `gpt-train` directly in the SSH session you're typing in** — use `tmux`.

```bash
make ssh                   # or: make iap-ssh (no open port, no IP dependency)
cd ~/tiny_llm/from_scratch/custom-gpt-50m   # repo_dir_name = tiny_llm, not
                                             # mini-llms-playground — see Known issues
tmux new -s train
```

Check what the checkpoint was trained under before picking new batch values:

```bash
python3 - <<'PY'
import torch
ckpt = torch.load("checkpoints/50m/latest.pt", map_location="cpu")
print("step:", ckpt.get("step"))
print("batch_size:", ckpt.get("batch_size"), "grad_accum_steps:", ckpt.get("grad_accum_steps"))
PY
```

```bash
uv run gpt-train   # add GPT_BATCH_SIZE=/GPT_GRAD_ACCUM= only if deliberately changing
```

Confirm the banner: `device=cuda`, bf16, and — since this is a resume —
`Resumed from checkpoints/50m/latest.pt at step <N>` with `N` matching the metadata
check above, not `0`.

```
Ctrl-b  d          # detach — training keeps running
```

**What actually happened**: checkpoint metadata check showed `step: 695057,
batch_size: 1, grad_accum_steps: 32` (batch=1 was the Mac/MPS default). Do NOT just
bump batch_size blindly on a new GPU — see Known issues' benchmark story for why
`GPT_BATCH_SIZE=32 GPT_GRAD_ACCUM=1` OOM'd and `GPT_BATCH_SIZE=4 GPT_GRAD_ACCUM=8`
(same effective batch of 32) was the actual launch command, chosen from a
`gpt-benchmark --sweep-batch` run, not a guess. Startup banner confirmed:

```
Model: 50m  |  51,475,968 parameters  |  device=cuda  |  attn_impl=sdpa
Precision: torch.bfloat16  |  batch 4 x accum 8 = 32 seqs/update
Budget: 1,000,000 steps x 4,096 tok = 4.10B tokens (79.6 tok/param, 14.71 epochs)
Resuming from checkpoints/50m/latest.pt...
Resumed at step 695058 (cumulative 34:36:42)
Progress: step 695,058/1,000,000 (69.5%)
ETA: 15.2 more training-hours (0.6 days)
```

`step 695058` matches the pre-launch metadata check (695057 saved -> 695058 resumed,
the expected off-by-one). Launched inside `tmux new-session -d -s train` (detached
from the start, via `tmux send-keys`, rather than attach-then-Ctrl-b-d — same end
state, no interactive terminal needed). Post-launch GPU check: `nvidia-smi
--query-gpu=utilization.gpu,memory.used,memory.total --format=csv` -> `98 %, 4760
MiB, 23034 MiB` — genuinely training, not idling, comfortable memory headroom.

## Phase 5 — Monitor while it runs ([DONE, initial check] 2026-08-17)

```bash
make ssh && tmux attach -t train    # back to the exact same session
make sync-log                        # (new session) checkpoint-sync / preempt-watch journal
make status                          # instance state, type, IP, zone
```

Initial post-launch check done (98% GPU util, ~13.5 steps/sec). Ongoing monitoring
for the remainder of this run is ***not*** yet done as of this doc update — with
both the budget alert and the self-stop watchdog disabled (see Known issues), **this
run needs to be checked on manually and stopped manually (`make down`) when it's far
enough along or you're done for the session** — nothing will do either automatically.

### Speed investigation: GCP L4 is only ~2.4x the MacBook (2026-08-17)

Raised by the user after the resume: observed L4 throughput (~13.3 steps/sec, 20,082
steps/hr) is only ~2.4x the checkpoint's own cumulative-average rate from its mostly-MPS
history (34:44:17 / 699,507 steps ≈ 5.58 steps/sec) — underwhelming for a dedicated
GPU vs. a laptop's integrated one. Investigated rather than assumed-fine:

`gpt-benchmark --sweep-batch 1,2,4,8,16` (this model: 51.5M params, E=512 L=8 ctx=1024)
showed **MFU tops out at 14.5-14.8%** regardless of batch size — this model is too
small to saturate an L4's compute; the bottleneck is per-step overhead (kernel launch,
Python dispatch), not raw FLOPs. That headroom made `torch.compile` (kernel fusion,
explicitly flagged as "untested, likely a further speedup" in
[`../GPU_TRAINING.md`](../../../from_scratch/custom-gpt-153m/docs/GPU_TRAINING.md)'s
sibling doc's "Known gaps") worth actually testing rather than dismissing.

**Added `GPT_COMPILE=1` support** to `benchmark.py` and `trainer.py` (both projects'
model construction point, gated behind the env var — `model = torch.compile(model)`
right after `TinyGPT.from_config(...).to(device)`, before the optimizer/resume, since
`load_state_dict`/`state_dict` round-trip transparently through `torch.compile`'s
wrapper in torch 2.13). Pushed via `scp` directly to the box for testing (local edits
aren't pushed to `tiny_llm.git` — see Phase 4's `repo_dir_name` note) — **not
committed**, since committing wasn't requested; if you want this kept, it needs an
explicit commit later, or it'll only exist on this box and in the local working tree.

First attempt failed: `torch._inductor.exc.InductorError` — Triton (torch.compile's
JIT backend) shells out to `gcc` to build a small CUDA-utils extension, and that
`gcc` invocation failed. The actual error was hidden (`subprocess.check_call(...,
stdout=subprocess.DEVNULL)` swallows it) — reproduced manually outside `uv run` to
surface it: **`fatal error: Python.h: No such file or directory`**. The
`ml-images`/`common-cu129-...` boot image ships `gcc` but not `python3.10-dev`, and
`bootstrap.sh.tftpl` only `apt-get install`s `tmux jq git` — never `python3-dev`.
Fixed live: `sudo apt-get install -y python3.10-dev`. **Not yet added to
`bootstrap.sh.tftpl`** — do that before relying on `GPT_COMPILE=1` working out of the
box on a future fresh launch; for now it's a manual step.

**Result, once the compiler issue was fixed**: `GPT_COMPILE=1` at batch=4 measured
**14.18 steps/sec / 58,084 tok/s / MFU 14.8%**, vs. baseline **13.89 steps/sec / 56,888
tok/s / MFU 14.5%** — a **~2% improvement**, not the 2-3x the MFU headroom might have
suggested. **Conclusion: torch.compile is not worth adopting for this model on this
GPU** — the bottleneck the 14% MFU points at isn't primarily kernel-launch/dispatch
overhead (which compile fixes), it's something compile doesn't touch (data
loading between micro-batches, `grad_accum` Python-loop overhead, or the model
genuinely being too small for the L4's compute units regardless of scheduling
efficiency — not further isolated, given the marginal payoff didn't justify more
investigation time while the instance was paused and billing regardless). Reverted to
the non-compiled launch command (`GPT_BATCH_SIZE=4 GPT_GRAD_ACCUM=8 uv run gpt-train`,
no `GPT_COMPILE`) — confirmed resumed cleanly at `step 699507`, matching the exact
step training was paused at (`tmux send-keys -t train C-c`, waited for the "Saved:
checkpoints/50m/latest.pt" line, confirmed clean before relaunching).

**Second hypothesis tested: the `.item()` call in the tqdm progress display.**
`trainer.py`'s main loop calls `loss.item()` on *every micro-step* (line ~400, to
build the `batch_loss` postfix string), which forces a CPU-GPU synchronization —
`gpt-benchmark`'s own timing loop never does this (`run_window()` in `benchmark.py`
only checks `time.perf_counter()`, no sync), so it doesn't rule this out. Wrote an
isolated script (`sync_test.py`, copied to the box, not committed — same
not-pushed-to-git situation as the `GPT_COMPILE` changes) replicating the trainer's
exact micro-batch/backward/accum-boundary pattern with and without the sync call,
run back-to-back on the paused, uncontended GPU:

```
WITH  .item() every micro-step: 13.54 steps/s
WITHOUT .item() every step:      13.46 steps/s
Difference: -0.6%  (noise, not a real effect)
```

Ruled out — the sync call costs nothing measurable at this batch/accum size (each
micro-step's forward+backward already takes far longer than a scalar sync).

## Why the ceiling is ~13.2-13.5 steps/sec (documented answer)

Two concrete, testable "software inefficiency" hypotheses were raised and both came
back negative:

| Hypothesis | Test | Result |
|---|---|---|
| Kernel-launch/Python-dispatch overhead | `GPT_COMPILE=1` (`torch.compile`, kernel fusion) | +2% only |
| Per-step CPU-GPU sync stall (`loss.item()` in the progress bar) | Isolated with/without comparison, uncontended GPU | -0.6% (noise) |

What's left, and what actually explains it: **`gpt-benchmark --sweep-batch
1,2,4,8,16` showed MFU (compute utilization) pinned at 14.5-14.8% across every batch
size tested — including batch=8 and batch=16, which measured the same or slightly
*worse* throughput than batch=4, not better.** More work per step *not* translating
to more throughput is the signature of a **memory-bandwidth-bound** workload, not a
compute-bound one: at this model's shape (`E=512, L=8, ctx=1024` — 51.5M params),
individual matmuls are small enough that the GPU spends more time moving weights and
activations through memory than doing arithmetic on them once they're loaded. Neither
kernel fusion (fixes *launch* overhead, not memory traffic) nor removing sync points
(fixes *pipeline stalls*, not the underlying memory movement) touches that — it's a
property of the computation's shape relative to the L4's own memory bandwidth
(~300GB/s — explicitly the lowest of the GPU options this project's own
`docs/GPU_TRAINING.md` considered, chosen for cost over an A10G's 600GB/s).

**In short: this isn't a bug, a misconfiguration, or something fixable in software at
this batch/model size — it's the actual arithmetic intensity of a 51M-param model on
this specific GPU's memory subsystem.** The ~2.4x over the MacBook is a real, if
modest, speedup; it isn't larger because the model was never big enough to need an L4
in the first place (the same MFU ceiling is likely present on any Ampere/Ada card at
this size — untested here, but consistent with the pattern being architectural, not
L4-specific).

### Further plan to speed this up (if ever worth pursuing)

None of these were tried — listed in rough order of expected payoff vs. effort,
for a future session to pick up:

1. ~~A GPU with more memory bandwidth~~ — **investigated 2026-08-17, see the dedicated
   section below ("Next GPU to test — pricing/bandwidth analysis").** Short version:
   no clear win available on GCP at current prices — skip straight to that section
   rather than re-deriving this.
2. **`torch.compile(mode="reduce-overhead")` specifically (CUDA graphs)** — only
   plain `torch.compile(model)` (default mode) was tested. `reduce-overhead` mode
   uses CUDA graphs to eliminate *per-kernel* launch overhead more aggressively than
   default inductor fusion, which is a different mechanism than what was measured
   here (2% gain). Worth a real test before assuming compile categorically doesn't
   help — the default-mode result doesn't fully rule this out. Caveat: CUDA graphs
   require static shapes/no data-dependent control flow between calls; verify
   `get_batch`'s output shapes are actually constant across calls before trying.
3. **Multiple small models trained in parallel on the same GPU** (not faster per
   model, but better $/model given the GPU is provably underused at 14.5% MFU) —
   e.g. two training processes sharing the L4 via MPS (Multi-Process Service) could
   plausibly get closer to 2x total throughput for near-$0 extra cost, since there's
   genuine headroom being left on the table. Relevant only if there's a second model
   worth training concurrently (e.g. `custom-gpt-10m`), not for speeding up this one
   run.
4. **Restructuring `grad_accum` into fewer, larger micro-batches isn't it** —
   already tested indirectly (`gpt-benchmark --sweep-batch` showed batch=8/16 flat
   vs. batch=4), don't re-try this without a new reason to think it'd differ from
   the measured result.
5. **Accept it.** At ~14.5 hours remaining, chasing a further speedup costs real
   engineering time against a training run that's most of the way done regardless.
   The economically rational choice, absent a next run where this ceiling would
   recur across many more GPU-hours, is probably this one.

### Next GPU to test — pricing/bandwidth analysis (2026-08-17)

Answering "what's the next affordable GPU, considering price, to beat the L4's
~13.3 steps/sec" — **for planning the next training run**, not acted on this session
(the run this analysis was done for was already stopped/torn down before this was
written). Read this before picking a `machine_type` for that next run.

**Framing**: since throughput is memory-bandwidth-bound, not compute-bound (flat MFU
regardless of batch size — see above), the metric that matters is **GB/s of memory
bandwidth per dollar**, not raw price or raw FLOPs. Pulled real, current GCP pricing
via the public Cloud Billing Catalog API (`cloudbilling.googleapis.com`, service
`6F81-5844-456A` = Compute Engine) rather than trust memorized/stale numbers —
command used, if repeating this for a future GPU generation or region:

```bash
TOKEN=$(gcloud auth print-access-token)
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?pageSize=5000&pageToken=<token>"
# paginate via nextPageToken in the response; filter category.resourceGroup == "GPU"
```

| GPU | Bandwidth (published spec) | bf16 tensor cores? | On-demand ($/hr, GPU only, us-central1) | Spot ($/hr) | GB/s per $ (on-demand) | GB/s per $ (spot) |
|---|---|---|---|---|---|---|
| T4 | 320 GB/s | **No** (Turing — fp16-only) | $0.35 | $0.18 | 914 | 1,768 |
| **L4 (used this session)** | 300 GB/s | Yes | $0.56 | $0.27–0.34 | 536 | 893–1,107 |
| V100 | 900 GB/s | **No** (Volta — fp16-only) | $2.48 | $1.29 | 363 | 698 |
| A100 40GB | 1,555 GB/s | Yes | $2.93 | $1.69 | 531 | 920 |
| A100 80GB | 2,039 GB/s | Yes | $3.93 | $2.27 | 519 | 898 |

(GPU-line-item price only — the full instance also carries vCPU/RAM cost on top,
somewhat higher for the `a2` family than `g2-standard-4`'s modest 4vCPU/16GB, not
itemized here. us-east4 was a few % cheaper than us-central1 on some rows; not
broken out above, re-check both when actually pricing a specific launch.)

**Conclusion: there is no clear win available.**

- **A100 (40GB or 80GB) has essentially the same GB/s-per-dollar as L4** (~520-531
  vs. 536 on-demand; ~898-920 vs. 893-1,107 on spot). GCP prices GPUs roughly
  proportional to their bandwidth at this tier, so a 5x-more-bandwidth card costs
  ~5x more — total cost to finish a fixed amount of *bandwidth-bound* work comes out
  close to a wash under perfect linear scaling, and **likely worse** in practice,
  since the fixed per-step overheads already measured (data loading, Python loop —
  small individually, ~2% and ~0.6% respectively) don't shrink just because memory
  got faster, so realistic scaling is sub-linear.
- **T4 has the best GB/s-per-dollar on paper** (914-1,768) but **no bf16 tensor
  cores** (Turing architecture). This codebase's `resolve_amp()` only implements
  bf16-on-CUDA or fp32-fallback — no fp16/`GradScaler` path exists anywhere in
  `trainer.py`/`benchmark.py`. Running on T4 today means silently training in
  **fp32**, which roughly doubles bytes moved per element and would likely erase
  (or reverse) T4's price advantage on a bandwidth-bound workload. Real value from
  T4 would require adding fp16 + `GradScaler` support first — a genuine code change
  with gradient-underflow risk (the reason bf16 was chosen originally, per
  `docs/GPU_TRAINING.md`'s "Instance choice" section) — not a config flag, and
  untested here.
- **V100 is dominated by A100 on both axes** (worse GB/s/$ *and* no bf16) — no
  scenario where it's the right choice over A100 or L4.

**Bottom line for planning the next run**: L4 is already close to the efficient
choice for this workload at current GCP prices — there's no cheap drop-in upgrade
that clearly beats it once architecture (bf16 requirement) and realistic
(sub-linear, not idealized-linear) bandwidth scaling are accounted for. If a next
session wants to test A100 empirically anyway rather than trust this analysis:
`NVIDIA_A100_GPUS` quota is currently **0** in this project (same
`gcloud alpha quotas preferences create` request that worked in ~2s for L4 earlier
this session — not guaranteed equally fast for A100, a costlier/more-contended SKU),
and the validation step before committing a full run should be the same
`gpt-benchmark --sweep-batch` measurement done on L4 here, not an assumption that
more bandwidth number = proportionally more steps/sec.

## Session 2: custom-gpt-153m speed measurement (2026-08-17)

A second, separate use of this same module — swapped `project_subdir`/`corpus_prefix`/
`checkpoint_prefix` in `terraform.tfvars` from `custom-gpt-50m`/`50m/...` to
`custom-gpt-153m`/`153m/...`, everything else (bucket region, image, on-demand,
disabled budget/self-stop) unchanged. **Goal was explicitly to measure and document
153m's training speed on L4, not to run its full pretraining budget to completion** —
the run below was intentionally stopped early once stable numbers were captured, then
torn down, the same cost-conscious pattern as the 50m session's mid-run stop.

**Corpus**: the 1.185B-token pretraining corpus built earlier this session
(cosmopedia-v2 + cosmopedia v1 + finemath-4plus, leak-free train/test split — see
`custom-gpt-153m/DATASET.md` and that project's own `data/raw/*/SOURCE.md` files for
how it was built). This is a **fresh pretraining run**, not a resume — no checkpoint
existed for 153m before this session.

**Zone stockouts, worse than the 50m session's**: `us-central1-a/-b/-c` (already known
stocked out from the 50m session) — not re-tried. `us-east4-a` (worked for 50m) now
stocked out too, its own error message pointed at `us-east4-c`, which *also* stocked
out. Succeeded on `us-west1-a` (the second documented fallback in `variables.tf`,
untried until now) on the first attempt. **Lesson**: L4 zone availability is genuinely
volatile session-to-session, not something to hardcode confidence in — the "known
issues" zone-fallback list existing at all (rather than picking one zone and trusting
it) is doing real work, not just documentation-for-its-own-sake.

**Bootstrap**: ran clean end-to-end with no manual intervention this time — the
`--only-show-errors=false` rsync bug fixed in `bootstrap.sh.tftpl` after the 50m
session held for this launch too (corpus pulled correctly on first boot, byte-identical
to what was uploaded). The `python3.10-dev` gap (needed only for `torch.compile`
testing, not for normal training) was never hit since `GPT_COMPILE` wasn't used this
run.

### Measured speed

`gpt-benchmark --sweep-batch 8,16,24` (project's own default config: batch=16,
grad_accum=4, effective batch 64):

| batch | steps/s | tok/s | peak VRAM | MFU |
|---|---|---|---|---|
| 8 | 3.08 | 25,270 | 10.1 GiB | 19.1% |
| **16 (default)** | **1.46** | **23,974** | **17.5 GiB** | **18.2%** |
| 24 | OOM | — | — | — |

batch=8 measured ~5% faster raw throughput, but was **not** used for the actual run —
switching to it would mean either halving the token budget for the same 150,000
`steps` (steps × batch_size × ctx_len is the total-tokens formula this codebase uses,
independent of `grad_accum`) or recalibrating `steps`/the LR schedule to compensate,
for a 5% gain not worth that risk. Launched with the project's own defaults, unmodified.

**Live training-loop measurement** (not the synthetic benchmark — the actual
`gpt-train` run, stopped intentionally at step 440 after ~5 minutes once the rate
stabilized): **1.45 steps/sec** — agrees with the benchmark's 1.46 to within noise,
a useful cross-check that the benchmark tool's numbers are trustworthy for planning
without needing a live run every time in the future.

- **Throughput**: 1.45 × 16 × 1024 ≈ **23,757 tok/s**
- **Full budget (150,000 steps / 2.46B tokens) estimate**: 150,000 / 1.45 / 3600 ≈
  **28.74 hours**, ≈ **$20.11** at $0.70/hr (GCP `g2-standard-4` on-demand) — matches
  the benchmark tool's own built-in projection (28.5h / $22.91, using its default
  AWS `g6.xlarge` $0.8048/hr price rather than the actual GCP rate) closely enough to
  trust either one for planning.
- Confirms `DATASET.md`'s own claim directly: `2.09 epochs` printed in the training
  banner itself, over 1.183B unique tokens against the 2.46B-token budget — exactly
  the "1-2.5B tokens → still train 153M, accept 1.5-2.5 epochs" guidance from that
  doc's sizing table, now empirically confirmed rather than just asserted.

### Speed vs. model size, same GPU (L4) — the full picture across this session

| Model | Params | steps/sec (measured) | tok/s | Source |
|---|---|---|---|---|
| 50m | 51.5M | 13.3 | ~56,900 | Live training, this session (see "Speed investigation" above) |
| **153m** | **152.8M** | **1.45** | **~23,760** | **Live training, this session (above)** |
| 350m (E=1024,L=24) | 354.8M | ~3-4 (estimated) | ~12,000-17,000 (estimated) | FLOPs/roofline estimate only, never run — see "Next GPU to test" section |

Note 153m's steps/sec (1.45) is *lower* than the 350m estimate's range (3-4) despite
153m being the smaller model — not a contradiction: steps/sec depends on the *chosen
batch size* per model (153m's batch=16 processes 16,384 tok/step; the 350m estimate
assumed batch=4, 4,096 tok/step, a quarter as much per step). **tok/s is the
comparable metric across model sizes**, and there tok/s does monotonically decrease
with model size as expected (56,900 → 23,760 → ~12,000-17,000), consistent with more
FLOPs/token dominating even as MFU (compute efficiency) improves for the larger
models — the same roofline argument made in "Why the ceiling is ~13.2-13.5 steps/sec"
above, now with a second real data point (153m) supporting it, not just the 50m
measurement and a theoretical 350m estimate.

### Time (and cost) to actually train each model, full budget, on this L4

The throughput numbers above answer "how fast" — this answers "how long would it
actually take to finish," using each model's own full configured token budget (not
an arbitrary fixed step count), at $0.70/hr on-demand:

| Model | Full budget | Speed | Wall-clock (full budget on L4, from scratch) | Cost |
|---|---|---|---|---|
| 50m | 1,000,000 steps (4.10B tokens) | 13.3 steps/s (measured) | **20.9 hours** | **$14.62** |
| 153m | 150,000 steps (2.46B tokens) | 1.45 steps/s (measured) | **28.7 hours** | **$20.11** |
| 350m (E=1024,L=24) | ~1,386,000 steps (5.68B tokens, this project's 16 tok/param convention) | ~3.0-4.2 steps/s (estimated) | **91.7-128.3 hours (3.8-5.3 days)** | **$64-$90 (estimated)** |

**What actually happened this session, vs. these full-budget numbers** — none of the
three runs above were carried to completion, and that was deliberate each time, not
a failure to finish:

- **50m**: the checkpoint entering this session was already at step 709,513/1,000,000
  (70.9%) from a mix of local MPS training (the bulk of it — 34:56:22 of the run's
  35:01:00 cumulative time) plus a few hours of this session's GCP resume. Stopped at
  709,513 and **never finished** — teardown happened with the run 29% short of its
  own budget, a deliberate stop (checkpoint downloaded, decided not worth the
  remaining ~3.9 hours/$2.70 to finish this session — see the GCP porting phase's
  teardown notes above).
- **153m**: stopped at step 440/150,000 (0.3%) — this run's whole purpose was
  measuring steady-state speed, never intended to run toward completion. The 28.7h
  figure above is a *projection* from the measured 1.45 steps/sec, not a completed run.
- **350m**: never run at all — architecture only exists as a `gpt-config` computation
  and a FLOPs-based estimate (see "Next GPU to test" section). The 91.7-128.3h figure
  is doubly speculative: estimated speed AND an assumed budget, neither measured.

**Reading the table correctly**: it answers "if you let each model's full run finish
uninterrupted on this L4, how long/much would it cost" — a planning number for a
*future* full run, not a record of what this session actually completed. This
session's actual GPU time across both real runs (50m resume + 153m measurement) was
well under 6 hours combined, far short of either model's full budget — consistent
with the session's actual goal (port infrastructure, fix bugs, measure speed, document)
rather than finish training either model.

### Teardown ([DONE] 2026-08-17)

Stopped training cleanly at **step 440/150,000** (`tmux send-keys -t train C-c`,
confirmed `Saved: checkpoints/153m/latest.pt`, confirmed no `gpt-train` process
remained). This checkpoint (step 440, ~5 minutes of training) was **not** downloaded
back to local — same reasoning as the 50m session's final checkpoint: this run's
purpose was measuring speed, not producing a usable partially-trained model, and
step 440/150,000 (0.3%) has no standalone value. It no longer exists once the
instance was destroyed.

Full teardown, same procedure as the 50m session: emptied the `153m/` bucket prefix,
then `terraform destroy -auto-approve`. Verified via `gcloud compute instances list`
(0 items) and `terraform state list` (empty) — nothing left billing.

## Session 3: custom-gpt-153m real resume, full command trail (2026-08-20)

Third use of this module. Unlike Session 2 (speed measurement only, torn down at
step 440), this is a genuine resume of an in-progress pretraining run — local
Mac/MPS training had reached **step 16,847/127,933** (`GPT_STEPS=127933`, a
recalibrated budget, not Session 2's 150,000) before being stopped cleanly
(`Ctrl-C` in the local `make train` session, confirmed `Saved:
checkpoints/153m/latest.pt`). Every command below was actually run, in order,
against the state that existed at the start of this session (module fully torn
down — no bucket, no instance, `terraform state list` empty).

### 1. Recreate free infra (bucket/SA/firewall, no instance)

```bash
cd infra/gcp-gpu-node
terraform init -input=false
terraform apply -auto-approve -var instance_count=0
```
Recreated the bucket (`mini-llm-gpu-llm-training-dev-us-central1`), service account,
and both firewall rules — 5 resources, $0 billing (no instance).

### 2. Upload corpus + checkpoint

```bash
make upload-corpus            # PROJECT_DIR defaults to ../../<project_subdir> = custom-gpt-153m
make upload-checkpoint
```
**Known gap surfaced here**: `upload-corpus`'s `gcloud storage rsync ... --include-file-pattern="*.bin,*.bin.json"`
has no filename filter beyond the extension, so it also swept up
`data/train.full-7b.bin` (14GB, an unrelated larger corpus sitting in the same
`data/` directory, not used by this run) alongside the real `train.bin`/`test.bin`
(~6GB). Uploaded **18.78GiB instead of the intended ~6GB** — harmless to training
correctness (the trainer only reads `train.bin`/`test.bin`, the extra file just sits
unused on the box's disk) but adds real upload/boot-time-pull minutes. **Not fixed
this session** (would have meant re-running an already-in-flight transfer) — worth
tightening the glob (e.g. exclude `*-7b*`, or keep that file outside `data/`) before
the next upload if `train.full-7b.bin` still exists alongside the active corpus then.

Checkpoint upload: 5.12GiB (`best.pt`/`latest.pt`/`serving.pt`, step 16,847).

### 3. Find GPU capacity — zone rotation

Quota was never the blocker this session (`NVIDIA_L4_GPUS` limit=1 usage=0 and
`GPUS_ALL_REGIONS` limit=1 usage=0 in every region checked) — pure regional
`STOCKOUT`, continuing the pattern from Sessions 1-2. Before this session's own
attempts, `terraform.tfvars`'s own comments already recorded `us-central1-a/-b/-c`,
`us-west1-a/-b`, and `us-east4-a` as STOCKOUT and `us-east4-b` as not offering
`g2-standard-4` at all (all tried earlier the same day, 2026-08-20, in a prior
attempt this session picked up from). `zone` is overridable per-`apply` on the CLI
(`-var zone=...`) without touching `terraform.tfvars`, since `main.tf` reads
`var.zone` directly — used to rotate zones without an edit-plan-apply cycle per zone:

```bash
for z in us-west1-c us-east4-c us-east1-b us-east1-c us-east1-d us-west4-a us-west4-c; do
  terraform apply -auto-approve -var instance_count=1 -var "zone=$z" && break
  # on STOCKOUT: terraform apply -auto-approve -var instance_count=0 -var "zone=$z"  (clean up, try next)
done
```
Results this session: `us-west1-c` STOCKOUT, `us-east4-c` STOCKOUT, `us-east1-b`
STOCKOUT, **`us-east1-c` succeeded** (instance `RUNNING` within ~30s). `us-east1-d`/
`us-west4-a`/`us-west4-c` were never needed. `region` stayed `us-central1` for the
bucket per the existing cross-region-is-cheap reasoning; `zone=us-east1-c` for the
instance.

### 4. Verify boot, GPU, data

```bash
# bootstrap-log — bounded snapshot pattern (no -f), same rationale as Session 1:
ssh -i ~/.ssh/id_ed25519 gpu@<ip> 'tail -n 300 /var/log/gpu-node-bootstrap.log'
# ... repeated until "=== bootstrap done ===" appeared (bootstrap took ~3.5 minutes,
# apt update + uv sync + corpus/checkpoint pull, corpus pull was the slowest step
# because of the extra 14GB file from step 2 above)

make gpu   # NVIDIA L4, driver 580.173.02, torch 2.11.0+cu130, bf16 True
ssh -i ~/.ssh/id_ed25519 gpu@<ip> 'ls -la ~/tiny_llm/from_scratch/custom-gpt-153m/data/'
ssh -i ~/.ssh/id_ed25519 gpu@<ip> 'sudo systemctl status checkpoint-sync.service --no-pager'
```
**New gap found**: `make gpu`'s own recipe (`uv run python3 -c ...`) failed with
`uv: command not found` when run via a *separate* one-off `ssh gpu@<ip> '<cmd>'`
invocation outside the Makefile — `uv` lives at `~/.local/bin/uv`, added to `PATH`
only for interactive login shells (`.bashrc`), not the non-interactive shell a bare
`ssh host 'cmd'` gets. The Makefile's own `make gpu` target works because `make`
itself runs the whole multi-command string through one `ssh ... 'nvidia-smi; cd ...
&& uv run ...'` invocation — still non-interactive, so this likely only worked
historically if `.bashrc` sourcing was in fact happening, or wasn't hit before
because Session 1/2 always used `make gpu` verbatim, never split into separate `ssh`
calls the way this session's verification steps did. **Workaround used**: invoke
`~/.local/bin/uv` by full path in any one-off `ssh host 'cmd'` call that isn't going
through the Makefile's own recipe.

Checked the checkpoint on the box matched what was uploaded before trusting it:
```bash
ssh -i ~/.ssh/id_ed25519 gpu@<ip> 'cd ~/tiny_llm/from_scratch/custom-gpt-153m && \
  ~/.local/bin/uv run python3 -c "
import torch
c = torch.load(\"checkpoints/153m/latest.pt\", map_location=\"cpu\")
print(c.get(\"step\"), c.get(\"batch_size\"), c.get(\"grad_accum_steps\"))
"'
# -> 16847 4 16, matching local exactly
```

### 5. Launch — precision decision, and a real interrupt-handling bug

Checkpoint metadata: `batch_size=4, grad_accum_steps=16`. This codebase's token
budget is `steps × batch_size × ctx_len`, **independent of `grad_accum`** — so
launching with a *different* `batch_size` at the same `GPT_STEPS` would silently
change the total tokens the budget represents, not just reshuffle compute. Matched
exactly rather than "optimizing" toward Session 1/2's known-good L4 batch=16/accum=4
(that pairing was never re-derived for the recalibrated 127,933-step budget, and
changing it now would have desynced the LR schedule from what the local steps
already assumed):

```bash
tmux new-session -d -s train
tmux send-keys -t train "export PATH=\$HOME/.local/bin:\$PATH" C-m
tmux send-keys -t train "cd ~/tiny_llm/from_scratch/custom-gpt-153m" C-m
tmux send-keys -t train "GPT_BATCH_SIZE=4 GPT_GRAD_ACCUM=16 GPT_STEPS=127933 uv run gpt-train 2>&1 | tee -a /tmp/gpt-train.log" C-m
```
First banner: `Precision: torch.float16` — surprising on an L4 (bf16-capable).
Traced to `src/gpt/config.py:138`: `precision: str = "fp16"` hardcoded as the
project's default (with `"auto"`, which picks bf16 on CUDA, commented out just
above it) — presumably set for the Mac/MPS runs, where this choice predates any GPU
session. `trainer.py`'s own comment states bf16 needs no `GradScaler` "because it
has fp32's exponent range" — implying fp16 without one genuinely can
overflow/underflow, a real risk left running unattended for ~75 hours. No
`GradScaler` exists anywhere in this codebase. **Decision (user's call): switch to
bf16** via the `GPT_PRECISION` env var override (`config.py:273` registers it),
rather than editing `config.py`'s default — model weights aren't stored in half
precision (autocast only affects the forward/backward compute), so switching
mid-run doesn't corrupt the resume.

Stopping the fp16 launch to relaunch is where the real bug surfaced:
```bash
tmux send-keys -t train C-c
```
No `Interrupted — saving a resumable checkpoint...` line ever appeared (contrast
with the local `make train-stop` session, which printed it every time), the
`gpt-train` process was already gone within ~8s, and `checkpoints/153m/latest.pt`'s
step was unchanged at 16847 — the ~664 microsteps made since resume (~11 minutes of
L4 time, ~$0.13) were lost. **Root cause, isolated**: the ` | tee -a
/tmp/gpt-train.log` pipe added to the launch command (for this session's own log
monitoring, not present in Session 1/2's plain `uv run gpt-train`) sits between the
terminal and the Python process; `tmux send-keys C-c`'s `SIGINT` did not reach
`gpt-train`'s `except KeyboardInterrupt` handler (`trainer.py:482`) through it.
**Fix**: use plain output redirection instead of a pipe — no second process in the
foreground group to interfere with signal delivery:
```bash
tmux kill-session -t train
tmux new-session -d -s train
tmux send-keys -t train "export PATH=\$HOME/.local/bin:\$PATH" C-m
tmux send-keys -t train "cd ~/tiny_llm/from_scratch/custom-gpt-153m" C-m
tmux send-keys -t train "GPT_BATCH_SIZE=4 GPT_GRAD_ACCUM=16 GPT_STEPS=127933 GPT_PRECISION=bf16 uv run gpt-train > /tmp/gpt-train.log 2>&1" C-m
```
Confirmed clean: `Precision: torch.bfloat16`, `Resumed at step 16848` (matching
16847+1, the expected off-by-one), 98% GPU util, 7.5/23GB VRAM. **Lesson for next
time**: this project's own `config.py` documents a signal-delivery-quirk-proof
fallback specifically for cases like this —
`paths.stop_file` = `checkpoint_root/STOP_TRAINING` (`config.py:222-236`), polled
every step. Prefer `touch checkpoints/STOP_TRAINING` on the box over `tmux
send-keys ... C-c` for any future stop in a session that redirects/pipes
`gpt-train`'s output, rather than assuming `Ctrl-C` will reach the process.

### Speed & GPU utilization, measured directly (2026-08-20, ~30min after this launch)

The training banner's own `eta_h` is misleading here: it's computed from
*cumulative* `processed_tokens / elapsed_hours` since step 0, which blends in all
the slow local Mac/MPS history (`total_h=11.50` at step ~19,635, i.e. the vast
majority of that 11.5h was MPS, not this L4) — it read `eta_h=63.4` at step 19,635,
implying ~63 more hours. That's the *historical blended* rate (`19635 / 11.50h /
3600 ≈ 0.47 steps/s`), not this GPU's actual current rate. Measured the real
GCP-only rate directly instead — two step-count snapshots a fixed wall-clock
interval apart:

```bash
ssh -i ~/.ssh/id_ed25519 gpu@<ip> "tail -c 400 /tmp/gpt-train.log | grep -oE '[0-9]+/127933' | tail -1"; date -u +%s
# ... wait ~60s ...
ssh -i ~/.ssh/id_ed25519 gpu@<ip> "tail -c 400 /tmp/gpt-train.log | grep -oE '[0-9]+/127933' | tail -1"; date -u +%s
```
Result: step 19,777 → 20,098 over 67s = **4.79 steps/sec** (batch=4, ctx=1024, so
`4.79 × 4,096 ≈ 19,624 tok/s`). At that rate, remaining budget (107,835 steps from
step 20,098) is **~6.25 hours, ~$4.38 more** at $0.70/hr — not the ~63-75h the
banner's own blended `eta_h` suggests. **Use a direct two-snapshot measurement like
this, not the banner's `eta_h`, whenever a run mixes GPU types across its history**
(same caution applies to any future resume that started on Mac/MPS or a different
GPU generation).

GPU utilization, same interval:
```bash
ssh -i ~/.ssh/id_ed25519 gpu@<ip> 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu --format=csv,noheader'
# 96-100 %, 7532 MiB / 23034 MiB (~33%), 72.0-72.2 W draw against a 72.0 W limit, 76-79°C
```
Running essentially flat-out on compute (96-100%) while sitting at exactly the L4's
rated 72W TDP — this is the L4's actual power envelope (a low-power card by design,
not an artificial GCP cap), and VRAM headroom is ample (~15.5GB free) at this
batch=4 config. Lower tok/s than Session 2's batch=16 measurement (23,760 tok/s) is
consistent with that session's own finding that this model is memory-bandwidth-bound
with MFU flat-to-worse at smaller batches, not a regression — batch=4 here was kept
deliberately (see "Launch" above) to match the local checkpoint's token-budget math,
not chosen for speed.

### Precision switch diagnosis: did fp16→bf16 hurt quality? (2026-08-20)

Raised by the user after seeing gibberish QA-report output on the GCP box and
worrying the mid-run fp16→bf16 switch (see "Launch" above) had caused it.
Investigated with the actual data rather than by inspection alone — checked
`logs/train_eval_history_153m.csv` on both sides and diffed real QA report HTML
files (not just described them):

**Mechanically, the switch cannot corrupt a resume.** This codebase never stores
half-precision weights — `trainer.py`'s own comment: "Weights/grads stay fp32
regardless — autocast only changes the dtype of the ops inside the block." The
checkpoint's fp32 master weights and fp32 Adam state are identical either way;
`precision` only selects the `torch.autocast` dtype wrapping forward/backward. The
LR schedule confirms this in practice: it decayed smoothly across the switch (LR
3.88e-04 at step 17,000 in both the fp16 and bf16 launches), since it's a pure
function of step count, untouched by precision.

**The local run was already diverging, independent of GCP entirely.** Local
`train_eval_history_153m.csv`: `best_test_loss` was set once, at step 1,500
(`8.1031`, 2026-08-20T03:42 UTC), and never improved again — every eval after that,
all the way to step 16,500 (`test_loss=52.5`, 2026-08-20T15:50 UTC, still local
Mac/MPS, still fp16), climbed further. This predates any GCP involvement (GCP wasn't
touched until ~16:00 UTC that day).

**The bf16 switch on GCP immediately reversed the divergence, on the identical
checkpoint.** Both launches resumed from the same step-16,847 weights. The first
(fp16) continuation kept climbing (`train_loss` 54.4→56.6 from step 17,000→17,500,
matching the local trend exactly). The second (bf16) launch, from that *same*
starting checkpoint, dropped to `train_loss=7.90` by step 17,500 and kept improving
to a new all-time-best `test_loss=7.6830` by step 21,000 — better than the local
run ever achieved (8.1031). **Conclusion: fp16 without a `GradScaler` (this
codebase has none — see "Launch" above) was almost certainly the actual cause of
the original divergence**, most likely silent gradient overflow/underflow as LR
ramped toward its ~4e-4 peak around step 2,000-5,000; bf16's wider exponent range
doesn't have that failure mode. The fp16→bf16 direction taken here is the strictly
safer one — the reverse (bf16→fp16) would be the risky switch.

**The "gibberish" itself is real but unrelated to precision.** Pulled and read the
actual HTML, not just metadata, on both sides: local `qa_report_153m_step1500.html`
(local's numerical best, pre-divergence) and `qa_report_153m_step14143.html`
(latest local) are both punctuation/function-word soup, no real sentences. GCP's
`qa_report_153m_step19999.html` (this run's actual best checkpoint, `test_loss=
7.7093`) is the same kind of soup. **All three look equally incoherent** — expected
at 0.03-0.18 epochs over a 2.9B-token corpus for a 152M-param model, nowhere near
enough data seen to form grammar yet, regardless of GPU or precision. No local
report was ever found showing coherent sentences to compare against.

**Process note**: generating a QA report (`uv run gpt-qa-report`) *while training is
still running* on the same box shares the GPU with the live `tmux` training session
— it completed, but took ~7 CPU-minutes (vs. presumably faster if run alone) due to
contention, long enough to exceed a 2-minute one-shot `ssh` timeout. Not a hang; just
slower than expected. Poll for the report file's existence + the process exiting
rather than assuming a timed-out one-shot check means it failed.

### Where this run stands at end of session

Instance `mini-llm-gpu`, zone `us-east1-c`, `RUNNING`, ~$0.70/hr, **no automated
cost guard** (same open issue as Sessions 1-2 — budget resource and self-stop IAM
both still broken, manual monitoring is mandatory). Training resumed at step
16,848/127,933 (13.2%), bf16, ETA ~75 hours if run continuously. **Not torn down at
end of this session** — left running deliberately, unlike Sessions 1-2's
intentional-stop-and-destroy pattern, since this is a real in-progress pretraining
run, not a measurement exercise. Next session (or later this one): `make ssh` +
`tmux attach -t train` to check progress, `touch checkpoints/STOP_TRAINING` on the
box for a safe stop, `make download-checkpoints
PROJECT_DIR=../../from_scratch/custom-gpt-153m` before any destructive step, `make
down` to stop billing.

### Teardown ([DONE — full destroy] 2026-08-20)

Explicitly requested — pausing for the day, resuming a later session, not leaving the
box running unattended overnight (no automated cost guard exists, see Known issues).

Stopped training cleanly using the `STOP_TRAINING` file fallback (not `Ctrl-C` — see
"Launch" above for why that's the safer choice when output has been redirected/piped):
```bash
ssh -i ~/.ssh/id_ed25519 gpu@<ip> 'touch ~/tiny_llm/from_scratch/custom-gpt-153m/checkpoints/STOP_TRAINING'
```
Confirmed via the log: `checkpoints/STOP_TRAINING found — stopping gracefully...` /
`Saved: checkpoints/153m/latest.pt` / no `gpt-train` process left. **Final step:
31,183/127,933 (24.4%)**, cumulative training time 12:12:16 (mostly local MPS +
~1.4 hours on the L4, bf16). Best `test_loss` this run: **7.166** at step 31,000 —
well past the local-only best of 8.103 (step 1,500), consistent with the earlier
precision diagnosis above (fp16 divergence, bf16 recovery).

**Downloading checkpoints before any destructive step — plain `scp` failed silently,
worth knowing for next time**: a direct `scp gpu@<ip>:.../latest.pt ./` of the three
~1.8GB checkpoint files was interrupted by the tool's own 2-minute command timeout
mid-transfer, and the resulting local file **looked complete (byte count matched) but
was actually truncated and failed to `torch.load`** — `scp` doesn't reliably resume or
signal partial failure the way `gcloud storage rsync` does. **Recovered by going
through the bucket instead of raw `scp`**, using the already-proven path:
```bash
# On the box — push the box's own latest checkpoint state to GCS (same command the
# periodic checkpoint-sync systemd service runs, invoked manually for an immediate,
# guaranteed-fresh sync rather than waiting up to checkpoint_sync_minutes):
ssh -i ~/.ssh/id_ed25519 gpu@<ip> 'sudo -u gpu /usr/local/bin/checkpoint-sync.sh --once'
# -> "[ckpt-sync] ok"

# Locally — the Makefile's own download path (gcloud storage rsync, not scp):
make download-checkpoints PROJECT_DIR=../../from_scratch/custom-gpt-153m
```
This ran past the 2-minute tool timeout too (home internet, not GCP, was the
bottleneck this time — took ~15+ minutes for 5.1GB) but completed correctly and every
file verified loadable (`torch.load` + step-number check) afterward. **Lesson: for
any multi-GB file leaving a GCP box, prefer `gcloud storage rsync` (via the bucket)
over raw `scp`** — it handles partial/interrupted transfers correctly where `scp`
silently produced a corrupted-but-plausible-looking file once already this session
(caught only because this SOP's own habit is to verify with `torch.load` immediately
after any download, not to trust file size alone).

The small `logs/train_eval_history_153m.csv` (not covered by checkpoint-sync, which
only syncs the `checkpoints/` prefix) was pulled directly with a plain `scp` instead
— fine for small files, saved locally as `logs/train_eval_history_153m_gcp.csv` to
avoid clobbering the separate local-only history file already in that project.

Full teardown, same procedure as prior sessions:
```bash
gcloud storage rm --recursive gs://mini-llm-gpu-llm-training-dev-us-central1/153m
terraform destroy -auto-approve
```
6 resources destroyed (instance, bucket, its IAM binding, service account, 2 firewall
rules) in ~15s, no errors. Verified: `gcloud compute instances list` (0 items),
`gcloud storage buckets list` (empty), `terraform state list` (empty) — nothing left
billing. `terraform.tfvars`'s `instance_count` reset to `0` and `zone` left at
`us-east1-c` (this session's working zone — try it first next time before rotating
through the fallback list again) to keep the file matching real state, per this
module's own safety convention (see `make plan` before any future `apply`).

**Next session, to resume**: `make init` (if needed) → `terraform apply -auto-approve
-var instance_count=0` (recreates bucket/SA/firewall, $0) → `make upload-corpus
PROJECT_DIR=../../from_scratch/custom-gpt-153m` (note: also re-uploads the unrelated
14GB `train.full-7b.bin` unless the glob gap in `README.md`'s Known issues is fixed
first — worth fixing before repeating this) → `make upload-checkpoint
PROJECT_DIR=../../from_scratch/custom-gpt-153m` (uploads the step-31,183 checkpoint
just downloaded) → rotate zones starting with `us-east1-c` → resume training with
`GPT_BATCH_SIZE=4 GPT_GRAD_ACCUM=16 GPT_STEPS=127933 GPT_PRECISION=bf16 uv run
gpt-train > /tmp/gpt-train.log 2>&1` (plain redirection, not `| tee` — see "Launch"
above for why) in a detached `tmux` session, using the `STOP_TRAINING` file (not
`Ctrl-C`) for the next stop.

## Session 4: resume following the exact Session-3 next-session plan (2026-08-22)

Local training continued unattended after Session 3's teardown — by the time this
session started, the local checkpoint was already at **step 32,048** (test_loss
7.166 unchanged, cumulative 14:11:20), ahead of the step-31,183 checkpoint Session 3
downloaded. This session uploaded and resumed from the newer local checkpoint, not
the older GCS-stored one — worth noting since the "next session" plan text above was
written against 31,183 and the actual resume point moved on its own between
sessions.

### 1. Recreate free infra, `instance_count=0` first

```bash
terraform init -input=false
terraform plan -input=false -var instance_count=0   # reviewed before applying
terraform apply -auto-approve -input=false -var instance_count=0
```
5 resources created (bucket, service account, IAM binding, 2 firewall rules), $0.
Confirmed via `gcloud compute instances list` / `gcloud storage buckets list`
(instance: 0 items; bucket: present) before touching billing at all.

### 2. Upload corpus + checkpoint — explicit filenames, not the Makefile target

**Deliberately bypassed `make upload-corpus`.** `data/` in this project now also
holds an unrelated `train.full-7b.bin` (13GB, not referenced anywhere in `src/` —
confirmed with `grep -rn "full-7b" src/`) sitting next to the real `train.bin`
(5.5GB). `make upload-corpus`'s fallback path (`gcloud storage cp data/*.bin
data/*.bin.json ...`, triggered whenever the primary `rsync
--include-file-pattern=...` call fails) is a plain shell glob with no filter — it
would silently pull that unrelated 13GB file up too, the same class of bug as the
`--only-show-errors=false` issue documented above. Confirmed the *Terraform output
hint* has the identical blind spot (`--exclude='.*\.txt$'` excludes text files, not
the extra `.bin`), so this isn't only a Makefile issue.

**Fix applied this session — explicit file list, no glob at all:**
```bash
gcloud storage cp \
  data/train.bin data/train.bin.json data/test.bin data/test.bin.json \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/
```
`make upload-checkpoint` was used as-is for the checkpoint side — its `rsync
--exclude=".*\.DS_Store$"` has no equivalent glob risk (`checkpoints/` only ever
holds the three legitimate `.pt` files).

Verified before spending anything on compute: `gcloud storage ls -l` against the
bucket showed all 7 objects with byte counts matching the local files exactly
(10.71 GiB total: 5.86GB `train.bin`, 143MB `test.bin`, ~1.8GB × 3 checkpoint
files) — no truncation, and no `train.full-7b.bin` anywhere in the bucket.

### 3. Launch — no zone rotation needed this time

`terraform apply -auto-approve -input=false -var instance_count=1` succeeded on the
first try, same zone as last session (`us-east1-c`) — no `STOCKOUT`, instance up in
25s. Public IP `34.148.31.49`.

### 4. Transient bootstrap corpus-sync failure, worked around

The boot log's tail showed `Terminated` immediately followed by `WARN: corpus sync
failed` — the checkpoint pull succeeded (3 files, 256 MiB/s) but the corpus pull
did not, leaving `data/` empty on the box. **Not the historical
`--only-show-errors=false` bug** (that fix is already in the template, and this
error surfaced differently) — root cause not fully isolated, most likely a
transient race between the tail end of `uv sync`'s dependency install and the
corpus rsync starting immediately after on a freshly-booted box, but this wasn't
confirmed. **Practical fix, not a root-cause fix**: re-ran the exact same rsync by
hand over SSH —
```bash
ssh -i ~/.ssh/id_ed25519 gpu@34.148.31.49 \
  'gcloud storage rsync "gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/" ~/tiny_llm/from_scratch/custom-gpt-153m/data/ --recursive'
```
Succeeded cleanly the second time (253 MiB/s), file sizes verified matching exactly
afterward. **Lesson for next time**: don't trust the bootstrap log's corpus-sync
line at face value — verify `ls -la ~/tiny_llm/.../data/` after boot even when the
log doesn't show an obvious fatal error, since this failure mode produces no
Python-level error until `gpt-train` itself tries to read a missing file.

Also hit, same session: `uv: command not found` over a plain non-interactive SSH
command, even though `uv sync` had clearly already run during bootstrap — `uv`
installs to `~/.local/bin`, which isn't on `PATH` for a non-login SSH shell.
Fix: `export PATH="$HOME/.local/bin:$PATH"` before any `uv run` invoked this way
(already the case for whatever sources `.bashrc` on an interactive login, e.g.
`make ssh` itself — only bit non-interactive one-off SSH commands).

### 5. Verified GPU, then launched

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv
# -> NVIDIA L4, 23034 MiB, 0 MiB used, 0% util (idle, as expected pre-launch)
uv run python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())'
# -> True True
```
Launched in a detached `tmux` session (`tmux new-session -d -s train ...`), same
command as Session 3's documented plan:
```bash
GPT_BATCH_SIZE=4 GPT_GRAD_ACCUM=16 GPT_STEPS=127933 GPT_PRECISION=bf16 \
  uv run gpt-train > /tmp/gpt-train.log 2>&1
```
Confirmed clean: `Precision: torch.bfloat16`, `Resumed at step 32048 (cumulative
14:11:20)`, `Progress: step 32,048/127,933 (25.1%)`, ETA ~42.5 more training-hours
(~1.8 days) if run continuously, no `Traceback`/OOM in the first several seconds of
output.

### Where this run stands at end of session

Instance `mini-llm-gpu`, zone `us-east1-c`, `RUNNING`, ~$0.70/hr, **no automated
cost guard** (same open, unresolved issue as every prior session — manual
monitoring is still mandatory). Training resumed at step 32,048/127,933 (25.1%),
bf16. **Left running deliberately** — same in-progress-run reasoning as Session 3.
To check on it next time: `make ssh` (or the SSH command above) + `tmux attach -t
train`. To stop safely: `touch
~/tiny_llm/from_scratch/custom-gpt-153m/checkpoints/STOP_TRAINING` on the box (not
`Ctrl-C`), then `make download-checkpoints PROJECT_DIR=../../from_scratch/custom-gpt-153m`
before any destructive step, then `make down`.

### Addendum: installing DCGM (NVIDIA Data Center GPU Manager) — verified working on L4

Not present on this image by default — `systemctl status nvidia-dcgm` /
`systemctl status dcgm` both report "could not be found" out of the box, and
`apt-cache search dcgm` returns nothing but unrelated `libnvidia-nscq-*` (NVSwitch)
libraries until NVIDIA's own CUDA repo is added. The base image ships the driver
(confirmed `580.173.02`), `nvidia-persistenced` (running), and a
`nvidia-fabricmanager.service` unit definition (present but `failed` — expected and
harmless, that daemon manages NVLink/NVSwitch fabric, which a single L4 doesn't have).
DCGM itself is a separate, opt-in NVIDIA product, mainly useful for feeding the DCGM
Prometheus exporter at fleet scale — not something a single-box run needs, but tested
here on request:

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -qq

# DCGM 4.x, matched to this image's CUDA 12 environment (not the older
# monolithic `datacenter-gpu-manager` 3.3.9 package, also available in the same repo)
sudo apt-get install -y datacenter-gpu-manager-4-core datacenter-gpu-manager-4-cuda12

sudo systemctl enable --now nvidia-dcgm
```

Confirmed working end-to-end, installed and verified *while training was actively
running* (no disruption — pure apt install, no driver touch, no reboot):
`dcgmi discovery -l` correctly identified the L4 (name, PCI bus ID, device UUID);
`dcgmi dmon -e 203,204,252 -c 5` (GPU util / mem-controller util / framebuffer used)
returned live values matching `nvidia-smi`'s own numbers from the same moment
(~97-100% GPU util, ~7.5GB VRAM) — cross-validated against a second, independent
tool, not just "the service started." One incidental convenience: enabling
`nvidia-dcgm` auto-creates a `dcgm.service` symlink alias, so both service names
resolve afterward.

Full command reference with exact real outputs — discovery, health checks, `dmon`
field IDs, and a full live-triage session for the power warning this install
surfaced — is kept separately at
[`dcgm_gpu_command_reference.md`](dcgm_gpu_command_reference.md), not duplicated here.
See also [`nvidia_smi_command_reference.md`](nvidia_smi_command_reference.md) for the
driver-bundled tool DCGM complements, and
[`checkpoint_download_command_reference.md`](checkpoint_download_command_reference.md)
for the exact sync-then-download-then-verify sequence used every time a checkpoint is
pulled down while training keeps running.

## Phase 6 — Day-to-day management

| Situation | Command | Effect |
|---|---|---|
| Pausing overnight, resuming same box | `make stop` | Compute billing stops; disk (~$8/mo/100GB) keeps billing until `start` or `down` |
| Resuming a stopped box | `make start` | New public IP — `terraform refresh && make status` after |
| Done, want billing at ~$0 | `make down` | Instance + disk destroyed; bucket/SA/firewall survive, free |
| **Checking spend right now** | Cloud Console → Billing → Reports (no budget alert is armed — see Known issues) | **Mandatory manual check**, not a fallback, until the budget resource is fixed |
| Home IP rotated, SSH refused | `make apply` (re-detects `/32`) or `make iap-ssh` | Observed flapping between two different IPs (redacted: `0.0.0.0`/`0.0.0.0`) during this session's `plan`s — re-run `apply` if `make ssh` ever refuses |
| A newer Deep Learning image shouldn't silently replace a mid-run box | *(automatic)* | Ignored by `lifecycle` until `make replace` |

**Always sync checkpoints down before any destructive step**:
```bash
make download-checkpoints PROJECT_DIR=../../from_scratch/custom-gpt-50m
```

## Phase 7 — Teardown ([DONE — full destroy] 2026-08-17)

```bash
make download-checkpoints PROJECT_DIR=../../from_scratch/custom-gpt-50m
make down                  # normal end state: bucket/SA/firewall survive, ~$0
```

`make destroy` additionally removes the service account/firewall rules, and will fail
on a non-empty bucket unless `force_destroy_bucket = true` — same caveat as the AWS
module.

**What actually happened**: full `destroy`, not the normal `make down` pause state —
explicitly requested, since this session was ending the GCP run entirely rather than
pausing it. Training was stopped cleanly first (`tmux send-keys -t train C-c`,
confirmed "Saved: checkpoints/50m/latest.pt", confirmed no `gpt-train` process left
with `ps -ef | grep gpt-train`) — **final step: 709,513/1,000,000 (71.0%)**, cumulative
training time 35:01:00 (mostly local MPS + ~5.5 hours on the L4).

**Checkpoint was deliberately NOT downloaded back to local** — explicit call: the GCP
session's extra progress (695,057 -> 709,513, ~14,500 steps over a few hours) wasn't
judged a significant enough improvement over the local checkpoint already in
`from_scratch/custom-gpt-50m/checkpoints/50m/` to justify the download step. **The
GCP box's checkpoint (step 709,513) is now gone** — it existed only in the bucket
(deleted below) and on the destroyed instance's boot disk. The authoritative
checkpoint going forward is the local one from before this session, at whatever step
it was at when `make train-stop` ran locally (see Phase 2). If step 709,513 turns out
to matter later, it's not recoverable — this was a considered trade, not an accident.

Bucket had to be emptied by hand before `destroy` — confirmed `force_destroy_bucket`
was left at its default `false`, and `terraform destroy` does not empty a bucket
itself:
```bash
gcloud storage rm --recursive gs://mini-llm-gpu-llm-training-dev-us-central1/50m
terraform destroy -auto-approve
```
`destroy` output: **6 resources destroyed** (bucket, its IAM binding, service account,
2 firewall rules, the compute instance) in ~1m20s, no errors.

**Verified fully torn down**:
```
$ gcloud compute instances list --project=llm-training-dev
Listed 0 items.
$ gcloud storage buckets list --project=llm-training-dev --format="value(name)"
(empty)
$ gcloud iam service-accounts list --project=llm-training-dev --format="value(email)"
llm-training-dev-sa@llm-training-dev.iam.gserviceaccount.com     # pre-existing, not this module's
260913878468-compute@developer.gserviceaccount.com               # GCP's default per-project SA, not this module's
$ terraform state list
(empty)
```
Nothing left billing. `terraform.tfvars` (with all the fixes/overrides from this
session — `repo_url`, `image_family`, `zone=us-east4-a`, `monthly_budget_usd=0`,
`instance_count=1`) is untouched on disk for the next session — set
`instance_count` back to `0` or pass `-var instance_count=0` before the next `make
plan` if picking this up again without immediately relaunching.

## Phase 8 — Using the trained model locally, after `download-checkpoints` — **PENDING**

Identical to the AWS SOP's Phase 8 — a checkpoint trained on GCP's CUDA loads on the
Mac's MPS/CPU with zero code changes (`get_device()` auto-detects). No conversion
step. See [`../../aws-gpu-node/docs/SOP.md`](../../aws-gpu-node/docs/SOP.md#phase-8--using-the-trained-model-locally-after-download-checkpoints)
for the full command table — not duplicated here since it's cloud-agnostic.

---

## Full command index

```
make init                    Phase 0 — terraform init
make plan                    Any time — see what would change, no side effects
make down                    Phase 1 & 7 — bucket/SA/firewall only, or full teardown
make up                      Phase 3 & 6 — launch the instance
make upload-corpus           Phase 2 — data/*.bin(.json) -> bucket
make upload-checkpoint       Phase 2 — checkpoints/ -> bucket, for auto-resume
make download-checkpoints    Phase 6 & 7 — bucket -> local checkpoints/
make bootstrap-log           Phase 3 — tail cloud-init boot log
make gpu                     Phase 3 — nvidia-smi + bf16 sanity check
make ssh                     Phase 4+ — SSH in (needs ~/.ssh/id_ed25519)
make iap-ssh                 Phase 4+ — Session-Manager-style, no open port
make tunnel                  Optional — forward the model API to localhost:8000
make sync-log                Phase 5 — tail checkpoint-sync / preempt-watch journal
make status                  Phase 5 & 6 — instance state/type/IP/zone
make stop / make start       Phase 6 — pause/resume, disk keeps billing while stopped
make replace                 Phase 6 — rebuild the instance on the current image
make destroy                 Phase 7 — remove everything Terraform manages
```
