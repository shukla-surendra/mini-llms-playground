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
`gcloud billing accounts describe 013592-BC270C-27B8EF` 403s. **Disabled for now**
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
`013592-BC270C-27B8EF`, found via `gcloud billing accounts list`.

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

1. **A GPU with more memory bandwidth, not more compute** — an A10G (600GB/s, the
   AWS module's default) or A100 (much higher still) would plausibly help *because*
   this is bandwidth-bound, not because it has more FLOPs the model can't use
   anyway. Cheapest way to check before renting anything: rerun `gpt-benchmark
   --sweep-batch` on whichever GPU is being considered — if MFU is still pinned
   regardless of batch there too, the bandwidth theory is confirmed; if MFU climbs
   with batch size, this model actually can use more compute after all and the L4
   conclusion doesn't transfer.
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
