# `gcp-gpu-node` — the training box, as code

The GCP equivalent of the sibling [`infra/aws-gpu-node`](../aws-gpu-node/) module —
built because AWS GPU quota was never approved for this account. Same purpose (a
single interactive GPU box for `from_scratch/custom-gpt-50m`, with the same cost
guards), different cloud, and — where GCP's actual mechanics genuinely differ from
AWS's, not just its naming — a different implementation, not a find-and-replace.

New to GCP's mental model coming from AWS (Organizations/Projects/IAM/service
accounts/`gcloud`)? Read [`docs/GCP_CONCEPTS.md`](docs/GCP_CONCEPTS.md) first —
every concept in it is grounded in this module's actual Terraform, not left
abstract.

## What it creates

- One Compute Engine VM (`g2-standard-4` = 1x NVIDIA L4 24GB, bf16-capable — the GCP
  analog of the AWS module's `g6.xlarge`)
- One GCS bucket for corpus + checkpoints (analog of the AWS module's S3 bucket)
- One service account, scoped to exactly this bucket plus the minimum permissions the
  idle/preemption watchdogs need to stop this one instance (analog of the AWS
  module's instance role — see `iam.tf` for why the scoping mechanism is genuinely
  different from AWS's tag-based IAM conditions, not just differently named)
- Firewall rules for SSH (direct + optional IAP tunnel) and, optionally, the model
  serving API
- An optional GCP Billing Budget with an 80%-actual / 100%-forecasted alert pair
  (analog of the AWS module's Budgets alarm)

## Prerequisites (one-time, on the Mac)

```bash
brew install --cask google-cloud-sdk   # gcloud CLI
gcloud init                             # authenticate, pick/create a project
gcloud auth application-default login   # credentials Terraform itself uses
gcloud services enable compute.googleapis.com storage.googleapis.com \
  iam.googleapis.com cloudbilling.googleapis.com billingbudgets.googleapis.com \
  monitoring.googleapis.com iap.googleapis.com

ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519   # if you don't already have one
```

New to why this generates *two* files, why the public half is safe to bake into every
future instance's `authorized_keys` at boot, and how this compares to an AWS EC2 `.pem`
file's very different-looking (but mechanically identical) key-issuance flow? See
[`../../../platform-lab/fundamentals/system_design_foundation/00_prerequisite_concepts/26_ssh_keys_and_public_key_cryptography.md`](../../../platform-lab/fundamentals/system_design_foundation/00_prerequisite_concepts/26_ssh_keys_and_public_key_cryptography.md).

## Quickstart

```bash
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: at minimum, set project_id (gcloud config get-value project)

make init
make plan     # read this before apply — confirm it's creating what you expect
make apply    # BILLING STARTS the moment the instance exists (skip with instance_count=0
              # to stand up just the bucket/service-account/firewall first)

make upload-corpus       # BEFORE the box needs it — see "order that matters" below
make ssh                 # or: make iap-ssh (no open port, no IP dependency)
make bootstrap-log       # watch the startup script finish
make gpu                  # nvidia-smi + confirm bf16 is actually available
```

## The order that matters: tokenize local, train remote

Identical reasoning to the AWS module: tokenizing a multi-GB corpus on a per-hour
billed GPU is money spent on CPU-bound work. Run `make data && make tokenize` in
`from_scratch/custom-gpt-50m` on your Mac first, THEN `make upload-corpus` here, THEN
`make apply`/`make up` — the boot script pulls `data/*.bin` + `*.bin.json` from the
bucket automatically once they're there.

## Decisions worth knowing about (where GCP genuinely isn't just AWS with different names)

- **Preemption notice is ~30 seconds, not ~2 minutes.** GCP Spot VMs get meaningfully
  less warning than AWS Spot instances. `templates/bootstrap.sh.tftpl`'s
  `preempt-watch.sh` compensates by long-polling the metadata server's
  `wait_for_change` mechanism (near-instant on flip) rather than AWS's simple
  poll-every-5-seconds loop — but the underlying window is still shorter, so
  `checkpoint_sync_minutes` matters even more here than in the AWS module.
- **No tag-based IAM conditions for the self-stop permission.** AWS's IAM lets a
  policy say "StopInstances, but only if tagged Project=X" — GCP's IAM Conditions
  don't reliably support that for Compute actions. This module instead binds a
  minimal custom role directly to the one instance resource it manages
  (`google_compute_instance_iam_member`) — see `iam.tf`'s comment for why that's
  arguably tighter, not looser, given this module only ever manages one instance.
- **`shutdown -h now` preserves the disk with zero extra config, on-demand.** AWS
  requires explicitly setting `instance_initiated_shutdown_behavior = "stop"` (its
  default permits either behavior). GCP's standard (non-Spot) instances already do
  this — a guest-initiated halt transitions to `TERMINATED` (GCP's "stopped"),
  keeping the boot disk, with nothing to configure.
- **Billing budgets attach to a Billing Account, not implicitly to "your account."**
  `billing_account_id` has no AWS-module equivalent variable — find it with
  `gcloud billing accounts list`. A custom alert email also needs its own Cloud
  Monitoring notification channel (`budget.tf` creates one); GCP notifies every
  Billing Account Admin/User by default regardless, which AWS Budgets doesn't do.
- **IAP tunneling is a grant on the *caller*, not the box.** Unlike AWS SSM (whose
  IAM lives entirely on the instance role), using `gcloud compute ssh
  --tunnel-through-iap` needs `roles/iap.tunnelResourceAccessor` on *your own* gcloud
  identity. If you can already run `terraform apply` here, you almost certainly
  already have it via broader project access — `iap_ssh_members` exists only for
  granting it to a different principal (a teammate) than whoever applied.
- **No AZ-capacity data-source lookup.** The AWS module resolves which Availability
  Zone actually offers `g6.xlarge` at plan time
  (`aws_ec2_instance_type_offerings`); this module has no equivalent for GCP zones,
  since Terraform's Google provider doesn't expose one as directly. `zone` is a plain
  variable — see its description for fallback zones to try if capacity errors out.

## Paying for the GPU and almost nothing else

Same pattern as the AWS module:

```bash
make down    # DESTROY the instance only. Bucket/service-account/firewall survive, cost ~$0.
make up      # recreate the instance for your next session
```

`make stop`/`make start` pause without destroying (still billed for the persistent
disk, ~$8/mo at the 100GB default — cheaper than re-syncing the corpus from scratch
between short sessions).

## Variables you will actually touch

See `variables.tf` for the full, commented list — every variable there explains its
own reasoning inline, same convention as the AWS module. The ones worth knowing exist
before your first `apply`: `project_id` (required), `zone`, `use_spot`,
`monthly_budget_usd` + `billing_account_id`, `project_subdir`/`corpus_prefix`/
`checkpoint_prefix` (already pre-filled in `terraform.tfvars.example` for
`custom-gpt-50m`).
