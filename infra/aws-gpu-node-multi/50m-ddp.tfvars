# Deploy this file against the "50m-ddp" Terraform workspace, not "default" —
# "default" holds custom-gpt-350m-ddp's state (its bucket/IAM/SG live there).
# Same module code, separate state, zero infra duplication:
#
#   terraform workspace select 50m-ddp   # or: terraform workspace new 50m-ddp
#   terraform plan  -var-file=50m-ddp.tfvars
#   terraform apply -var-file=50m-ddp.tfvars

region  = "us-east-1"
project = "mini-llm-gpu-50m-ddp"   # distinct from 350m-ddp's "mini-llm-gpu-ddp" — separate bucket/SG/key-pair names, same AWS account

# ---- compute ---------------------------------------------------------------
# Tried L4 (g6.xlarge) first on 2026-09-01 per cost preference — hit
# Server.InsufficientInstanceCapacity repeatedly (~11 min, 5+ failed RunInstances
# calls per CloudTrail, zero instances created) in this same subnet's AZ
# (us-east-1c), same failure custom-gpt-350m-ddp hit on 2026-08-31. Region-wide
# G6 capacity crunch appears to still be ongoing, not a one-off. Falling back to
# g5.xlarge (A10G 24GB), the type already confirmed working on this exact subnet.
instance_type  = "g5.xlarge" # A10G 24GB
root_volume_gb = 100

# Same AZ/subnet g5.xlarge was confirmed to actually get capacity on 2026-08-31 —
# a reasonable starting point, not a guarantee; if RunInstances stalls, check
# CloudTrail for Server.InsufficientInstanceCapacity before assuming a real hang
# (see custom-gpt-350m-ddp/infra README's "Real deployment log").
subnet_id = "subnet-0ddd28a9cc6a2f624" # us-east-1c, default VPC

use_spot = false

# ---- access ------------------------------------------------------------------
public_key_path      = "~/.ssh/id_ed25519.pub"
ssh_private_key_path = "~/.ssh/id_ed25519"
# allowed_ssh_cidrs = ["203.0.113.10/32"]   # leave empty to auto-detect this machine's public IP

# ---- storage / bootstrap ----------------------------------------------------
create_bucket = true

repo_url       = "https://github.com/shukla-surendra/tiny_llm.git"
project_subdir = "from_scratch/custom-gpt-50m-ddp"

corpus_prefix    = "50m-ddp/corpus/"
checkpoint_prefix = "50m-ddp/checkpoints/"
# No tokenizer_prefix: this model uses GPT-2's off-the-shelf tiktoken encoding
# (vocab_size=50257), which ships inside the pip package itself — unlike
# custom-gpt-350m-ddp's project-trained 32,768-vocab tokenizer.json, there is no
# separate tokenizer file to sync to either node.
tokenizer_prefix = ""

# ---- training budget ---------------------------------------------------------
# Phase 1 (pretrain) target — real number, from `gpt-tokenize` on 2026-09-01
# against the 30GB pretrain corpus truncated to a ~4.7GB prefix (data/train.txt,
# line-boundary safe cut; full corpus was far more than this 51.48M-param model's
# Chinchilla-optimal ~1.03B tokens needs — see docs/RESUME_POINT_2026-08-31.md).
# data/train.bin.json: 996,638,534 tokens exactly — essentially one full epoch at
# Chinchilla-optimal, no need to inflate this the way 350m-ddp's 4x-epoch budget
# did (that model needs ~20x more tokens per param at its larger size).
target_tokens = 996638534

# NOT YET VERIFIED ON REAL A10G HARDWARE — this model (51.48M params) is ~7x
# smaller than custom-gpt-350m-ddp's 347M, so these are a reasonable starting
# GUESS, not a measured-safe number the way 350m-ddp's batch_size=4 was. Per this
# project's own docs/MULTI_NODE_DDP.md and the 350m-ddp OOM saga it's grounded
# in: run a short single-GPU test (GPT_STEPS=30-60) on ONE node BEFORE deploying
# the second node or committing to a real multi-hour run, and adjust these two
# values together (same effective-batch-size math) if it OOMs or if a much
# larger batch turns out to fit comfortably.
batch_size       = 16
grad_accum_steps = 16

# ---- cost guards ---------------------------------------------------------------
checkpoint_sync_minutes = 15
idle_shutdown_minutes   = 0   # disabled deliberately for multi-node — see variables.tf

monthly_budget_usd = 60
# budget_alert_email = "you@example.com"
