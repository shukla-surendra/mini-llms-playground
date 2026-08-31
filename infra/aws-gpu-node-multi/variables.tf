########################################
# Identity / placement
########################################

variable "project" {
  description = "Name prefix for every resource, and the tag the instances' self-stop IAM policy is scoped to."
  type        = string
  default     = "mini-llm-gpu-ddp"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,30}$", var.project))
    error_message = "project must be lowercase alphanumeric/hyphen, 2-31 chars (it is used in an S3 bucket name)."
  }
}

variable "region" {
  description = "Region for everything. Keep the bucket and both instances in the SAME region — in-region S3<->EC2 transfer is free, cross-region is billed and slow."
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC to launch into. Empty = use the account's default VPC."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Subnet BOTH instances launch into. Empty = pick a default-VPC subnet in an AZ that actually offers instance_type. Both nodes always land in this one subnet (one AZ) deliberately — see network.tf: same-AZ, private-IP traffic between EC2 instances is free and lower-latency than cross-AZ, which matters directly for DDP gradient sync, not just cost."
  type        = string
  default     = ""
}

########################################
# Compute
########################################

variable "instance_type" {
  description = "GPU instance type, one per node. g6.xlarge = 1x L4 24GB, bf16 + TF32, ~$0.80/hr on-demand. Do not silently fall back to g4dn (T4/Turing) — no bf16, see docs/AWS_RUNBOOK.md."
  type        = string
  default     = "g6.xlarge"
}

variable "instance_count" {
  description = "0 = keep all the supporting infra (bucket, role, SG, key) but no billing instances. 2 = both DDP nodes. This module manages an all-or-nothing pair — DDP with one live rank just hangs waiting for the other's collective, so there is no useful '1' here."
  type        = number
  default     = 2

  validation {
    condition     = contains([0, 2], var.instance_count)
    error_message = "This module manages a matched pair of nodes for multi-node DDP: 0 or 2, never 1."
  }
}

variable "ami_id" {
  description = "Pin a specific AMI id. Empty = resolve the newest Deep Learning Base OSS Nvidia Driver GPU AMI via ami_name_filter."
  type        = string
  default     = ""
}

variable "ami_name_filter" {
  description = "Name filter used when ami_id is empty. The Base OSS AMI ships the NVIDIA driver + CUDA but no frameworks — this repo installs its own torch via uv, so the bigger PyTorch DLAMI is wasted disk and boot time."
  type        = string
  default     = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"
}

variable "ami_owners" {
  description = "Owner filter for the AMI lookup. 'amazon' is the alias for the DLAMI publishing account (898082745236)."
  type        = list(string)
  default     = ["amazon"]
}

variable "root_volume_gb" {
  description = "Root gp3 size, per node. This project's corpus .bin (~1.9GB train + ~20MB test) + checkpoints (~4.2GB each, fp32 weights + AdamW moments) + OS + venv need real headroom."
  type        = number
  default     = 100

  validation {
    condition     = var.root_volume_gb >= 50
    error_message = "root_volume_gb below 50 will run out mid-run; the baseline is 100."
  }
}

variable "use_spot" {
  description = "Request Spot for both instances. Real caveat specific to this module, not just a cost note: a reclaim of EITHER node kills the whole synchronized DDP job (the survivor hangs on a gradient all-reduce the dead peer never completes), not just that one instance's own work. checkpoint_sync_minutes still caps the loss at one interval on RESTART, but 'safely resumable' here means 'both nodes restart from the last synced checkpoint,' not 'the run quietly continues.' Consider on-demand for a real paid multi-hour run and Spot only for the cheap CPU-mechanism-equivalent smoke testing this module doesn't do (that's scripts/ddp_smoke_test.py, which needs no cloud spend at all)."
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Max hourly USD for Spot, applied per instance. Empty = pay up to the on-demand price."
  type        = string
  default     = ""
}

variable "shutdown_behavior" {
  description = "What an in-guest `shutdown -h now` does on each node. 'stop' keeps the EBS volume so a restart resumes from the synced checkpoint. Forced to 'terminate' for Spot by AWS."
  type        = string
  default     = "stop"

  validation {
    condition     = contains(["stop", "terminate"], var.shutdown_behavior)
    error_message = "shutdown_behavior must be 'stop' or 'terminate'."
  }
}

########################################
# DDP networking
########################################

variable "master_ip_host_offset" {
  description = "Host number (within the resolved subnet's CIDR) statically assigned to the master (rank 0) node's private IP — see network.tf's cidrhost() use. A static, plan-time-known IP (rather than AWS's auto-assigned one) is what lets the worker's bootstrap script be given --master_addr without a resource depending on its own not-yet-created sibling's computed attribute, which Terraform would reject as a cycle. 10 is arbitrary but safely past the small range some VPC setups reserve at the bottom of a subnet (AWS itself reserves the first 4 + the last 1 address of every subnet automatically, regardless of this offset)."
  type        = number
  default     = 10
}

variable "dist_port" {
  description = "Port `torchrun`'s rendezvous uses (--master_port) between the two nodes. NCCL negotiates additional ephemeral ports for the actual gradient traffic after rendezvous — see network.tf's security-group rule, which opens a full range between the two nodes rather than trying to enumerate NCCL's dynamic port choices exactly (the standard real-world pattern for small NCCL clusters)."
  type        = number
  default     = 29500
}

########################################
# Access
########################################

variable "public_key_path" {
  description = "Local PUBLIC key uploaded as the EC2 key pair (shared by both instances). The private half never leaves this machine."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "existing_key_pair_name" {
  description = "Use an EC2 key pair that already exists instead of uploading public_key_path."
  type        = string
  default     = ""
}

variable "ssh_private_key_path" {
  description = "Cosmetic only: used to build the ssh_command outputs. Not read by Terraform."
  type        = string
  default     = "~/.ssh/id_ed25519"
}

variable "allowed_ssh_cidrs" {
  description = "CIDRs allowed on port 22, on both nodes. Empty = auto-detect this machine's public IP as a /32. Home IPs are usually dynamic — re-apply after your ISP rotates it."
  type        = list(string)
  default     = []
}

variable "allow_open_ssh" {
  description = "Escape hatch to permit 0.0.0.0/0 on port 22. Left false on purpose: an open SSH port on a GPU box is a crypto-miner magnet, and each box holds an IAM role with write access to your bucket."
  type        = bool
  default     = false
}

variable "api_port" {
  description = "Port the repo's FastAPI serving scripts bind to, opened identically on both nodes."
  type        = number
  default     = 8000
}

variable "allowed_api_cidrs" {
  description = "CIDRs allowed on api_port. Empty = closed; reach it over an SSH tunnel instead."
  type        = list(string)
  default     = []
}

variable "enable_ssm" {
  description = "Attach AmazonSSMManagedInstanceCore on both instances so `aws ssm start-session` works — a second way in that survives losing the key or an IP change, no inbound port needed."
  type        = bool
  default     = true
}

########################################
# Storage / bootstrap
########################################

variable "create_bucket" {
  description = "Create the corpus/checkpoint/tokenizer bucket, shared by both nodes. Set false to reuse an existing one via bucket_name."
  type        = bool
  default     = true
}

variable "bucket_name" {
  description = "Bucket name. Empty = <project>-<account-id>-<region>, globally unique without a random suffix that would churn on every apply."
  type        = string
  default     = ""
}

variable "force_destroy_bucket" {
  description = "Allow `terraform destroy` to delete a non-empty bucket. False means your checkpoints survive a fat-fingered destroy."
  type        = bool
  default     = false
}

variable "repo_url" {
  description = "Git repo cloned onto both instances at boot."
  type        = string
  default     = "https://github.com/shukla-surendra/mini-llms-playground.git"
}

variable "repo_dir_name" {
  description = "Directory name the clone lands in under /home/ubuntu, on both nodes."
  type        = string
  default     = "mini-llms-playground"
}

variable "project_subdir" {
  description = "Path inside the repo that gets `uv sync`'d and treated as the working dir on both nodes. Defaults to the DDP-hardened fork, not the un-forked custom-gpt-350m — that project's DDP code has never been run for real; this one's has (scripts/ddp_smoke_test.py)."
  type        = string
  default     = "from_scratch/custom-gpt-350m-ddp"
}

variable "corpus_prefix" {
  description = "S3 prefix synced into <project_subdir>/data on BOTH nodes at boot. Empty = skip. Ship .bin, never .txt — re-tokenizing on two hourly-billed GPUs is money for nothing, twice."
  type        = string
  default     = "350m-ddp/corpus/"
}

variable "tokenizer_prefix" {
  description = "S3 prefix synced into <project_subdir>/tokenizer on BOTH nodes at boot. This project's embedding table is sized to its own 32,768-token vocabulary — a .bin without the matching tokenizer.json trains on ids that index the wrong embedding rows, silently. Empty = skip (upload by hand before training)."
  type        = string
  default     = "350m-ddp/tokenizer/"
}

variable "checkpoint_prefix" {
  description = "S3 prefix synced into <project_subdir>/checkpoints on BOTH nodes at boot, for resuming an existing run. Empty = fresh run. Only rank 0 writes checkpoints during training (see trainer.py) — periodic sync (below) uploads from rank 0's node only; both nodes still pull the same prefix at boot so either one can resume as rank 0 if roles are reassigned later."
  type        = string
  default     = "350m-ddp/checkpoints/"
}

########################################
# Training budget — baked into the generated launch script, not a code edit
########################################

variable "target_tokens" {
  description = "GPT_TARGET_TOKENS for the generated launch script (see templates/bootstrap.sh.tftpl). This project's actual corpus (books + cosmopedia, tokenized into data/train.bin) is 1,015,850,483 train tokens — genuinely smaller than this model's Chinchilla-optimal 20N (~6.95B) at 347.36M params, and no larger corpus is currently planned, so the budget here is sized against the *real* data rather than the theoretical ideal. 4,000,000,000 is ~3.94 epochs over that corpus — inside the ~4-epoch mark this project's own DATASET.md documents as where repeated data stays close to as-good-as-fresh (repetition cost 'decays toward worthless' only by ~16 epochs, not at 4). Raise this only after actually growing the corpus past 1.02B tokens, not just to chase a bigger number against a corpus that hasn't grown."
  type        = number
  default     = 4000000000
}

variable "grad_accum_steps" {
  description = "GPT_GRAD_ACCUM for the generated launch script. Higher than this project's own single-GPU default (4) on purpose: without EFA, DDP's gradient all-reduce is a real, largely-unhidden cost over plain TCP between two g6.xlarge nodes, and it fires once per grad_accum_steps window, not once per micro-step. Raising this amortizes that fixed sync cost over more sync-free local compute — the concrete lever, not a knob to leave at the single-GPU value. Tune from a real measured run, not this default, before a long paid one."
  type        = number
  default     = 64
}

########################################
# Cost guards
########################################

variable "idle_shutdown_minutes" {
  description = "Stop both instances after this many consecutive minutes of ~0% GPU utilization, checked INDEPENDENTLY per node. Defaults to 0 (disabled) here, unlike the single-GPU sibling module's 30 — a genuine risk specific to multi-node DDP: one node's GPU can look idle for a stretch it's actually legitimately blocked on a gradient all-reduce waiting for the other rank, and an independent per-node watchdog stopping that node mid-collective hangs (or corrupts) the other node's run too. Enable only if you understand and accept that risk for your specific training cadence."
  type        = number
  default     = 0
}

variable "idle_gpu_threshold_pct" {
  description = "GPU utilization at or below this counts as idle for the watchdog, when enabled."
  type        = number
  default     = 5
}

variable "checkpoint_sync_minutes" {
  description = "Sync checkpoints/ to S3 every N minutes from rank 0's node (only rank 0 writes checkpoints — see trainer.py), plus immediately on a Spot interruption notice on either node. 0 disables."
  type        = number
  default     = 15
}

variable "monthly_budget_usd" {
  description = "AWS Budgets monthly cost budget in USD, covering both instances combined. 0 disables. Requires budget_alert_email."
  type        = number
  default     = 0
}

variable "budget_alert_email" {
  description = "Email for budget alerts. AWS Budgets emails directly, so no SNS topic + subscription confirmation dance."
  type        = string
  default     = ""
}
