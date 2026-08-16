########################################
# Identity / placement
########################################

variable "project" {
  description = "Name prefix for every resource, and the tag the instance's self-stop IAM policy is scoped to."
  type        = string
  default     = "mini-llm-gpu"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,30}$", var.project))
    error_message = "project must be lowercase alphanumeric/hyphen, 2-31 chars (it is used in an S3 bucket name)."
  }
}

variable "region" {
  description = "Region for everything. Keep the bucket and the instance in the SAME region — in-region S3<->EC2 transfer is free, cross-region is billed and slow."
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC to launch into. Empty = use the account's default VPC."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Subnet to launch into. Empty = pick a default-VPC subnet in an AZ that actually offers instance_type (not every AZ has G6 capacity)."
  type        = string
  default     = ""
}

########################################
# Compute
########################################

variable "instance_type" {
  description = "GPU instance type. g6.xlarge = 1x L4 24GB, bf16 + TF32, ~$0.80/hr on-demand. Do not silently fall back to g4dn (T4/Turing) — no bf16, see docs/AWS_RUNBOOK.md."
  type        = string
  default     = "g6.xlarge"
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
  description = "Root gp3 size. The 8 GB default is nowhere near enough: corpus .bin (~5 GB) + ~1.8 GB per checkpoint + OS + venv."
  type        = number
  default     = 100

  validation {
    condition     = var.root_volume_gb >= 50
    error_message = "root_volume_gb below 50 will run out mid-run; the AWS_RUNBOOK baseline is 100."
  }
}

variable "use_spot" {
  description = "Request a Spot instance: the single biggest cost lever here (~60-70% off), and safe only because gpt-train resumes from latest.pt and checkpoint_sync_minutes keeps latest.pt in S3. Spot cannot 'stop' — a reclaim (or an in-guest shutdown) terminates and deletes the root volume, so shutdown_behavior is forced to 'terminate' automatically."
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Max hourly USD for Spot. Empty = pay up to the on-demand price (the sane default; capping too low just means the request is never fulfilled)."
  type        = string
  default     = ""
}

variable "shutdown_behavior" {
  description = "What an in-guest `shutdown -h now` does. 'stop' is what makes the runbook's dead-man switch safe: the run ends, billing for compute ends, and the EBS volume (checkpoints, corpus, venv) survives so you can restart and resume. Forced to 'terminate' for Spot by AWS."
  type        = string
  default     = "stop"

  validation {
    condition     = contains(["stop", "terminate"], var.shutdown_behavior)
    error_message = "shutdown_behavior must be 'stop' or 'terminate'."
  }
}

variable "instance_count" {
  description = "0 = keep all the supporting infra (bucket, role, SG, key) but no billing instance. Useful between runs when you want the bucket to outlive the box."
  type        = number
  default     = 1

  validation {
    condition     = var.instance_count >= 0 && var.instance_count <= 1
    error_message = "This module manages a single interactive box: 0 or 1."
  }
}

########################################
# Access
########################################

variable "public_key_path" {
  description = "Local PUBLIC key uploaded as the EC2 key pair. Generating the pair locally beats letting AWS mint a .pem — the private half never leaves this machine and is never in AWS's hands."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "existing_key_pair_name" {
  description = "Use an EC2 key pair that already exists instead of uploading public_key_path."
  type        = string
  default     = ""
}

variable "ssh_private_key_path" {
  description = "Cosmetic only: used to build the ssh_command output. Not read by Terraform."
  type        = string
  default     = "~/.ssh/id_ed25519"
}

variable "allowed_ssh_cidrs" {
  description = "CIDRs allowed on port 22. Empty = auto-detect this machine's public IP as a /32 (the console's 'My IP'). Home IPs are usually dynamic — re-apply after your ISP rotates it."
  type        = list(string)
  default     = []
}

variable "allow_open_ssh" {
  description = "Escape hatch to permit 0.0.0.0/0 on port 22. Left false on purpose: an open SSH port on a GPU box is a crypto-miner magnet, and the box holds an IAM role with write access to your bucket."
  type        = bool
  default     = false
}

variable "api_port" {
  description = "Port the repo's FastAPI serving scripts bind to (base_models/*, from_scratch/*/api_server.py)."
  type        = number
  default     = 8000
}

variable "allowed_api_cidrs" {
  description = "CIDRs allowed on api_port. Empty = closed; reach the API over an SSH tunnel instead (`ssh -L 8000:localhost:8000 ...`), which needs no extra opening."
  type        = list(string)
  default     = []
}

variable "enable_ssm" {
  description = "Attach AmazonSSMManagedInstanceCore so `aws ssm start-session` works. A second way in that survives losing the key or an IP change, and it needs no inbound port at all."
  type        = bool
  default     = true
}

########################################
# Storage / bootstrap
########################################

variable "create_bucket" {
  description = "Create the corpus/checkpoint bucket. Set false to reuse an existing one via bucket_name."
  type        = bool
  default     = true
}

variable "bucket_name" {
  description = "Bucket name. Empty = <project>-<account-id>-<region>, which is globally unique without a random suffix that churns on every apply."
  type        = string
  default     = ""
}

variable "force_destroy_bucket" {
  description = "Allow `terraform destroy` to delete a non-empty bucket. False means your checkpoints survive a fat-fingered destroy."
  type        = bool
  default     = false
}

variable "repo_url" {
  description = "Git repo cloned onto the instance at boot."
  type        = string
  default     = "https://github.com/shukla-surendra/mini-llms-playground.git"
}

variable "repo_dir_name" {
  description = "Directory name the clone lands in under /home/ubuntu."
  type        = string
  default     = "tiny_llm"
}

variable "project_subdir" {
  description = "Path inside the repo that gets `uv sync`'d and treated as the working dir, e.g. from_scratch/custom-gpt-153m."
  type        = string
  default     = "from_scratch/custom-gpt-153m"
}

variable "corpus_prefix" {
  description = "S3 prefix synced into <project_subdir>/data at boot. Empty = skip (upload later by hand). Ship .bin + .bin.json, never .txt — re-tokenizing on an hourly GPU is money for nothing."
  type        = string
  default     = ""
}

variable "checkpoint_prefix" {
  description = "S3 prefix synced into <project_subdir>/checkpoints at boot — for resuming an existing run rather than starting fresh. Empty = skip (fresh run, no checkpoint to resume from). Point it at a prefix that mirrors this project's own checkpoints/<label>/ layout (e.g. '50m/checkpoints/' holding a checkpoints/50m/*.pt tree upload via `make upload-checkpoint`), so gpt-train's auto-resume finds latest.pt exactly where it looks — no code or command-line flag needed."
  type        = string
  default     = ""
}

########################################
# Cost guards
########################################

variable "idle_shutdown_minutes" {
  description = "Stop the box after this many consecutive minutes of ~0% GPU utilization. 0 disables it. This is the guard for the failure the runbook keeps warning about: a finished (or crashed) run idling overnight at $19/day."
  type        = number
  default     = 30
}

variable "idle_gpu_threshold_pct" {
  description = "GPU utilization at or below this counts as idle for the watchdog."
  type        = number
  default     = 5
}

variable "checkpoint_sync_minutes" {
  description = "Sync checkpoints/ to S3 every N minutes, plus immediately on a Spot interruption notice. 0 disables. This is what makes cheap Spot capacity safe: it caps what a reclaim can destroy at one interval instead of the whole run."
  type        = number
  default     = 15
}

variable "monthly_budget_usd" {
  description = "AWS Budgets monthly cost budget in USD. 0 disables. Requires budget_alert_email."
  type        = number
  default     = 0
}

variable "budget_alert_email" {
  description = "Email for budget alerts. AWS Budgets emails directly, so no SNS topic + subscription confirmation dance."
  type        = string
  default     = ""
}
