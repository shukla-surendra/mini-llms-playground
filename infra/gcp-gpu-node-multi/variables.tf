########################################
# Identity / placement
########################################

variable "project_id" {
  description = "GCP project id (not the project *number*, and not this module's 'project' name label below). No sensible default — unlike AWS, Terraform cannot discover this from your credentials alone. Find it: `gcloud config get-value project`."
  type        = string

  validation {
    condition     = length(var.project_id) > 0
    error_message = "project_id is required — GCP has no equivalent of AWS's auto-discovered account id."
  }
}

variable "project" {
  description = "Name prefix for every resource, and the label the instance's self-stop IAM binding is scoped to. Same role as the AWS module's identically-named variable."
  type        = string
  default     = "mini-llm-gpu"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,28}$", var.project))
    error_message = "project must be lowercase alphanumeric/hyphen, start with a letter, 2-29 chars (it is used in a GCS bucket name and a service account id, both of which have their own stricter length limits)."
  }
}

variable "region" {
  description = "Region for the bucket and for regional lookups. Keep the bucket and the instance in the SAME region — in-region GCS<->Compute Engine transfer is free, cross-region is billed and slow (identical reasoning to the AWS module)."
  type        = string
  default     = "us-central1"
}

variable "network" {
  description = "VPC network self-link/name to launch into. Empty = use the project's 'default' auto-mode network (the GCP analog of the AWS module's default-VPC lookup)."
  type        = string
  default     = ""
}

variable "zone" {
  description = "Zone to launch the VM into. Unlike the AWS module, this is NOT auto-resolved from GPU capacity — GCP's Terraform provider has no direct equivalent of `aws_ec2_instance_type_offerings`. us-central1-a is a reliable, broadly-stocked L4 zone as of this module's writing; if capacity errors out at apply time (a real, fairly common GCP GPU experience), try us-central1-b/c, us-east4-a, or us-west1-a next — `gcloud compute accelerator-types list --filter=\"name=nvidia-l4\"` shows current zone availability."
  type        = string
  default     = "us-central1-a"
}

########################################
# Compute
########################################

variable "machine_type" {
  description = "GCP's G2 family bundles L4 GPUs directly into the machine type — no separate guest_accelerator block needed, GPU count is implied by the shape name alone. g2-standard-24 = 24 vCPU, 96GB RAM, 2x L4 24GB each, bf16 + TF32 — verified via `gcloud compute machine-types describe g2-standard-24` (guestAcceleratorCount=2), not assumed: this module's single-GPU sibling's own outputs.tf mislabels g2-standard-12 as '2x L4', which is wrong — g2-standard-12 is still 1x L4 with more vCPU/RAM allocated to it, don't copy that error forward. This is this module's whole reason to exist as a separate module rather than a tfvars override of the single-GPU one: multi-process (torchrun) launch orchestration differs, not just the machine shape."
  type        = string
  default     = "g2-standard-24"
}

variable "image_project" {
  description = "Project that owns the boot-image family looked up below. 'ml-images' publishes GCP's Deep Learning VM Images (NVIDIA driver + CUDA preinstalled, minimal framework bloat) — the closest GCP analog of the AWS module's 'Base OSS Nvidia Driver' AMI choice."
  type        = string
  default     = "ml-images"
}

variable "image_family" {
  description = "Boot image family, resolved to the newest image in it at plan time (same dynamic-lookup pattern the AWS module uses for its AMI, via data \"google_compute_image\" below) — a wrong family name fails loudly at `terraform plan`, not silently. GCP's exact published family names shift over time; verify this one still exists before a first apply with `gcloud compute images list --project ml-images --no-standard-images | grep cu1`, and override here if it has moved on."
  type        = string
  default     = "common-cu124-debian-12"
}

variable "image_id" {
  description = "Pin a specific image self-link/id instead of resolving image_family dynamically. Empty = use the family lookup."
  type        = string
  default     = ""
}

variable "root_volume_gb" {
  description = "Boot persistent-disk size in GB. Same reasoning as the AWS module: corpus .bin (~5GB) + ~1.8GB per checkpoint + OS + venv needs real headroom, not the platform default."
  type        = number
  default     = 100

  validation {
    condition     = var.root_volume_gb >= 50
    error_message = "root_volume_gb below 50 will run out mid-run; the AWS module's baseline (and this one's) is 100."
  }
}

variable "root_volume_type" {
  description = "Persistent disk type. pd-balanced is the GCP analog of AWS gp3 — SSD-backed, no separate provisioned-IOPS purchase needed at this scale."
  type        = string
  default     = "pd-balanced"
}

variable "use_spot" {
  description = "Request a Spot VM: the single biggest cost lever here (60-91% off depending on machine type/region), safe only because gpt-train resumes from latest.pt and checkpoint_sync_minutes keeps latest.pt in GCS. Real behavioral difference from AWS Spot, not just a naming difference: GCP gives ONLY ~30 seconds' preemption notice (via the instance/preempted metadata key), not AWS's ~2 minutes — see templates/bootstrap.sh.tftpl's preempt-watch service, which polls far more aggressively to compensate."
  type        = bool
  default     = false
}

variable "instance_count" {
  description = "0 = keep all the supporting infra (bucket, service account, firewall) but no billing instance. Same between-runs pattern as the AWS module."
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
  description = "Local PUBLIC key installed into the VM's OS Login-less SSH metadata. Same reasoning as the AWS module: generating the pair locally beats letting the platform mint one — the private half never leaves this machine."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "ssh_user" {
  description = "SSH username the public key above is installed for (GCP's metadata-based SSH keys are `username:key` pairs, unlike AWS's key-pair-per-instance model)."
  type        = string
  default     = "gpu"
}

variable "ssh_private_key_path" {
  description = "Cosmetic only: used to build the ssh_command output. Not read by Terraform."
  type        = string
  default     = "~/.ssh/id_ed25519"
}

variable "allowed_ssh_cidrs" {
  description = "CIDRs allowed on port 22. Empty = auto-detect this machine's public IP as a /32, same http-provider trick the AWS module uses."
  type        = list(string)
  default     = []
}

variable "allow_open_ssh" {
  description = "Escape hatch to permit 0.0.0.0/0 on port 22. Left false on purpose — same reasoning as the AWS module: an open SSH port on a GPU box with a service account that can write to your bucket is a crypto-miner magnet."
  type        = bool
  default     = false
}

variable "api_port" {
  description = "Port the repo's FastAPI serving scripts bind to (base_models/*, from_scratch/*/api_server.py). Same as the AWS module."
  type        = number
  default     = 8000
}

variable "allowed_api_cidrs" {
  description = "CIDRs allowed on api_port. Empty = closed; reach the API over an SSH tunnel instead, which needs no extra firewall rule."
  type        = list(string)
  default     = []
}

variable "enable_iap_ssh" {
  description = "Allow SSH via Identity-Aware Proxy tunneling (`gcloud compute ssh --tunnel-through-iap`) from Google's IAP range (35.235.240.0/20) regardless of allowed_ssh_cidrs — the GCP analog of the AWS module's SSM Session Manager: a second way in that survives an IP change and needs no key on the box's metadata to work, gated by IAM (roles/iap.tunnelResourceAccessor) rather than a network rule."
  type        = bool
  default     = true
}

variable "iap_ssh_members" {
  description = "GCP IAM principals (e.g. [\"user:you@example.com\"]) granted roles/iap.tunnelResourceAccessor on this specific instance. Empty by default — see iam.tf's comment on why: whoever runs `terraform apply` almost always already has enough project-level access to use IAP without an extra grant here. Only needed for a teammate or a different principal than the one applying."
  type        = list(string)
  default     = []
}

########################################
# Storage / bootstrap
########################################

variable "create_bucket" {
  description = "Create the corpus/checkpoint bucket. Set false to reuse an existing one via bucket_name. Same as the AWS module."
  type        = bool
  default     = true
}

variable "bucket_name" {
  description = "Bucket name. Empty = <project>-<project_id>-<region> — GCS bucket names are globally unique across ALL of GCP (not just your project), so project_id substitutes for the AWS module's account-id uniqueness trick."
  type        = string
  default     = ""
}

variable "force_destroy_bucket" {
  description = "Allow `terraform destroy` to delete a non-empty bucket. False means your checkpoints survive a fat-fingered destroy. Same as the AWS module."
  type        = bool
  default     = false
}

variable "repo_url" {
  description = "Git repo cloned onto the instance at boot. Same as the AWS module."
  type        = string
  default     = "https://github.com/shukla-surendra/mini-llms-playground.git"
}

variable "repo_dir_name" {
  description = "Directory name the clone lands in under the ssh_user's home directory."
  type        = string
  default     = "tiny_llm"
}

variable "project_subdir" {
  description = "Path inside the repo that gets `uv sync`'d and treated as the working dir. Defaults to custom-gpt-50m — different default from the AWS module (which defaults to custom-gpt-153m), matching this repo's actual current GCP target."
  type        = string
  default     = "from_scratch/custom-gpt-50m"
}

variable "corpus_prefix" {
  description = "GCS prefix synced into <project_subdir>/data at boot. Empty = skip. Same 'ship .bin, never .txt' reasoning as the AWS module."
  type        = string
  default     = "50m/corpus/"
}

variable "checkpoint_prefix" {
  description = "GCS prefix synced into <project_subdir>/checkpoints at boot, for resuming an existing run. Empty = fresh run. Same reasoning as the AWS module: point it at a prefix mirroring this project's checkpoints/<label>/ layout so gpt-train's auto-resume finds latest.pt with no flag."
  type        = string
  default     = "50m/checkpoints/"
}

########################################
# Cost guards
########################################

variable "idle_shutdown_minutes" {
  description = "Stop the box after this many consecutive minutes of ~0% GPU utilization. 0 disables it. Same guard as the AWS module, same failure mode it protects against: a finished/crashed run idling overnight."
  type        = number
  default     = 15
}

variable "idle_gpu_threshold_pct" {
  description = "GPU utilization at or below this counts as idle for the watchdog."
  type        = number
  default     = 5
}

variable "checkpoint_sync_minutes" {
  description = "Sync checkpoints/ to GCS every N minutes, plus immediately on a Spot preemption notice. 0 disables. Same reasoning as the AWS module — what makes Spot safe."
  type        = number
  default     = 10
}

variable "monthly_budget_usd" {
  description = "GCP Billing Budget monthly amount in USD. 0 disables. Requires billing_account_id. Unlike AWS Budgets, GCP billing budgets don't take a single ad-hoc email address directly — see budget.tf for exactly who gets notified and why budget_alert_email below is used differently than in the AWS module."
  type        = number
  default     = 60 # ~2x the single-GPU module's 30 default — this box's hourly rate
  # (~$1.96/hr estimated for g2-standard-24's 2x L4, vs ~$0.70/hr for the single-GPU
  # module's g2-standard-4) is roughly double, so the same number of GPU-hours costs
  # roughly double; sized to match, not picked independently.
}

variable "billing_account_id" {
  description = "GCP Billing Account id (format XXXXXX-XXXXXX-XXXXXX), required only if monthly_budget_usd > 0. Find it: `gcloud billing accounts list`. No AWS equivalent variable exists because AWS Budgets attaches to the account implicitly; GCP billing budgets attach to a Billing Account explicitly, which can span multiple projects."
  type        = string
  default     = ""
}

variable "budget_alert_email" {
  description = "Best-effort only — see budget.tf's comment. GCP billing budget threshold rules notify every user with a billing-account IAM role (Billing Account Administrator/User) by default; this address is used to create a Cloud Monitoring notification channel ADDED on top of that default audience, not a replacement for it the way the AWS module's identically-named variable is a complete audience list."
  type        = string
  default     = ""
}
