output "project_id" {
  description = "GCP project id everything lives in (the Makefile reads this)."
  value       = var.project_id
}

output "zone" {
  description = "Zone the instance lives in (the Makefile reads this)."
  value       = var.zone
}

output "instance_name" {
  description = "Compute Engine instance name (empty when instance_count = 0). GCP names, rather than AWS-style opaque instance ids, are what every gcloud command below actually takes."
  value       = try(google_compute_instance.gpu[0].name, "")
}

output "public_ip" {
  description = "Ephemeral external IPv4. Changes on every stop/start — no static/reserved IP here, since an idle one is billed, same trade-off as the AWS module."
  value       = try(google_compute_instance.gpu[0].network_interface[0].access_config[0].nat_ip, "")
}

output "ssh_command" {
  description = "Copy-paste SSH, direct (needs allowed_ssh_cidrs to include your current IP)."
  value       = try("ssh -i ${var.ssh_private_key_path} ${var.ssh_user}@${google_compute_instance.gpu[0].network_interface[0].access_config[0].nat_ip}", "")
}

output "iap_ssh_command" {
  description = "SSH via Identity-Aware Proxy tunneling — the GCP analog of the AWS module's SSM alternative: no open port, works after your IP changes, gated by IAM instead of a firewall rule."
  value       = var.enable_iap_ssh ? try("gcloud compute ssh ${var.ssh_user}@${google_compute_instance.gpu[0].name} --zone=${var.zone} --project=${var.project_id} --tunnel-through-iap", "") : "disabled (enable_iap_ssh = false)"
}

output "bucket" {
  description = "Bucket the instance's service account can read/write. Upload token files here BEFORE launching."
  value       = local.bucket_name
}

output "upload_corpus_hint" {
  description = "What to run on the Mac from inside a project dir. Token files only — .txt and hf_cache stay home."
  value       = "gcloud storage rsync data/ gs://${local.bucket_name}/${var.corpus_prefix != "" ? var.corpus_prefix : "corpus/"} --exclude='.*\\.txt$' --recursive"
}

output "upload_checkpoint_hint" {
  description = "What to run on the Mac from inside a project dir, to make an existing run resumable on the instance. Empty unless checkpoint_prefix is set."
  value       = var.checkpoint_prefix != "" ? "gcloud storage rsync checkpoints/ gs://${local.bucket_name}/${var.checkpoint_prefix} --recursive" : ""
}

output "project_subdir" {
  description = "Which project the instance is set up for (the Makefile reads this for PROJECT_DIR)."
  value       = var.project_subdir
}

output "repo_dir_name" {
  description = "Directory the repo clones into on the instance under the ssh_user's home directory (the Makefile reads this to build the remote project path for `make gpu`)."
  value       = var.repo_dir_name
}

output "corpus_prefix" {
  description = "GCS prefix the instance pulls data/ from (the Makefile reads this for PREFIX)."
  value       = var.corpus_prefix
}

output "checkpoint_prefix" {
  description = "GCS prefix the instance pulls checkpoints/ from (the Makefile reads this for CKPT_PREFIX)."
  value       = var.checkpoint_prefix
}

output "image_id" {
  description = "Resolved boot image. Pin this into image_id if you want a byte-identical box next time."
  value       = local.image_id
}

output "allowed_ssh_cidrs" {
  description = "What port 22 is actually open to (direct path — IAP is a separate, always-on-if-enabled path regardless of this list)."
  value       = local.ssh_cidrs
}

output "stop_command" {
  description = "Ends compute billing, keeps the boot disk and everything on it."
  value       = try("gcloud compute instances stop ${google_compute_instance.gpu[0].name} --zone=${var.zone} --project=${var.project_id}", "")
}

output "estimated_hourly_usd" {
  description = "Rough on-demand rate for the chosen machine type, us-central1, for sanity only — not a billing source of truth. Spot pricing varies continuously; check `gcloud compute machine-types describe` / the Pricing Calculator before a real run."
  value = lookup({
    "g2-standard-4"  = 0.70 # 1x L4 24GB
    "g2-standard-8"  = 0.85
    "g2-standard-12" = 1.28 # 1x L4 (the single-GPU module's own table mislabels this as
    #                         "2x L4" — verified wrong via `gcloud compute machine-types
    #                         describe g2-standard-12`: guestAcceleratorCount=1. Left in
    #                         this table with the correction noted, not deleted, so the
    #                         mistake doesn't silently resurface if someone copies this
    #                         table again later.)
    "g2-standard-24" = 1.96 # 2x L4 24GB — DERIVED, not API-pulled: g2-standard-4's
    #                         real $0.70/hr total minus L4's real $0.56/hr GPU-only
    #                         rate (both already verified this session) gives a
    #                         ~$0.14/hr vCPU+RAM component at 4 vCPU/16GB; G2's
    #                         vCPU:RAM ratio is constant (1:4) across every shape, so
    #                         that component scales ~linearly with vCPU count — 6x
    #                         more vCPU (24 vs 4) -> ~$0.84/hr, plus 2x $0.56 GPU ->
    #                         ~$1.96/hr total. Verify via `gcloud compute
    #                         machine-types describe g2-standard-24` / the Pricing
    #                         Calculator before trusting this for a real budget.
    "a2-highgpu-1g" = 3.67 # 1x A100 40GB
  }, var.machine_type, -1)
}
