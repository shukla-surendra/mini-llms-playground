########################################
# Boot image
#
# Same dynamic-lookup pattern as the AWS module's `data "aws_ami" "dl_base"`: resolve
# the newest image in image_family at plan time rather than hardcoding an id, so a
# wrong/moved family name fails loudly at `terraform plan`, not at boot.
########################################

data "google_compute_image" "dl_base" {
  count   = var.image_id == "" ? 1 : 0
  family  = var.image_family
  project = var.image_project
}

locals {
  image_id = var.image_id != "" ? var.image_id : try(data.google_compute_image.dl_base[0].self_link, "")

  bootstrap_script = templatefile("${path.module}/templates/bootstrap.sh.tftpl", {
    ssh_user                = var.ssh_user
    bucket                  = local.bucket_name
    corpus_prefix           = var.corpus_prefix
    checkpoint_prefix       = var.checkpoint_prefix
    repo_url                = var.repo_url
    repo_dir_name           = var.repo_dir_name
    project_subdir          = var.project_subdir
    idle_shutdown_minutes   = var.idle_shutdown_minutes
    idle_gpu_threshold_pct  = var.idle_gpu_threshold_pct
    checkpoint_sync_minutes = var.checkpoint_sync_minutes
    use_spot                = var.use_spot
    zone                    = var.zone
  })
}

########################################
# The box
#
# g2-standard-24 bundles its 2x L4 GPUs directly into the machine type — no separate
# guest_accelerator block, unlike an N1 + attached-GPU shape. That also means there is
# no equivalent of the AWS module's per-GPU-family AZ-offering lookup to write: G2
# availability is a property of the zone you pick (var.zone), not something this
# module resolves dynamically — see that variable's description for the fallback
# zones to try if capacity errors out at apply time. Also: the GPUS_ALL_REGIONS quota
# is a global count across every GPU on the project, not per-instance — confirm it
# covers 2 (not just the 1 the single-GPU module needed) before the first real apply.
########################################

resource "google_compute_instance" "gpu" {
  count = var.instance_count

  name         = var.project
  machine_type = var.machine_type
  zone         = var.zone
  tags         = [var.project]

  boot_disk {
    auto_delete = true # deleted when this instance resource is destroyed — same as the AWS module's delete_on_termination

    initialize_params {
      image = local.image_id
      size  = var.root_volume_gb
      type  = var.root_volume_type
    }
  }

  network_interface {
    network = local.network

    access_config {
      # Empty block = ephemeral external IP, no reservation. Changes on every
      # stop/start — same trade-off the AWS module makes explicitly (no Elastic IP,
      # since an idle one is billed).
    }
  }

  # GPU-equipped machine types cannot live-migrate — required for any GPU shape,
  # Spot or not. On Spot, GCP deletes (not stops) a preempted instance by default;
  # this module does not attempt the newer STOP-on-preemption reservation feature,
  # matching the AWS module's own choice to force "terminate" behavior for Spot
  # rather than rely on a stop-and-resume path for reclaimed capacity.
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = !var.use_spot
    provisioning_model  = var.use_spot ? "SPOT" : "STANDARD"
    # instance_termination_action left unset (DELETE, the default for Spot) rather
    # than requesting the newer STOP-on-preemption behavior — see the comment above
    # this block for why.
  }

  # On a STANDARD (non-Spot) instance, an in-guest `sudo shutdown -h now` transitions
  # the VM to TERMINATED (GCP's "stopped" state) and keeps the boot disk, with NO
  # extra configuration needed — unlike AWS, which requires explicitly setting
  # instance_initiated_shutdown_behavior = "stop" (its default allows either
  # behavior). This is what makes the runbook's dead-man switch
  # (`uv run gpt-train && gcloud storage rsync checkpoints/ gs://... && sudo shutdown -h now`)
  # safe here without an equivalent setting to remember.

  metadata = {
    ssh-keys = "${var.ssh_user}:${file(pathexpand(var.public_key_path))}"
    # enable-oslogin left at its project default deliberately: OS Login (GCP's IAM-
    # backed alternative to metadata SSH keys) is a real, more auditable option, but
    # turning it on here would silently ignore public_key_path/ssh_user, which is a
    # confusing failure mode for something this module can't detect at plan time.
  }

  metadata_startup_script = local.bootstrap_script

  service_account {
    email = google_service_account.gpu.email
    # cloud-platform + IAM bindings (iam.tf) is the modern pattern: the actual access
    # boundary is IAM, not this scope list — same "no long-lived keys on the box"
    # guarantee as the AWS module's instance-profile design, via a different
    # mechanism (rotating metadata-server credentials either way).
    scopes = ["cloud-platform"]
  }

  labels = { name = var.project }

  lifecycle {
    # A newer published image must never quietly replace a box mid-run — same
    # reasoning as the AWS module's `ignore_changes = [ami]`. Explicit opt-in via
    # `make replace-instance`.
    ignore_changes = [boot_disk[0].initialize_params[0].image]

    # Spot's real requirement is not a particular shutdown setting — it is that
    # checkpoints live somewhere the instance's death cannot reach. GCP gives only
    # ~30 seconds' preemption notice (see templates/bootstrap.sh.tftpl's preempt-watch
    # service) — meaningfully less than AWS's ~2 minutes, which makes periodic sync
    # even more load-bearing here, not less.
    precondition {
      condition     = !var.use_spot || (var.checkpoint_sync_minutes > 0 && (var.create_bucket || var.bucket_name != ""))
      error_message = "use_spot = true requires checkpoint_sync_minutes > 0 and a bucket. A preempted Spot VM is deleted with its boot disk — and every unsynced checkpoint — within ~30 seconds of notice."
    }
  }
}
