########################################
# Instance service account
#
# The whole point: no long-lived key file ever touches the box. The instance gets
# rotating credentials from the metadata server, and terminating the box revokes them
# — same guarantee as the AWS module's instance-profile-only design, different
# mechanism (GCP has no direct analog of an EC2 instance profile; a service account
# attached to the VM's `service_account` block plays the same role).
########################################

resource "google_service_account" "gpu" {
  account_id   = "${var.project}-vm"
  display_name = "${var.project} training VM"

  lifecycle {
    precondition {
      condition     = var.create_bucket || var.bucket_name != ""
      error_message = "create_bucket = false requires bucket_name to point at an existing bucket."
    }
  }
}

########################################
# Bucket access: scoped to exactly this bucket, not roles/storage.admin project-wide
########################################

resource "google_storage_bucket_iam_member" "gpu_rw" {
  count = var.create_bucket ? 1 : 0

  bucket = google_storage_bucket.corpus[0].name
  role   = "roles/storage.objectAdmin" # get/create/delete/list objects in THIS bucket only
  member = "serviceAccount:${google_service_account.gpu.email}"
}

########################################
# Self-stop: what the idle watchdog and preempt watcher call
#
# GCP has no direct analog of the AWS module's tag-conditioned IAM policy (ABAC on
# resource tags is a mature, first-class AWS IAM feature; GCP's IAM Conditions don't
# reliably support matching arbitrary instance labels for compute actions the same
# way). The tighter GCP-native equivalent instead: bind a role directly to the ONE
# instance resource this module manages (`google_compute_instance_iam_member`,
# resource-level IAM, not project-level) — arguably a stronger scope than AWS's tag
# match, since it names the exact instance rather than "any instance with this tag."
# A minimal custom role, not a predefined one, was the original intent here:
# predefined compute roles (even the narrowest, roles/compute.instanceAdmin.v1) also
# grant start/reset/delete/setMachineType — a purpose-built role would grant only
# what the watchdog scripts actually call.
#
# DEVIATION (2026-08-17, first real apply of this never-before-applied module): the
# deploying service account (llm-training-dev-sa) can create SAs/buckets/firewalls
# but lacks iam.roles.create at the project level (403 IAM_PERMISSION_DENIED),
# so google_project_iam_custom_role fails outright regardless of what it declares.
# Falling back to the predefined roles/compute.instanceAdmin.v1, still bound only to
# this ONE instance resource (not project-wide) via google_compute_instance_iam_member
# — tighter than a project-level grant, but broader than the intended get+stop-only
# role on this specific box (also grants start/reset/delete/setMachineType on it).
# Revert to the custom-role version above if the deploying principal is ever granted
# roles/iam.roleAdmin (or Owner) on this project.
########################################

# FURTHER DEVIATION (2026-08-17, same apply): even the predefined-role fallback above
# fails — this service account also lacks compute.instances.setIamPolicy (403), one
# level deeper than the iam.roles.create gap. Disabled (count=0) rather than keep
# spending apply cycles on a non-blocking safety feature while the instance bills.
# Effect: the idle-shutdown and preempt-watch scripts on the box (still installed by
# bootstrap.sh.tftpl) will detect the condition but CANNOT self-stop the instance —
# manual `make down`/`make stop` is required, same as the disabled budget alert in
# budget.tf. Re-enable (count = var.instance_count > 0 ? 1 : 0) once the deploying
# principal has compute.instances.setIamPolicy, or an Owner grants it directly via
# `gcloud compute instances add-iam-policy-binding` outside Terraform.
resource "google_compute_instance_iam_member" "self_stop" {
  count = 0

  project       = var.project_id
  zone          = var.zone
  instance_name = google_compute_instance.gpu[0].name
  role          = "roles/compute.instanceAdmin.v1"
  member        = "serviceAccount:${google_service_account.gpu.email}"
}

########################################
# IAP tunneling: a way in that needs no open port and no on-box key
#
# This grants the ABILITY to tunnel to THIS instance to the operators listed in
# iap_ssh_members — unlike the AWS module's SSM attachment (which grants the
# INSTANCE permission to be managed), IAP tunnel access is a permission on the
# CALLER, not the box. If you can already `terraform apply` this module, your own
# GCP identity almost certainly already has enough project-level access to use IAP
# without this — iap_ssh_members exists for granting it to a narrower or different
# principal (a teammate, a service account, a group) than whoever ran apply.
########################################

resource "google_compute_instance_iam_member" "iap_tunnel" {
  for_each = var.enable_iap_ssh && var.instance_count > 0 ? toset(var.iap_ssh_members) : toset([])

  project       = var.project_id
  zone          = var.zone
  instance_name = google_compute_instance.gpu[0].name
  role          = "roles/iap.tunnelResourceAccessor"
  member        = each.value
}
