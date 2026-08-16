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
# A minimal custom role, not a predefined one: predefined compute roles (even the
# narrowest, roles/compute.instanceAdmin.v1) also grant start/reset/delete/
# setMachineType — this role grants only what the watchdog scripts actually call.
########################################

resource "google_project_iam_custom_role" "self_stop" {
  role_id     = replace("${var.project}_self_stop", "-", "_")
  title       = "${var.project} self-stop"
  description = "Exactly what the idle-shutdown and spot-preemption watchdogs need: read the instance's own status and stop it. Nothing else."
  permissions = [
    "compute.instances.get",
    "compute.instances.stop",
  ]
}

# Bound to the specific instance resource once it exists — see main.tf. Deferred
# here as a separate resource (rather than inline) because it must reference
# google_compute_instance.gpu, which is declared after this file in read order.
resource "google_compute_instance_iam_member" "self_stop" {
  count = var.instance_count > 0 ? 1 : 0

  project       = var.project_id
  zone          = var.zone
  instance_name = google_compute_instance.gpu[0].name
  role          = google_project_iam_custom_role.self_stop.id
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
