########################################
# Where to launch
########################################

data "google_compute_network" "default" {
  count = var.network == "" ? 1 : 0
  name  = "default"
}

locals {
  network = var.network != "" ? var.network : try(data.google_compute_network.default[0].self_link, "")

  # "My IP", resolved the same way the AWS module's console-style "My IP" lookup works.
  detected_ip_cidr = try("${chomp(data.http.my_ip[0].response_body)}/32", "")
  ssh_cidrs        = length(var.allowed_ssh_cidrs) > 0 ? var.allowed_ssh_cidrs : compact([local.detected_ip_cidr])

  # Google's own IAP TCP-forwarding range — traffic from `gcloud compute ssh
  # --tunnel-through-iap` originates here, not from your own IP, so it needs its own
  # firewall rule regardless of what allowed_ssh_cidrs resolves to.
  iap_cidr = "35.235.240.0/20"
}

data "http" "my_ip" {
  count = length(var.allowed_ssh_cidrs) == 0 ? 1 : 0
  url   = "https://checkip.amazonaws.com" # provider-agnostic; same endpoint the AWS module already trusts
}

########################################
# Firewall
#
# Unlike an AWS Security Group, GCP VPC firewalls are directionless global rules
# matched by target tag, not attached to the instance as a single resource — the
# instance opts in via `tags` in main.tf. There is no explicit-egress-allow resource
# here because GCP's default network already implies allow-all egress unless a DENY
# rule says otherwise (the AWS module needs an explicit egress-allow rule only because
# Security Groups default-deny in both directions).
########################################

resource "google_compute_firewall" "ssh" {
  count = length(local.ssh_cidrs) > 0 ? 1 : 0

  name    = "${var.project}-allow-ssh"
  network = local.network
  # Keep this false: default false already refuses 0.0.0.0/0 unless allow_open_ssh
  # explicitly widens source_ranges to include it below.
  direction = "INGRESS"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = local.ssh_cidrs
  target_tags   = [var.project]

  lifecycle {
    precondition {
      condition     = var.allow_open_ssh || !contains(local.ssh_cidrs, "0.0.0.0/0")
      error_message = "Refusing to open port 22 to 0.0.0.0/0. This box carries a service account with write access to your bucket. Set allowed_ssh_cidrs to your /32, or set allow_open_ssh = true if you truly mean it."
    }

    precondition {
      condition     = length(local.ssh_cidrs) > 0
      error_message = "No SSH CIDR resolved: public-IP auto-detection failed and allowed_ssh_cidrs is empty. Applying now would leave a box with no way in on port 22 (and IAP is a separate opt-in, see enable_iap_ssh). Set allowed_ssh_cidrs explicitly."
    }
  }
}

resource "google_compute_firewall" "iap_ssh" {
  count = var.enable_iap_ssh ? 1 : 0

  name      = "${var.project}-allow-iap-ssh"
  network   = local.network
  direction = "INGRESS"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = [local.iap_cidr]
  target_tags   = [var.project]
}

resource "google_compute_firewall" "api" {
  count = length(var.allowed_api_cidrs) > 0 ? 1 : 0

  name      = "${var.project}-allow-api"
  network   = local.network
  direction = "INGRESS"

  allow {
    protocol = "tcp"
    ports    = [tostring(var.api_port)]
  }

  source_ranges = var.allowed_api_cidrs
  target_tags   = [var.project]
}
