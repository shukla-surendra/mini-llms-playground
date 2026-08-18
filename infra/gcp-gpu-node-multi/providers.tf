provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone

  # Every resource this module creates that supports labels gets these — the GCP
  # analog of the sibling AWS module's default_tags, and what makes the self-stop
  # IAM binding safely scoped by label rather than a hardcoded instance name.
  default_labels = {
    project    = var.project
    managed-by = "terraform"
    repo       = "mini-llms-playground"
    module     = "infra-gcp-gpu-node"
  }
}
