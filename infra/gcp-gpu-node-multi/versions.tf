terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30, < 7.0"
    }
    # Only used to auto-detect your public IP when allowed_ssh_cidrs is left empty —
    # same trick the sibling infra/aws-gpu-node module uses.
    http = {
      source  = "hashicorp/http"
      version = "~> 3.4"
    }
  }

  # State is local by default: this is a single-operator lab, and a local
  # terraform.tfstate is one less thing to provision before the first GPU hour.
  # Move it to GCS the moment a second machine (or a teammate) applies this module —
  # two people with divergent local state will happily launch two billing VMs.
  #
  # backend "gcs" {
  #   bucket = "<your-tf-state-bucket>"
  #   prefix = "mini-llms-playground/gcp-gpu-node"
  # }
}
