provider "aws" {
  region = var.region

  # Every resource here carries these, which is what makes the IAM self-stop policy
  # (see iam.tf) safely scoped by tag rather than by a hardcoded instance id.
  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
      Repo      = "mini-llms-playground"
      Module    = "infra/aws-gpu-node-multi"
    }
  }
}
