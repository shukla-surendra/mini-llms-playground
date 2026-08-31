terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.60, < 7.0"
    }
    # Only used to auto-detect your public IP when allowed_ssh_cidrs is left empty.
    http = {
      source  = "hashicorp/http"
      version = "~> 3.4"
    }
  }

  # State is local by default: this is a single-operator lab, and a local
  # terraform.tfstate is one less thing to provision before the first GPU hour.
  # Move it to S3 the moment a second machine (or a teammate) applies this module —
  # two people with divergent local state will happily launch two pairs of instances.
  #
  # backend "s3" {
  #   bucket       = "<your-tf-state-bucket>"
  #   key          = "mini-llms-playground/aws-gpu-node-multi.tfstate"
  #   region       = "us-east-1"
  #   encrypt      = true
  #   use_lockfile = true # S3-native locking; no DynamoDB table needed
  # }
}
