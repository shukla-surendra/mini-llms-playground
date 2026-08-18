locals {
  # project_id substitutes for the AWS module's account-id uniqueness trick — GCS
  # bucket names are unique across ALL of GCP, not just this project, so a bare
  # project-name prefix (unlike S3, scoped per-account) would collide with someone
  # else's bucket. No random suffix here either, for the same reason as the AWS
  # module: a random_id would rename (and destroy) the bucket on state loss.
  bucket_name = var.bucket_name != "" ? var.bucket_name : "${var.project}-${var.project_id}-${var.region}"
}

resource "google_storage_bucket" "corpus" {
  count = var.create_bucket ? 1 : 0

  name          = local.bucket_name
  location      = var.region
  storage_class = "STANDARD"
  force_destroy = var.force_destroy_bucket

  uniform_bucket_level_access = true # no ACL surface — IAM only, the GCS analog of the AWS module's public-access-block

  public_access_prevention = "enforced"

  # No explicit `encryption` block: omitting it means Google-managed encryption keys,
  # equivalent in practice to the AWS module's explicit AES256 SSE — no separate
  # resource or block needed to turn it on, it's the unconditional default.

  # `gsutil rsync`/`gcloud storage rsync` of a multi-GB .bin uses resumable/multipart
  # uploads under the hood. A dropped home connection can leave incomplete objects
  # behind — this rule is GCS's analog of the AWS module's abort-incomplete-multipart
  # lifecycle rule.
  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }

  labels = { name = local.bucket_name }
}
