locals {
  # Account id + region make the name globally unique without a random suffix — a
  # random_id here would rename (and therefore destroy) the bucket on state loss.
  bucket_name = var.bucket_name != "" ? var.bucket_name : "${var.project}-${data.aws_caller_identity.current.account_id}-${var.region}"
}

resource "aws_s3_bucket" "corpus" {
  count = var.create_bucket ? 1 : 0

  bucket        = local.bucket_name
  force_destroy = var.force_destroy_bucket

  tags = { Name = local.bucket_name }
}

resource "aws_s3_bucket_public_access_block" "corpus" {
  count = var.create_bucket ? 1 : 0

  bucket                  = aws_s3_bucket.corpus[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "corpus" {
  count = var.create_bucket ? 1 : 0

  bucket = aws_s3_bucket.corpus[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "corpus" {
  count = var.create_bucket ? 1 : 0

  bucket = aws_s3_bucket.corpus[0].id

  # `aws s3 sync` of a multi-GB .bin uses multipart uploads. A dropped home connection
  # leaves the parts behind, billed as storage, invisible in the console's object list.
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
