########################################
# Instance role
#
# The whole point: no long-lived access keys ever touch the box. The instance gets
# rotating credentials from IMDS, and terminating the box revokes them.
########################################

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "gpu" {
  name               = "${var.project}-instance-role"
  assume_role_policy = data.aws_iam_policy_document.assume.json

  lifecycle {
    precondition {
      condition     = var.create_bucket || var.bucket_name != ""
      error_message = "create_bucket = false requires bucket_name to point at an existing bucket."
    }
  }
}

resource "aws_iam_instance_profile" "gpu" {
  name = "${var.project}-instance-profile"
  role = aws_iam_role.gpu.name
}

########################################
# S3: exactly one bucket, not s3:* on *
########################################

data "aws_iam_policy_document" "s3" {
  statement {
    sid       = "ListTheBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = ["arn:aws:s3:::${local.bucket_name}"]
  }

  statement {
    sid = "ReadWriteObjects"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
    ]
    resources = ["arn:aws:s3:::${local.bucket_name}/*"]
  }
}

resource "aws_iam_role_policy" "s3" {
  name   = "${var.project}-s3"
  role   = aws_iam_role.gpu.id
  policy = data.aws_iam_policy_document.s3.json
}

########################################
# Self-stop: what the idle watchdog calls (the runbook's `shutdown -h now` dead-man
# switch needs no IAM at all — it relies on instance_initiated_shutdown_behavior)
#
# StopInstances is scoped by tag rather than by instance id — an id would create a
# cycle (policy -> role -> profile -> instance). Describe* has no resource-level
# permissions in EC2, so it is "*" by necessity; it is read-only.
########################################

data "aws_iam_policy_document" "self_stop" {
  statement {
    sid       = "DescribeForWatchdog"
    actions   = ["ec2:DescribeInstances", "ec2:DescribeTags"]
    resources = ["*"]
  }

  statement {
    sid       = "StopOwnProjectInstances"
    actions   = ["ec2:StopInstances"]
    resources = ["arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:instance/*"]

    condition {
      test     = "StringEquals"
      variable = "ec2:ResourceTag/Project"
      values   = [var.project]
    }
  }
}

resource "aws_iam_role_policy" "self_stop" {
  name   = "${var.project}-self-stop"
  role   = aws_iam_role.gpu.id
  policy = data.aws_iam_policy_document.self_stop.json
}

########################################
# Session Manager: a way in that needs no open port and no key
########################################

resource "aws_iam_role_policy_attachment" "ssm" {
  count = var.enable_ssm ? 1 : 0

  role       = aws_iam_role.gpu.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
