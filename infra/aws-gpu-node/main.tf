########################################
# AMI
#
# The Base OSS Nvidia Driver AMI ships the NVIDIA driver + CUDA and nothing else.
# On a plain Ubuntu AMI you spend the first billed hour on drivers; on the full
# PyTorch DLAMI you pay for frameworks this repo replaces with its own uv-managed
# torch anyway.
########################################

data "aws_ami" "dl_base" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = var.ami_owners

  filter {
    name   = "name"
    values = [var.ami_name_filter]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

locals {
  ami_id = var.ami_id != "" ? var.ami_id : try(data.aws_ami.dl_base[0].id, "")

  key_name = var.existing_key_pair_name != "" ? var.existing_key_pair_name : try(aws_key_pair.this[0].key_name, "")

  user_data = templatefile("${path.module}/templates/bootstrap.sh.tftpl", {
    region                 = var.region
    bucket                 = local.bucket_name
    corpus_prefix          = var.corpus_prefix
    checkpoint_prefix      = var.checkpoint_prefix
    repo_url               = var.repo_url
    repo_dir_name          = var.repo_dir_name
    project_subdir         = var.project_subdir
    idle_shutdown_minutes   = var.idle_shutdown_minutes
    idle_gpu_threshold_pct  = var.idle_gpu_threshold_pct
    checkpoint_sync_minutes = var.checkpoint_sync_minutes
    use_spot                = var.use_spot
  })
}

########################################
# Key pair
########################################

resource "aws_key_pair" "this" {
  count = var.existing_key_pair_name == "" ? 1 : 0

  key_name   = "${var.project}-key"
  public_key = file(pathexpand(var.public_key_path))
}

########################################
# The box
########################################

resource "aws_instance" "gpu" {
  count = var.instance_count

  ami           = local.ami_id
  instance_type = var.instance_type
  subnet_id     = local.subnet_id
  key_name      = local.key_name

  vpc_security_group_ids = [aws_security_group.gpu.id]
  iam_instance_profile   = aws_iam_instance_profile.gpu.name

  associate_public_ip_address = true
  ebs_optimized               = true

  # 'stop' is the load-bearing setting for the runbook's dead-man switch:
  #   uv run gpt-train && aws s3 sync checkpoints/ s3://... && sudo shutdown -h now
  # With 'stop', that ends compute billing and keeps the EBS volume, so a restart
  # resumes from latest.pt. With 'terminate' the same command destroys the run.
  instance_initiated_shutdown_behavior = var.use_spot ? "terminate" : var.shutdown_behavior

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []

    content {
      market_type = "spot"

      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
        max_price                      = var.spot_max_price != "" ? var.spot_max_price : null
      }
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    throughput            = 250
    iops                  = 3000
    encrypted             = true
    delete_on_termination = true

    tags = { Name = "${var.project}-root" }
  }

  # IMDSv2 only. On a box reachable from the internet with a role that can write to
  # your bucket, IMDSv1's unauthenticated GET is the classic SSRF-to-credentials path.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
    instance_metadata_tags      = "enabled"
  }

  user_data                   = local.user_data
  user_data_replace_on_change = false

  tags = { Name = "${var.project}" }

  lifecycle {
    # A newer DLAMI must never quietly replace a box that is 14 hours into a run.
    # AMI upgrades are an explicit act: `make replace-instance`.
    ignore_changes = [ami]

    # Spot's real requirement is not a particular shutdown setting — it is that
    # checkpoints live somewhere the instance's death cannot reach. A reclaim gives
    # ~2 minutes' notice and then deletes the root volume; without periodic sync to
    # S3, every GPU-hour of the run goes with it.
    precondition {
      condition     = !var.use_spot || (var.checkpoint_sync_minutes > 0 && (var.create_bucket || var.bucket_name != ""))
      error_message = "use_spot = true requires checkpoint_sync_minutes > 0 and a bucket. A reclaimed Spot instance takes its root volume — and every unsynced checkpoint — with it."
    }
  }
}
