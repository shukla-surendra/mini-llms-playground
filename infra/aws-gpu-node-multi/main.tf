########################################
# AMI
#
# The Base OSS Nvidia Driver AMI ships the NVIDIA driver + CUDA and nothing else.
# On a plain Ubuntu AMI you spend the first billed hour, on both boxes, on drivers;
# on the full PyTorch DLAMI you pay for frameworks this repo replaces with its own
# uv-managed torch anyway.
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

  # Common bootstrap args both nodes share; node_rank/master_addr differ per node
  # and are added at each templatefile() call below.
  bootstrap_common = {
    region                  = var.region
    bucket                  = local.bucket_name
    corpus_prefix           = var.corpus_prefix
    tokenizer_prefix        = var.tokenizer_prefix
    checkpoint_prefix       = var.checkpoint_prefix
    repo_url                = var.repo_url
    repo_dir_name           = var.repo_dir_name
    project_subdir          = var.project_subdir
    idle_shutdown_minutes   = var.idle_shutdown_minutes
    idle_gpu_threshold_pct  = var.idle_gpu_threshold_pct
    checkpoint_sync_minutes = var.checkpoint_sync_minutes
    use_spot                = var.use_spot
    world_size              = 2
    dist_port               = var.dist_port
    master_addr             = local.master_private_ip
    target_tokens           = var.target_tokens
    grad_accum_steps        = var.grad_accum_steps
    batch_size              = var.batch_size
  }

  user_data_master = templatefile("${path.module}/templates/bootstrap.sh.tftpl", merge(local.bootstrap_common, {
    node_rank = 0
    is_master = true
    sync_ckpt = true # only rank 0 writes checkpoints (see trainer.py) — only rank 0 uploads them
  }))

  user_data_worker = templatefile("${path.module}/templates/bootstrap.sh.tftpl", merge(local.bootstrap_common, {
    node_rank = 1
    is_master = false
    sync_ckpt = false
  }))
}

########################################
# Key pair — shared by both nodes
########################################

resource "aws_key_pair" "this" {
  count = var.existing_key_pair_name == "" ? 1 : 0

  key_name   = "${var.project}-key"
  public_key = file(pathexpand(var.public_key_path))
}

########################################
# Node 0 — rank 0 / master. Gets a static private_ip (see network.tf's
# local.master_private_ip) so node 1's bootstrap can reference it without a
# resource-creation-order dependency between the two instances.
########################################

resource "aws_instance" "gpu_master" {
  count = var.instance_count > 0 ? 1 : 0

  ami           = local.ami_id
  instance_type = var.instance_type
  subnet_id     = local.subnet_id
  private_ip    = local.master_private_ip
  key_name      = local.key_name

  vpc_security_group_ids = [aws_security_group.gpu.id]
  iam_instance_profile   = aws_iam_instance_profile.gpu.name

  associate_public_ip_address = true
  ebs_optimized               = true

  # 'stop' is the load-bearing setting for the dead-man switch documented in
  # docs/RESUME_TRAINING.md. With 'stop', an in-guest shutdown ends compute billing
  # and keeps the EBS volume, so a restart resumes from the last synced checkpoint.
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

    tags = { Name = "${var.project}-master-root" }
  }

  # IMDSv2 only — same reasoning as the single-GPU sibling module.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
    instance_metadata_tags      = "enabled"
  }

  user_data                   = local.user_data_master
  user_data_replace_on_change = false

  tags = { Name = "${var.project}-master", Role = "ddp-rank-0" }

  lifecycle {
    ignore_changes = [ami]

    precondition {
      condition     = !var.use_spot || (var.checkpoint_sync_minutes > 0 && (var.create_bucket || var.bucket_name != ""))
      error_message = "use_spot = true requires checkpoint_sync_minutes > 0 and a bucket. A reclaimed Spot instance takes its root volume — and every unsynced checkpoint — with it, and here a reclaim of EITHER node also kills the worker's half of the run."
    }
  }
}

########################################
# Node 1 — rank 1 / worker. Same shape as the master except no static private_ip
# (nothing depends on the worker's own address) and node_rank/is_master differ.
########################################

resource "aws_instance" "gpu_worker" {
  count = var.instance_count > 0 ? 1 : 0

  ami           = local.ami_id
  instance_type = var.instance_type
  subnet_id     = local.subnet_id
  key_name      = local.key_name

  vpc_security_group_ids = [aws_security_group.gpu.id]
  iam_instance_profile   = aws_iam_instance_profile.gpu.name

  associate_public_ip_address = true
  ebs_optimized               = true

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

    tags = { Name = "${var.project}-worker-root" }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
    instance_metadata_tags      = "enabled"
  }

  user_data                   = local.user_data_worker
  user_data_replace_on_change = false

  tags = { Name = "${var.project}-worker", Role = "ddp-rank-1" }

  lifecycle {
    ignore_changes = [ami]

    precondition {
      condition     = !var.use_spot || (var.checkpoint_sync_minutes > 0 && (var.create_bucket || var.bucket_name != ""))
      error_message = "use_spot = true requires checkpoint_sync_minutes > 0 and a bucket. A reclaimed Spot instance takes its root volume with it, and here a reclaim of EITHER node also kills the master's half of the run."
    }
  }
}
