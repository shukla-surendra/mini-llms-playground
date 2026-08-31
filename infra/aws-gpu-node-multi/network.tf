data "aws_caller_identity" "current" {}

########################################
# Where to launch — one subnet, one AZ, for BOTH nodes
########################################

data "aws_vpc" "default" {
  count   = var.vpc_id == "" ? 1 : 0
  default = true
}

# G6 capacity is not in every AZ of every region. Asking EC2 which AZs offer the
# instance type — instead of grabbing the first default subnet — turns a launch-time
# "Unsupported" error into a plan-time selection.
data "aws_ec2_instance_type_offerings" "gpu" {
  count         = var.subnet_id == "" ? 1 : 0
  location_type = "availability-zone"

  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

data "aws_subnets" "candidates" {
  count = var.subnet_id == "" ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }

  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.gpu[0].locations
  }
}

# Needed for its CIDR block (to statically derive the master's private IP below) —
# the aws_subnets data source above only returns ids, not full subnet attributes.
data "aws_subnet" "selected" {
  id = local.subnet_id
}

locals {
  # try() rather than a bare conditional: Terraform does not reliably short-circuit
  # the unused branch, so indexing a count = 0 data source would error even when the
  # variable is set.
  vpc_id = var.vpc_id != "" ? var.vpc_id : try(data.aws_vpc.default[0].id, "")

  # Both nodes always resolve to this ONE subnet — deliberately no per-instance
  # variation. Same-AZ is the whole point (free + lower-latency private-IP traffic,
  # which matters for DDP's gradient sync specifically, not just for cost).
  subnet_id = var.subnet_id != "" ? var.subnet_id : try(sort(data.aws_subnets.candidates[0].ids)[0], "")

  # Statically assigned, not AWS-auto-assigned, specifically so the worker's
  # bootstrap script can be given the master's address without main.tf's
  # aws_instance.gpu_worker depending on aws_instance.gpu_master's own computed
  # private_ip attribute — see variables.tf's master_ip_host_offset for why that
  # would otherwise be a problem (it isn't a cycle across the two *resources*, but
  # this keeps both instances' user_data renderable from plan-time-known values
  # alone, with no creation-order dependency between them at all).
  master_private_ip = cidrhost(data.aws_subnet.selected.cidr_block, var.master_ip_host_offset)

  # "My IP", resolved the same way the EC2 console's dropdown does it.
  detected_ip_cidr = try("${chomp(data.http.my_ip[0].response_body)}/32", "")
  ssh_cidrs        = length(var.allowed_ssh_cidrs) > 0 ? var.allowed_ssh_cidrs : compact([local.detected_ip_cidr])
}

data "http" "my_ip" {
  count = length(var.allowed_ssh_cidrs) == 0 ? 1 : 0
  url   = "https://checkip.amazonaws.com"
}

########################################
# Security group — shared by both nodes
########################################

resource "aws_security_group" "gpu" {
  name        = "${var.project}-sg"
  description = "SSH, optional API, and inter-node DDP traffic for the ${var.project} training pair"
  vpc_id      = local.vpc_id

  tags = { Name = "${var.project}-sg" }

  lifecycle {
    precondition {
      condition     = var.allow_open_ssh || !contains(local.ssh_cidrs, "0.0.0.0/0")
      error_message = "Refusing to open port 22 to 0.0.0.0/0. Both boxes carry an IAM role with write access to your bucket. Set allowed_ssh_cidrs to your /32, or set allow_open_ssh = true if you truly mean it."
    }

    precondition {
      condition     = length(local.ssh_cidrs) > 0
      error_message = "No SSH CIDR resolved: public-IP auto-detection failed and allowed_ssh_cidrs is empty. Applying now would leave both boxes with no way in on port 22. Set allowed_ssh_cidrs explicitly."
    }

    precondition {
      condition     = local.subnet_id != ""
      error_message = "No subnet in the chosen VPC sits in an AZ that offers ${var.instance_type}. Pass subnet_id explicitly, or try another region."
    }
  }
}

# Rules as standalone resources: editing one CIDR then re-applying does not churn the
# whole group (and does not momentarily drop your own SSH access mid-apply).
resource "aws_vpc_security_group_ingress_rule" "ssh" {
  for_each = toset(local.ssh_cidrs)

  security_group_id = aws_security_group.gpu.id
  description       = "SSH"
  cidr_ipv4         = each.value
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "api" {
  for_each = toset(var.allowed_api_cidrs)

  security_group_id = aws_security_group.gpu.id
  description       = "Model serving API"
  cidr_ipv4         = each.value
  from_port         = var.api_port
  to_port           = var.api_port
  ip_protocol       = "tcp"
}

# torchrun's own rendezvous port, between the two nodes only. Self-referencing (the
# source is this SAME security group, not a CIDR) — both instances share it, so this
# one rule covers node-to-node traffic in both directions without naming either
# instance's IP explicitly, and survives either instance being replaced.
resource "aws_vpc_security_group_ingress_rule" "dist_rendezvous" {
  security_group_id            = aws_security_group.gpu.id
  description                  = "torchrun rendezvous (master_port)"
  referenced_security_group_id = aws_security_group.gpu.id
  from_port                    = var.dist_port
  to_port                      = var.dist_port
  ip_protocol                  = "tcp"
}

# NCCL negotiates additional ephemeral TCP ports for the actual gradient/tensor
# traffic after rendezvous completes on dist_port — it does not stay on one port.
# Rather than trying to enumerate NCCL's dynamic range exactly (version- and
# config-dependent), open the standard ephemeral range between the two nodes only —
# the common real-world pattern for small NCCL clusters, and still scoped to
# "traffic between these two training boxes," not the internet.
resource "aws_vpc_security_group_ingress_rule" "dist_ephemeral" {
  security_group_id            = aws_security_group.gpu.id
  description                  = "NCCL data-plane ports (post-rendezvous)"
  referenced_security_group_id = aws_security_group.gpu.id
  from_port                    = 1024
  to_port                      = 65535
  ip_protocol                  = "tcp"
}

# Egress stays wide open on purpose: both boxes pull from PyPI, Hugging Face, GitHub and S3.
resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.gpu.id
  description       = "All outbound"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}
