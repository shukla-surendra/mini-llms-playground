data "aws_caller_identity" "current" {}

########################################
# Where to launch
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

locals {
  # try() rather than a bare conditional: Terraform does not reliably short-circuit
  # the unused branch, so indexing a count = 0 data source would error even when the
  # variable is set.
  vpc_id = var.vpc_id != "" ? var.vpc_id : try(data.aws_vpc.default[0].id, "")

  subnet_id = var.subnet_id != "" ? var.subnet_id : try(sort(data.aws_subnets.candidates[0].ids)[0], "")

  # "My IP", resolved the same way the EC2 console's dropdown does it.
  detected_ip_cidr = try("${chomp(data.http.my_ip[0].response_body)}/32", "")
  ssh_cidrs        = length(var.allowed_ssh_cidrs) > 0 ? var.allowed_ssh_cidrs : compact([local.detected_ip_cidr])
}

data "http" "my_ip" {
  count = length(var.allowed_ssh_cidrs) == 0 ? 1 : 0
  url   = "https://checkip.amazonaws.com"
}

########################################
# Security group
########################################

resource "aws_security_group" "gpu" {
  name        = "${var.project}-sg"
  description = "SSH (and optionally the model API) for the ${var.project} training box"
  vpc_id      = local.vpc_id

  tags = { Name = "${var.project}-sg" }

  lifecycle {
    precondition {
      condition     = var.allow_open_ssh || !contains(local.ssh_cidrs, "0.0.0.0/0")
      error_message = "Refusing to open port 22 to 0.0.0.0/0. This box carries an IAM role with write access to your bucket. Set allowed_ssh_cidrs to your /32, or set allow_open_ssh = true if you truly mean it."
    }

    precondition {
      condition     = length(local.ssh_cidrs) > 0
      error_message = "No SSH CIDR resolved: public-IP auto-detection failed and allowed_ssh_cidrs is empty. Applying now would leave a box with no way in on port 22. Set allowed_ssh_cidrs explicitly."
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

# Egress stays wide open on purpose: the box pulls from PyPI, Hugging Face, GitHub and S3.
resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.gpu.id
  description       = "All outbound"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}
