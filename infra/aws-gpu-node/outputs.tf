output "region" {
  description = "Region everything lives in (the Makefile reads this)."
  value       = var.region
}

output "instance_id" {
  description = "EC2 instance id (empty when instance_count = 0)."
  value       = try(aws_instance.gpu[0].id, "")
}

output "public_ip" {
  description = "Public IPv4. Changes on every stop/start — no Elastic IP here, since an idle EIP is billed."
  value       = try(aws_instance.gpu[0].public_ip, "")
}

output "ssh_command" {
  description = "Copy-paste SSH."
  value       = try("ssh -i ${var.ssh_private_key_path} ubuntu@${aws_instance.gpu[0].public_ip}", "")
}

output "ssm_command" {
  description = "Session Manager alternative — no open port, no key, works after your IP changes."
  value       = var.enable_ssm ? try("aws ssm start-session --region ${var.region} --target ${aws_instance.gpu[0].id}", "") : "disabled (enable_ssm = false)"
}

output "bucket" {
  description = "Bucket the instance role can read/write. Upload token files here BEFORE launching."
  value       = local.bucket_name
}

output "upload_corpus_hint" {
  description = "What to run on the Mac from inside a project dir. Token files only — .txt and hf_cache stay home."
  value       = "aws s3 sync data/ s3://${local.bucket_name}/${var.corpus_prefix != "" ? var.corpus_prefix : "corpus/"} --exclude '*' --include '*.bin' --include '*.bin.json'"
}

output "upload_checkpoint_hint" {
  description = "What to run on the Mac from inside a project dir, to make an existing run resumable on the instance. Empty unless checkpoint_prefix is set."
  value       = var.checkpoint_prefix != "" ? "aws s3 sync checkpoints/ s3://${local.bucket_name}/${var.checkpoint_prefix}" : ""
}

output "project_subdir" {
  description = "Which project the instance is set up for (the Makefile reads this for PROJECT_DIR)."
  value       = var.project_subdir
}

output "repo_dir_name" {
  description = "Directory the repo clones into on the instance under /home/ubuntu (the Makefile reads this to build the remote project path for `make gpu`)."
  value       = var.repo_dir_name
}

output "corpus_prefix" {
  description = "S3 prefix the instance pulls data/ from (the Makefile reads this for PREFIX)."
  value       = var.corpus_prefix
}

output "checkpoint_prefix" {
  description = "S3 prefix the instance pulls checkpoints/ from (the Makefile reads this for CKPT_PREFIX)."
  value       = var.checkpoint_prefix
}

output "ami_id" {
  description = "Resolved AMI. Pin this into ami_id if you want a byte-identical box next time."
  value       = local.ami_id
}

output "allowed_ssh_cidrs" {
  description = "What port 22 is actually open to."
  value       = local.ssh_cidrs
}

output "stop_command" {
  description = "Ends compute billing, keeps the EBS volume and everything on it."
  value       = try("aws ec2 stop-instances --region ${var.region} --instance-ids ${aws_instance.gpu[0].id}", "")
}

output "estimated_hourly_usd" {
  description = "Rough on-demand rate for the chosen type, us-east-1, for sanity only — not a billing source of truth."
  value = lookup({
    "g6.xlarge"   = 0.8048
    "g6.2xlarge"  = 0.9776
    "g5.xlarge"   = 1.006
    "g4dn.xlarge" = 0.526
    "p4d.24xlarge" = 32.7726
  }, var.instance_type, -1)
}
