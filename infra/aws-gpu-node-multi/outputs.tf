output "region" {
  description = "Region everything lives in (the Makefile reads this)."
  value       = var.region
}

output "master_instance_id" {
  description = "EC2 instance id, rank 0 (empty when instance_count = 0)."
  value       = try(aws_instance.gpu_master[0].id, "")
}

output "worker_instance_id" {
  description = "EC2 instance id, rank 1 (empty when instance_count = 0)."
  value       = try(aws_instance.gpu_worker[0].id, "")
}

output "master_public_ip" {
  description = "Master's public IPv4. Changes on every stop/start — no Elastic IP, since an idle one is billed."
  value       = try(aws_instance.gpu_master[0].public_ip, "")
}

output "worker_public_ip" {
  description = "Worker's public IPv4. Changes on every stop/start."
  value       = try(aws_instance.gpu_worker[0].public_ip, "")
}

output "master_private_ip" {
  description = "Master's private IP — statically assigned (see network.tf), this is the --master_addr both nodes' generated launch_ddp.sh already uses."
  value       = local.master_private_ip
}

output "ssh_master" {
  description = "Copy-paste SSH into the master (rank 0)."
  value       = try("ssh -i ${var.ssh_private_key_path} ubuntu@${aws_instance.gpu_master[0].public_ip}", "")
}

output "ssh_worker" {
  description = "Copy-paste SSH into the worker (rank 1)."
  value       = try("ssh -i ${var.ssh_private_key_path} ubuntu@${aws_instance.gpu_worker[0].public_ip}", "")
}

output "ssm_master" {
  description = "Session Manager alternative into the master — no open port, no key."
  value       = var.enable_ssm ? try("aws ssm start-session --region ${var.region} --target ${aws_instance.gpu_master[0].id}", "") : "disabled (enable_ssm = false)"
}

output "ssm_worker" {
  description = "Session Manager alternative into the worker."
  value       = var.enable_ssm ? try("aws ssm start-session --region ${var.region} --target ${aws_instance.gpu_worker[0].id}", "") : "disabled (enable_ssm = false)"
}

output "bucket" {
  description = "Bucket both instances' roles can read/write. Upload the token files and tokenizer here BEFORE launching."
  value       = local.bucket_name
}

output "upload_corpus_hint" {
  description = "What to run on the Mac from inside the project dir. Token files only — .txt stays home."
  value       = "aws s3 sync data/ s3://${local.bucket_name}/${var.corpus_prefix != "" ? var.corpus_prefix : "corpus/"} --exclude '*' --include '*.bin'"
}

output "upload_tokenizer_hint" {
  description = "What to run on the Mac from inside the project dir, to make the trained tokenizer available on both nodes."
  value       = "aws s3 sync tokenizer/ s3://${local.bucket_name}/${var.tokenizer_prefix != "" ? var.tokenizer_prefix : "tokenizer/"} --exclude '*' --include 'tokenizer.json'"
}

output "upload_checkpoint_hint" {
  description = "What to run on the Mac, to make an existing run resumable on both nodes. Empty unless checkpoint_prefix is set."
  value       = var.checkpoint_prefix != "" ? "aws s3 sync checkpoints/ s3://${local.bucket_name}/${var.checkpoint_prefix}" : ""
}

output "launch_reminder" {
  description = "The actual multi-node start sequence — both boxes generate their own ~/launch_ddp.sh at boot (see templates/bootstrap.sh.tftpl), already filled in with the correct --node_rank and --master_addr for that box."
  value       = "SSH into both nodes, then on EACH: tmux new -s train, then: bash ~/launch_ddp.sh"
}

output "project_subdir" {
  description = "Which project both instances are set up for (the Makefile reads this for PROJECT_DIR)."
  value       = var.project_subdir
}

output "repo_dir_name" {
  description = "Directory the repo clones into on both nodes under /home/ubuntu."
  value       = var.repo_dir_name
}

output "corpus_prefix" {
  description = "S3 prefix both instances pull data/ from."
  value       = var.corpus_prefix
}

output "tokenizer_prefix" {
  description = "S3 prefix both instances pull tokenizer/ from."
  value       = var.tokenizer_prefix
}

output "checkpoint_prefix" {
  description = "S3 prefix both instances pull checkpoints/ from (only the master writes back)."
  value       = var.checkpoint_prefix
}

output "ami_id" {
  description = "Resolved AMI, used by both instances. Pin this into ami_id for a byte-identical pair next time."
  value       = local.ami_id
}

output "allowed_ssh_cidrs" {
  description = "What port 22 is actually open to, on both nodes."
  value       = local.ssh_cidrs
}

output "stop_master_command" {
  description = "Ends compute billing for the master, keeps its EBS volume."
  value       = try("aws ec2 stop-instances --region ${var.region} --instance-ids ${aws_instance.gpu_master[0].id}", "")
}

output "stop_worker_command" {
  description = "Ends compute billing for the worker, keeps its EBS volume."
  value       = try("aws ec2 stop-instances --region ${var.region} --instance-ids ${aws_instance.gpu_worker[0].id}", "")
}

output "estimated_hourly_usd_per_node" {
  description = "Rough on-demand rate for the chosen type, us-east-1, per node — double it for the pair. Sanity only, not a billing source of truth."
  value = lookup({
    "g6.xlarge"   = 0.8048
    "g6.2xlarge"  = 0.9776
    "g5.xlarge"   = 1.006
    "g4dn.xlarge" = 0.526
  }, var.instance_type, -1)
}

output "target_tokens" {
  description = "GPT_TARGET_TOKENS baked into both nodes' generated launch_ddp.sh — see variables.tf for the Chinchilla-optimal reasoning behind the default."
  value       = var.target_tokens
}
