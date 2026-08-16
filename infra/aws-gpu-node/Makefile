TF      ?= terraform
REGION  ?= $(shell $(TF) output -raw region 2>/dev/null || echo us-east-1)
IID     ?= $(shell $(TF) output -raw instance_id 2>/dev/null)
IP      ?= $(shell $(TF) output -raw public_ip 2>/dev/null)
BUCKET  ?= $(shell $(TF) output -raw bucket 2>/dev/null)
KEY     ?= ~/.ssh/id_ed25519
TYPE    ?= g6.xlarge

# Derived from terraform.tfvars via `terraform output`, so PROJECT_DIR/PREFIX can
# never drift from what the instance was actually bootstrapped to sync — override
# any of the three on the command line (e.g. `make upload-corpus PROJECT_DIR=...`)
# only for a one-off that intentionally differs from the configured module.
SUBDIR      ?= $(shell $(TF) output -raw project_subdir 2>/dev/null)
PROJECT_DIR ?= ../../$(SUBDIR)
PREFIX      ?= $(shell $(TF) output -raw corpus_prefix 2>/dev/null)
CKPT_PREFIX ?= $(shell $(TF) output -raw checkpoint_prefix 2>/dev/null)
REPO_DIR_NAME ?= $(shell $(TF) output -raw repo_dir_name 2>/dev/null)
REMOTE_DIR    ?= ~/$(REPO_DIR_NAME)/$(SUBDIR)

.PHONY: help init fmt validate plan apply ssh ssm tunnel bootstrap-log gpu \
        upload-corpus upload-checkpoint download-checkpoints stop start status \
        replace destroy up down spot-price sync-log

help:
	@echo "Setup"
	@echo "  make init                 - terraform init"
	@echo "  make plan                 - what would change"
	@echo "  make apply                - create/update the GPU node (BILLING STARTS)"
	@echo ""
	@echo "Data (run these BEFORE apply — tokenizing on a GPU is money for nothing)"
	@echo "  make upload-corpus        - sync *.bin + *.bin.json from PROJECT_DIR to s3://$(BUCKET)/$(PREFIX)"
	@echo "  make upload-checkpoint    - sync an existing run's checkpoints/ up, for gpt-train to auto-resume"
	@echo "  make download-checkpoints - pull checkpoints/ back down from S3"
	@echo ""
	@echo "Use"
	@echo "  make ssh                  - ssh in"
	@echo "  make ssm                  - Session Manager shell (no open port, no key)"
	@echo "  make tunnel               - forward the model API to localhost:8000"
	@echo "  make bootstrap-log        - tail cloud-init bootstrap output"
	@echo "  make gpu                  - nvidia-smi + bf16 check on the box"
	@echo "  make status               - instance state + type + public ip"
	@echo ""
	@echo "  make sync-log             - tail the periodic checkpoint-sync journal"
	@echo ""
	@echo "Money  (cheapest usable pattern: make spot-price -> up -> ... -> down)"
	@echo "  make spot-price           - live spot price for $(TYPE) per AZ, before you commit"
	@echo "  make down                 - DESTROY the instance, keep bucket/IAM/SG. Bill -> ~\$$0"
	@echo "  make up                   - recreate the instance when you next need a GPU"
	@echo "  make stop                 - pause instead: ends compute billing, still ~\$$8/mo for EBS"
	@echo "  make start                - resume a stopped instance (new public IP)"
	@echo "  make replace              - rebuild the instance on the current AMI"
	@echo "  make destroy              - tear everything down (bucket survives unless force_destroy_bucket)"

init:
	$(TF) init

fmt:
	$(TF) fmt -recursive

validate: init
	$(TF) validate

plan:
	$(TF) plan

apply:
	$(TF) apply

ssh:
	ssh -i $(KEY) ubuntu@$(IP)

# No inbound rule, no key — works after your home IP rotates.
ssm:
	aws ssm start-session --region $(REGION) --target $(IID)

# The API stays closed in the security group; this is the way in that needs no rule.
tunnel:
	ssh -i $(KEY) -N -L 8000:localhost:8000 ubuntu@$(IP)

bootstrap-log:
	ssh -i $(KEY) ubuntu@$(IP) 'tail -f /var/log/gpu-node-bootstrap.log'

# torch lives in the project's uv-managed venv (created by `uv sync` at boot), not
# system Python — bare `python3 -c "import torch"` would fail with ModuleNotFoundError
# regardless of whether the GPU itself is fine. `uv run` picks up that venv correctly.
gpu:
	ssh -i $(KEY) ubuntu@$(IP) 'nvidia-smi; cd $(REMOTE_DIR) && uv run python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"'

upload-corpus:
	cd $(PROJECT_DIR) && aws s3 sync data/ s3://$(BUCKET)/$(PREFIX) \
		--exclude "*" --include "*.bin" --include "*.bin.json"

# Makes an existing local run resumable on the instance: gpt-train auto-resumes
# from checkpoints/<label>/latest.pt with no flag once this lands in the right
# place at boot (checkpoint_prefix in terraform.tfvars must be set to match).
upload-checkpoint:
	cd $(PROJECT_DIR) && aws s3 sync checkpoints/ s3://$(BUCKET)/$(CKPT_PREFIX) \
		--exclude "*.DS_Store"

download-checkpoints:
	cd $(PROJECT_DIR) && aws s3 sync s3://$(BUCKET)/$(CKPT_PREFIX) checkpoints/

sync-log:
	ssh -i $(KEY) ubuntu@$(IP) 'journalctl -u checkpoint-sync -u spot-watch -f'

# What the spot market is actually charging right now, per AZ. Read this before
# `make up` — the discount varies by AZ and by hour, and the cheapest AZ is not
# always the one the module would otherwise pick.
spot-price:
	aws ec2 describe-spot-price-history --region $(REGION) \
		--instance-types $(TYPE) --product-descriptions "Linux/UNIX" \
		--start-time $$(date -u +%Y-%m-%dT%H:%M:%S) \
		--query 'SpotPriceHistory[].{AZ:AvailabilityZone,USD_per_hour:SpotPrice}' \
		--output table

# The between-runs pattern. Destroying the instance drops the EBS bill to zero;
# the bucket, IAM role, security group and key pair cost nothing to keep, so the
# next `make up` is a 3-minute bootstrap rather than a re-setup.
# Sync your checkpoints first — this deletes the root volume.
down:
	$(TF) apply -var instance_count=0

up:
	$(TF) apply -var instance_count=1

stop:
	aws ec2 stop-instances --region $(REGION) --instance-ids $(IID)

start:
	aws ec2 start-instances --region $(REGION) --instance-ids $(IID)
	@echo "public IP changes on start — run 'terraform refresh && make status'"

status:
	aws ec2 describe-instances --region $(REGION) --instance-ids $(IID) \
		--query 'Reservations[].Instances[].{State:State.Name,Type:InstanceType,IP:PublicIpAddress,AZ:Placement.AvailabilityZone}' \
		--output table

# AMI changes are ignored by design (see main.tf) so a newer DLAMI can never replace
# a box mid-run. This is the explicit opt-in.
replace:
	$(TF) apply -replace='aws_instance.gpu[0]'

destroy:
	$(TF) destroy
