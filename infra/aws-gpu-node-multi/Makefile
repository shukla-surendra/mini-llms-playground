TF        ?= terraform
REGION    ?= $(shell $(TF) output -raw region 2>/dev/null || echo us-east-1)
MASTER_IID ?= $(shell $(TF) output -raw master_instance_id 2>/dev/null)
WORKER_IID ?= $(shell $(TF) output -raw worker_instance_id 2>/dev/null)
MASTER_IP  ?= $(shell $(TF) output -raw master_public_ip 2>/dev/null)
WORKER_IP  ?= $(shell $(TF) output -raw worker_public_ip 2>/dev/null)
BUCKET     ?= $(shell $(TF) output -raw bucket 2>/dev/null)
KEY        ?= ~/.ssh/id_ed25519
TYPE       ?= g6.xlarge

# Derived from terraform.tfvars via `terraform output`, so PROJECT_DIR/PREFIX can
# never drift from what both instances were actually bootstrapped to sync.
SUBDIR         ?= $(shell $(TF) output -raw project_subdir 2>/dev/null)
PROJECT_DIR    ?= ../../$(SUBDIR)
CORPUS_PREFIX  ?= $(shell $(TF) output -raw corpus_prefix 2>/dev/null)
TOKENIZER_PREFIX ?= $(shell $(TF) output -raw tokenizer_prefix 2>/dev/null)
CKPT_PREFIX    ?= $(shell $(TF) output -raw checkpoint_prefix 2>/dev/null)
REPO_DIR_NAME  ?= $(shell $(TF) output -raw repo_dir_name 2>/dev/null)
REMOTE_DIR     ?= ~/$(REPO_DIR_NAME)/$(SUBDIR)

.PHONY: help init fmt validate plan apply \
        ssh-master ssh-worker ssm-master ssm-worker tunnel \
        bootstrap-log-master bootstrap-log-worker gpu-master gpu-worker \
        upload-corpus upload-tokenizer upload-checkpoint download-checkpoints \
        launch-reminder stop start status destroy up down spot-price sync-log

help:
	@echo "Setup"
	@echo "  make init                 - terraform init"
	@echo "  make plan                 - what would change"
	@echo "  make apply                - create/update BOTH GPU nodes (BILLING STARTS ON BOTH)"
	@echo ""
	@echo "Data (run these BEFORE apply — tokenizing on two GPUs is money for nothing, twice)"
	@echo "  make upload-corpus        - sync *.bin from PROJECT_DIR to s3://$(BUCKET)/$(CORPUS_PREFIX)"
	@echo "  make upload-tokenizer     - sync tokenizer.json from PROJECT_DIR to s3://$(BUCKET)/$(TOKENIZER_PREFIX)"
	@echo "  make upload-checkpoint    - sync an existing run's checkpoints/ up, for gpt-train to auto-resume"
	@echo "  make download-checkpoints - pull checkpoints/ back down from S3 (master only ever wrote them)"
	@echo ""
	@echo "Use"
	@echo "  make ssh-master / ssh-worker   - ssh into either node"
	@echo "  make ssm-master / ssm-worker   - Session Manager shell into either node"
	@echo "  make bootstrap-log-master/-worker - tail cloud-init bootstrap output on either node"
	@echo "  make gpu-master / gpu-worker   - nvidia-smi + bf16 check on either node"
	@echo "  make launch-reminder           - print the exact 2-terminal start sequence"
	@echo "  make status                    - both instances' state + type + public ip"
	@echo ""
	@echo "  make sync-log             - tail the master's periodic checkpoint-sync journal"
	@echo ""
	@echo "Money  (cheapest usable pattern: make spot-price -> up -> ... -> down)"
	@echo "  make spot-price           - live spot price for $(TYPE) per AZ, before you commit"
	@echo "  make down                 - DESTROY both instances, keep bucket/IAM/SG. Bill -> ~\$$0"
	@echo "  make up                   - recreate both instances when you next need the pair"
	@echo "  make stop                 - pause both instead: ends compute billing, still ~\$$16/mo for both EBS volumes"
	@echo "  make start                - resume both stopped instances (new public IPs)"
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

ssh-master:
	ssh -i $(KEY) ubuntu@$(MASTER_IP)

ssh-worker:
	ssh -i $(KEY) ubuntu@$(WORKER_IP)

ssm-master:
	aws ssm start-session --region $(REGION) --target $(MASTER_IID)

ssm-worker:
	aws ssm start-session --region $(REGION) --target $(WORKER_IID)

bootstrap-log-master:
	ssh -i $(KEY) ubuntu@$(MASTER_IP) 'tail -f /var/log/gpu-node-bootstrap.log'

bootstrap-log-worker:
	ssh -i $(KEY) ubuntu@$(WORKER_IP) 'tail -f /var/log/gpu-node-bootstrap.log'

# torch lives in the project's uv-managed venv (created by `uv sync` at boot), not
# system Python.
gpu-master:
	ssh -i $(KEY) ubuntu@$(MASTER_IP) 'nvidia-smi; cd $(REMOTE_DIR) && uv run python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"'

gpu-worker:
	ssh -i $(KEY) ubuntu@$(WORKER_IP) 'nvidia-smi; cd $(REMOTE_DIR) && uv run python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"'

launch-reminder:
	@$(TF) output -raw launch_reminder; echo

upload-corpus:
	cd $(PROJECT_DIR) && aws s3 sync data/ s3://$(BUCKET)/$(CORPUS_PREFIX) \
		--exclude "*" --include "*.bin"

upload-tokenizer:
	cd $(PROJECT_DIR) && aws s3 sync tokenizer/ s3://$(BUCKET)/$(TOKENIZER_PREFIX) \
		--exclude "*" --include "tokenizer.json"

# Makes an existing local run resumable on both nodes: gpt-train auto-resumes from
# checkpoints/<label>/latest.pt with no flag once this lands at boot (checkpoint_prefix
# in terraform.tfvars must be set to match).
upload-checkpoint:
	cd $(PROJECT_DIR) && aws s3 sync checkpoints/ s3://$(BUCKET)/$(CKPT_PREFIX) \
		--exclude "*.DS_Store"

download-checkpoints:
	cd $(PROJECT_DIR) && aws s3 sync s3://$(BUCKET)/$(CKPT_PREFIX) checkpoints/

sync-log:
	ssh -i $(KEY) ubuntu@$(MASTER_IP) 'journalctl -u checkpoint-sync -u spot-watch -f'

# What the spot market is actually charging right now, per AZ. Read this before
# `make up` if use_spot = true — see variables.tf for the multi-node Spot caveat
# (a reclaim of EITHER node kills the whole synchronized job).
spot-price:
	aws ec2 describe-spot-price-history --region $(REGION) \
		--instance-types $(TYPE) --product-descriptions "Linux/UNIX" \
		--start-time $$(date -u +%Y-%m-%dT%H:%M:%S) \
		--query 'SpotPriceHistory[].{AZ:AvailabilityZone,USD_per_hour:SpotPrice}' \
		--output table

# The between-runs pattern. Destroying both instances drops the EBS bill to zero;
# the bucket, IAM role, security group and key pair cost nothing to keep.
# Sync your checkpoints first — this deletes both root volumes.
down:
	$(TF) apply -var instance_count=0

up:
	$(TF) apply -var instance_count=2

stop:
	aws ec2 stop-instances --region $(REGION) --instance-ids $(MASTER_IID) $(WORKER_IID)

start:
	aws ec2 start-instances --region $(REGION) --instance-ids $(MASTER_IID) $(WORKER_IID)
	@echo "public IPs change on start — run 'terraform refresh && make status'"
	@echo "NOTE: the master's PRIVATE ip is static (see network.tf) and does not change, so launch_ddp.sh on both nodes stays correct across a stop/start."

status:
	aws ec2 describe-instances --region $(REGION) --instance-ids $(MASTER_IID) $(WORKER_IID) \
		--query 'Reservations[].Instances[].{Name:Tags[?Key==`Name`]|[0].Value,State:State.Name,Type:InstanceType,IP:PublicIpAddress,AZ:Placement.AvailabilityZone}' \
		--output table

destroy:
	$(TF) destroy
