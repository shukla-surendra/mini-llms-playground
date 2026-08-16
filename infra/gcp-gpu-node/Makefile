TF       ?= terraform
PROJECT_ID ?= $(shell $(TF) output -raw project_id 2>/dev/null)
ZONE     ?= $(shell $(TF) output -raw zone 2>/dev/null)
NAME     ?= $(shell $(TF) output -raw instance_name 2>/dev/null)
IP       ?= $(shell $(TF) output -raw public_ip 2>/dev/null)
BUCKET   ?= $(shell $(TF) output -raw bucket 2>/dev/null)
KEY      ?= ~/.ssh/id_ed25519
SSH_USER ?= gpu
TYPE     ?= g2-standard-4

# Derived from terraform.tfvars via `terraform output`, so PROJECT_DIR/PREFIX can
# never drift from what the instance was actually bootstrapped to sync — override
# any of the three on the command line (e.g. `make upload-corpus PROJECT_DIR=...`)
# only for a one-off that intentionally differs from the configured module.
SUBDIR        ?= $(shell $(TF) output -raw project_subdir 2>/dev/null)
PROJECT_DIR   ?= ../../$(SUBDIR)
PREFIX        ?= $(shell $(TF) output -raw corpus_prefix 2>/dev/null)
CKPT_PREFIX   ?= $(shell $(TF) output -raw checkpoint_prefix 2>/dev/null)
REPO_DIR_NAME ?= $(shell $(TF) output -raw repo_dir_name 2>/dev/null)
REMOTE_DIR    ?= ~/$(REPO_DIR_NAME)/$(SUBDIR)

.PHONY: help init fmt validate plan apply ssh iap-ssh tunnel bootstrap-log gpu \
        upload-corpus upload-checkpoint download-checkpoints stop start status \
        replace destroy up down spot-price sync-log

help:
	@echo "Setup"
	@echo "  make init                 - terraform init"
	@echo "  make plan                 - what would change"
	@echo "  make apply                - create/update the GPU node (BILLING STARTS)"
	@echo ""
	@echo "Data (run these BEFORE apply — tokenizing on a GPU is money for nothing)"
	@echo "  make upload-corpus        - sync *.bin + *.bin.json from PROJECT_DIR to gs://$(BUCKET)/$(PREFIX)"
	@echo "  make upload-checkpoint    - sync an existing run's checkpoints/ up, for gpt-train to auto-resume"
	@echo "  make download-checkpoints - pull checkpoints/ back down from GCS"
	@echo ""
	@echo "Use"
	@echo "  make ssh                  - ssh in directly (needs allowed_ssh_cidrs to include your IP)"
	@echo "  make iap-ssh              - ssh via Identity-Aware Proxy (no open port, no IP dependency)"
	@echo "  make tunnel               - forward the model API to localhost:8000 (via IAP)"
	@echo "  make bootstrap-log        - tail the startup-script bootstrap output"
	@echo "  make gpu                  - nvidia-smi + bf16 check on the box"
	@echo "  make status               - instance state + type + public ip"
	@echo ""
	@echo "  make sync-log             - tail the periodic checkpoint-sync journal"
	@echo ""
	@echo "Money  (cheapest usable pattern: check pricing -> up -> ... -> down)"
	@echo "  make down                 - DESTROY the instance, keep bucket/IAM/firewall. Bill -> ~\$$0"
	@echo "  make up                   - recreate the instance when you next need a GPU"
	@echo "  make stop                 - pause instead: ends compute billing, still ~\$$8/mo for the disk"
	@echo "  make start                - resume a stopped instance (new public IP)"
	@echo "  make replace              - rebuild the instance on the current image"
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

# Direct SSH needs allowed_ssh_cidrs to include your current IP (re-apply after your
# ISP rotates it) — iap-ssh below has no such dependency.
ssh:
	ssh -i $(KEY) $(SSH_USER)@$(IP)

# No inbound firewall rule, no IP dependency — works after your home IP rotates.
# Requires roles/iap.tunnelResourceAccessor on your gcloud identity (see iam.tf).
iap-ssh:
	gcloud compute ssh $(SSH_USER)@$(NAME) --zone=$(ZONE) --project=$(PROJECT_ID) --tunnel-through-iap

# The API stays closed in the firewall; this is the way in that needs no rule.
tunnel:
	gcloud compute ssh $(SSH_USER)@$(NAME) --zone=$(ZONE) --project=$(PROJECT_ID) --tunnel-through-iap -- -N -L 8000:localhost:8000

bootstrap-log:
	ssh -i $(KEY) $(SSH_USER)@$(IP) 'tail -f /var/log/gpu-node-bootstrap.log'

# torch lives in the project's uv-managed venv (created by `uv sync` at boot), not
# system Python — bare `python3 -c "import torch"` would fail with ModuleNotFoundError
# regardless of whether the GPU itself is fine. `uv run` picks up that venv correctly.
gpu:
	ssh -i $(KEY) $(SSH_USER)@$(IP) 'nvidia-smi; cd $(REMOTE_DIR) && uv run python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"'

upload-corpus:
	cd $(PROJECT_DIR) && gcloud storage rsync data/ gs://$(BUCKET)/$(PREFIX) \
		--recursive --exclude=".*" --include-file-pattern="*.bin,*.bin.json" 2>/dev/null \
		|| gcloud storage cp data/*.bin data/*.bin.json gs://$(BUCKET)/$(PREFIX)

# Makes an existing local run resumable on the instance: gpt-train auto-resumes from
# checkpoints/<label>/latest.pt with no flag once this lands in the right place at
# boot (checkpoint_prefix in terraform.tfvars must be set to match).
upload-checkpoint:
	cd $(PROJECT_DIR) && gcloud storage rsync checkpoints/ gs://$(BUCKET)/$(CKPT_PREFIX) \
		--recursive --exclude=".*\.DS_Store$$"

download-checkpoints:
	cd $(PROJECT_DIR) && gcloud storage rsync gs://$(BUCKET)/$(CKPT_PREFIX) checkpoints/ --recursive

sync-log:
	ssh -i $(KEY) $(SSH_USER)@$(IP) 'journalctl -u checkpoint-sync -u preempt-watch -f'

# The between-runs pattern. Destroying the instance drops the disk bill to zero; the
# bucket, service account, and firewall rules cost nothing to keep, so the next
# `make up` is a few-minute bootstrap rather than a re-setup.
# Sync your checkpoints first — this deletes the boot disk.
down:
	$(TF) apply -var instance_count=0

up:
	$(TF) apply -var instance_count=1

stop:
	gcloud compute instances stop $(NAME) --zone=$(ZONE) --project=$(PROJECT_ID)

start:
	gcloud compute instances start $(NAME) --zone=$(ZONE) --project=$(PROJECT_ID)
	@echo "public IP changes on start — run 'terraform refresh && make status'"

status:
	gcloud compute instances describe $(NAME) --zone=$(ZONE) --project=$(PROJECT_ID) \
		--format='table(status,machineType.basename(),networkInterfaces[0].accessConfigs[0].natIP,zone.basename())'

# image is ignored by design (see main.tf) so a newer published image can never
# replace a box mid-run. This is the explicit opt-in.
replace:
	$(TF) apply -replace='google_compute_instance.gpu[0]'

destroy:
	$(TF) destroy
