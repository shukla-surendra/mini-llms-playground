HOST ?= 127.0.0.1
PORT ?= 8010
PROMPT ?= Once upon a time,

.PHONY: sync data train train-resume train-fresh infer serve snapshot list-snapshots restore-snapshot publish clean help

help:
	@echo "Targets:"
	@echo "  make sync          - create/update .venv from pyproject.toml (via uv)"
	@echo "  make data          - download TinyStories, train tokenizer, tokenize"
	@echo "  make train         - train (resumes automatically if a checkpoint exists)"
	@echo "  make train-resume  - explicitly resume from tinystories_gpt_checkpoint_latest.pt"
	@echo "  make train-fresh   - start training from scratch, ignoring any checkpoint"
	@echo "  make infer         - generate a sample from the best checkpoint (PROMPT=... to override)"
	@echo "  make serve         - start the FastAPI server (HOST=... PORT=... to override)"
	@echo "  make snapshot NAME=round2-shakespeare  - archive current checkpoints under checkpoints_archive/"
	@echo "  make list-snapshots                     - list archived snapshots"
	@echo "  make restore-snapshot NAME=...           - restore an archived snapshot as the active checkpoint"
	@echo "  make publish REPO_ID=user/model-name     - upload checkpoint+tokenizer+code to the HF Hub"
	@echo "                                              (reads HF_TOKEN from .env — see .env.example)"
	@echo "  make clean         - remove checkpoints, logs, and .venv"

# uv sync creates/updates .venv from pyproject.toml + uv.lock automatically — every
# other target below runs through `uv run`, which calls this implicitly, so this target
# mainly exists for explicitly (re)provisioning the environment on its own.
sync:
	uv sync

data: sync
	uv run prepare_dataset.py --max-samples $${MAX_SAMPLES:-100000} --vocab-size $${VOCAB_SIZE:-4096}

# Training auto-resumes if tinystories_gpt_checkpoint_latest.pt exists (see docs/TRAINING.md)
train: sync
	uv run train.py

train-resume: sync
	RESUME_TRAINING=1 uv run train.py

train-fresh: sync
	RESUME_TRAINING=0 uv run train.py

infer: sync
	uv run inference.py --prompt "$(PROMPT)"

serve: sync
	uv run uvicorn api_server:app --host $(HOST) --port $(PORT) --reload

# Archive the current checkpoints + eval log before starting a new dataset round, so you
# can roll back if the new round hurts quality more than it helps. See
# docs/CONTINUAL_TRAINING_LOW_RESOURCE.md.
snapshot:
	@if [ -z "$(NAME)" ]; then echo "Usage: make snapshot NAME=round2-shakespeare"; exit 1; fi
	@mkdir -p checkpoints_archive/$(NAME)
	@cp tinystories_gpt_checkpoint.pt checkpoints_archive/$(NAME)/ 2>/dev/null || true
	@cp tinystories_gpt_checkpoint_latest.pt checkpoints_archive/$(NAME)/ 2>/dev/null || true
	@cp -r logs checkpoints_archive/$(NAME)/logs 2>/dev/null || true
	@echo "Snapshotted current checkpoints to checkpoints_archive/$(NAME)/"

list-snapshots:
	@ls -1 checkpoints_archive/ 2>/dev/null || echo "(no snapshots yet)"

restore-snapshot:
	@if [ -z "$(NAME)" ]; then echo "Usage: make restore-snapshot NAME=round2-shakespeare"; exit 1; fi
	@if [ ! -d "checkpoints_archive/$(NAME)" ]; then echo "No snapshot named $(NAME)"; exit 1; fi
	@cp checkpoints_archive/$(NAME)/tinystories_gpt_checkpoint.pt . 2>/dev/null || true
	@cp checkpoints_archive/$(NAME)/tinystories_gpt_checkpoint_latest.pt . 2>/dev/null || true
	@echo "Restored checkpoints from checkpoints_archive/$(NAME)/ as the active checkpoint"

# Reads HF_TOKEN from .env automatically (scripts/upload_to_hf.py loads it via
# python-dotenv) — see .env.example and docs/PUBLISHING_TO_HUGGING_FACE.md.
publish: sync
	@if [ -z "$(REPO_ID)" ]; then echo "Usage: make publish REPO_ID=your-username/tinystories-gpt-6m"; exit 1; fi
	uv run scripts/upload_to_hf.py --repo-id "$(REPO_ID)"

clean:
	rm -f tinystories_gpt_checkpoint*.pt
	rm -rf logs
	rm -rf .venv
