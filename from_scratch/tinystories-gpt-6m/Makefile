PY ?= .venv/bin/python
HOST ?= 127.0.0.1
PORT ?= 8010
PROMPT ?= Once upon a time,

.PHONY: venv install data train train-resume train-fresh infer serve clean help

help:
	@echo "Targets:"
	@echo "  make venv          - create .venv"
	@echo "  make install       - install requirements.txt into .venv"
	@echo "  make data          - download TinyStories, train tokenizer, tokenize"
	@echo "  make train         - train (resumes automatically if a checkpoint exists)"
	@echo "  make train-resume  - explicitly resume from tinystories_gpt_checkpoint_latest.pt"
	@echo "  make train-fresh   - start training from scratch, ignoring any checkpoint"
	@echo "  make infer         - generate a sample from the best checkpoint (PROMPT=... to override)"
	@echo "  make serve         - start the FastAPI server (HOST=... PORT=... to override)"
	@echo "  make clean         - remove checkpoints, logs, and venv"

venv:
	python3 -m venv .venv

install: venv
	$(PY) -m pip install --upgrade pip -q
	$(PY) -m pip install -r requirements.txt -q

data: install
	$(PY) prepare_dataset.py --max-samples $${MAX_SAMPLES:-100000} --vocab-size $${VOCAB_SIZE:-4096}

# Training auto-resumes if tinystories_gpt_checkpoint_latest.pt exists (see docs/TRAINING.md)
train: install
	$(PY) train.py

train-resume: install
	RESUME_TRAINING=1 $(PY) train.py

train-fresh: install
	RESUME_TRAINING=0 $(PY) train.py

infer: install
	$(PY) inference.py --prompt "$(PROMPT)"

serve: install
	$(PY) -m uvicorn api_server:app --host $(HOST) --port $(PORT) --reload

clean:
	rm -f tinystories_gpt_checkpoint*.pt
	rm -rf logs
	rm -rf .venv
