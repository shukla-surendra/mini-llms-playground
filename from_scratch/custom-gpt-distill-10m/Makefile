.PHONY: help setup config distill train train-soft infer eval clean

help:
	@echo "custom-gpt-distill-10m — targets:"
	@echo "  make setup                          # create environment and install deps"
	@echo "  make config                          # inspect architecture, params, corpus/epoch math"
	@echo "  make distill                         # generate an instruction/response batch from a local Ollama teacher"
	@echo "  make distill COUNT=500 MODEL=gemma3:4b  # override batch size / teacher"
	@echo "  make train                           # sequence-level: plain SFT on gpt-distill's corpus (resumes automatically)"
	@echo "  make train-soft                      # soft-label (Hinton et al.): real gpt2-medium as teacher (resumes separately)"
	@echo "  make infer PROMPT='Give me one tip for saving money.'"
	@echo "  make eval                            # score checkpoints on held-out data"
	@echo "  make clean                           # remove checkpoints (NOT the corpus/distilled data)"

setup:
	uv sync

config:
	uv run gpt-config

distill:
	uv run gpt-distill --count $(or $(COUNT),200) --model $(or $(MODEL),gemma3:4b)

train:
	uv run gpt-train

train-soft:
	uv run gpt-train-soft

infer:
	uv run gpt-infer --prompt "$(PROMPT)"

eval:
	uv run gpt-eval

clean:
	rm -rf checkpoints/*.pt
