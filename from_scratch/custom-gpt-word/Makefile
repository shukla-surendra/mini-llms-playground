.PHONY: help setup config dry-run train generate eval clean

help:
	@echo "custom-gpt-word — targets:"
	@echo "  make setup                         # create environment and install PyTorch"
	@echo "  make config                        # inspect vocabulary, shapes, parameters"
	@echo "  make dry-run                       # one forward/backward/update; writes nothing"
	@echo "  make train                         # train and save checkpoints/word-gpt.pt"
	@echo "  make generate PROMPT='the cat'     # continue a prompt after training"
	@echo "  make eval                          # score checkpoints/word-gpt.pt on held-out data"
	@echo "  make clean                         # remove the generated checkpoint"

setup:
	uv sync

config:
	uv run word-gpt-config

dry-run:
	uv run word-gpt-dry-run

train:
	uv run word-gpt-train

generate:
	uv run word-gpt-generate --prompt "$(PROMPT)"

eval:
	uv run word-gpt-eval

clean:
	rm -f checkpoints/word-gpt.pt
