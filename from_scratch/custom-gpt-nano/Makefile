.PHONY: help setup config train generate clean

help:
	@echo "custom-gpt-nano — targets:"
	@echo ""
	@echo "  make setup                        # uv sync: create .venv, install torch"
	@echo "  make config                       # show model shape + parameter breakdown"
	@echo "  make train                        # train on data/corpus.txt, save checkpoints/nano.pt"
	@echo "  make generate PROMPT=\"The cat\"     # generate text from the trained checkpoint"
	@echo "  make clean                        # remove the checkpoint (start over)"

setup:
	uv sync

config:
	uv run nano-config

train:
	uv run nano-train

generate:
	uv run nano-generate --prompt "$(PROMPT)"

clean:
	rm -f checkpoints/nano.pt
