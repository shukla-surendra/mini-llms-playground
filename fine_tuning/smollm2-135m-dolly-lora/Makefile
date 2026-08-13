MAX_SAMPLES ?= 4000
EPOCHS ?= 3

.PHONY: train train-resume compare clean help

help:
	@echo "Targets:"
	@echo "  make train          - LoRA fine-tune SmolLM2-135M on Dolly-15k (MAX_SAMPLES=... EPOCHS=... to override)"
	@echo "  make train-resume   - resume from the last saved checkpoint"
	@echo "  make compare        - generate from BOTH base and fine-tuned model, side by side"
	@echo "  make clean          - remove trained outputs"

train:
	uv run train_lora.py --max-samples $(MAX_SAMPLES) --epochs $(EPOCHS)

train-resume:
	uv run train_lora.py --max-samples $(MAX_SAMPLES) --epochs $(EPOCHS) --resume

compare:
	uv run compare_before_after.py

clean:
	rm -rf outputs
