# custom-gpt-distill-10m — a from-scratch GPT trained entirely on distillation

A ~9.98M-parameter decoder-only Transformer, architecturally identical to
[`../custom-gpt-10m`](../custom-gpt-10m/) (context 512, embed 160, 8 heads, 6 layers,
GPT-2 BPE tokenizer) — the only project in this repo with a *measured*, currently
in-progress local training run on this exact MacBook, so it's the lowest-risk size to
build a second local project around.

What's different: every token in this project's training corpus comes from
**sequence-level knowledge distillation** — a locally-hosted open-weight teacher model
(`gemma3:4b` via [Ollama](https://ollama.com)) generates instruction/response text,
which this project then trains a much smaller student model on directly. No
HuggingFace pretraining corpus, no multi-source data pipeline. See
[`../../docs/LLM_AS_JUDGE_AND_DISTILLATION.md`](../../docs/LLM_AS_JUDGE_AND_DISTILLATION.md)
for why this is the only distillation kind viable at this scale (logit distillation
needs a shared vocabulary with the teacher; this project's tokenizer and the teacher's
don't correspond), and for the legal reasoning on which teacher models are safe to
distill from locally (open-weight, not commercial API output).

## Quick start

```bash
cd from_scratch/custom-gpt-distill-10m
make setup

ollama serve &
ollama pull gemma3:4b

make distill                    # generates 200 instruction/response pairs (a few minutes)
make config                     # check corpus size and the resulting tokens/param ratio
make distill COUNT=1000         # run again (and again) to build up the corpus further -
                                 # each run APPENDS, it doesn't overwrite
make train                      # sequence-level: plain SFT on the distilled corpus, resumes automatically
make train-soft                 # soft-label: real gpt2-medium as teacher (downloads ~1.5GB once), resumes separately
make infer PROMPT="Give me one tip for staying focused while studying."
make eval                       # loss/perplexity/accuracy on held-out data
```

## Two distillation mechanisms, both real

- **`make train` (sequence-level)** — plain next-token cross-entropy on `gpt-distill`'s
  generated text. Mechanically identical to ordinary SFT; the only difference is who
  wrote the training labels.
- **`make train-soft` (soft-label, Hinton et al. 2015)** — loads real `gpt2-medium`
  (Modified MIT license, and the *exact* `tiktoken` `gpt2` vocabulary this project
  already uses, verified token-id-for-token-id identical) as a teacher running
  alongside the student, and trains against a blend of the normal hard-label loss and
  a KL-divergence term matching the teacher's full output distribution — see
  [`src/gpt/cli/train_soft.py`](src/gpt/cli/train_soft.py) and
  [`docs/llm-engineering/32_knowledge_distillation_mechanism_by_mechanism.md`](../../docs/llm-engineering/32_knowledge_distillation_mechanism_by_mechanism.md).
  Checkpoints land in `checkpoints/soft/`, kept separate from `make train`'s so neither
  run overwrites the other — compare both with `gpt-eval`.

Worth knowing honestly: this is a ~35x student/teacher parameter ratio (10M vs
gpt2-medium's 355M) — well past the 1.5x-15x range real named distillations
(DistilBERT, TinyBERT, MobileBERT) actually use. Chapter 32's "capacity gap" section
covers why that ratio is a real risk, not just a footnote — this project runs it anyway
as an honest demonstration of the mechanism, not a claim that 35x is the right target.

## Why 200 pairs isn't the target, and what is

A 10M-param model wants roughly 200M training tokens to sit near Chinchilla-optimal
(20 tokens/param) — nowhere close to what a single `make distill` batch produces (a
few hundred short instruction/response pairs is more like 20-40K tokens). Two things
make this workable anyway:

1. **Training re-uses the corpus across many epochs** — `gpt-train`'s `TrainConfig`
   defaults assume this; `make config` prints the exact epoch count for the current
   corpus size and step budget, so it's never a hidden assumption.
2. **`make distill` is designed to be run repeatedly**, appending each time. Roughly
   150-250K unique generated tokens (reachable in a couple of hours of local
   generation, per this repo's own measured Ollama throughput of ~25-45 tok/s) lands
   close to a genuine 20 tokens/param ratio without ever needing hundreds of epochs
   over a tiny handful of examples.

## Read it in this order

1. [`src/gpt/config.py`](src/gpt/config.py) — architecture + training knobs, all in one place.
2. [`src/gpt/model.py`](src/gpt/model.py) — the Transformer (SDPA attention, weight-tied output).
3. [`src/gpt/cli/distill.py`](src/gpt/cli/distill.py) — where the training data actually comes from.
4. [`src/gpt/data/dataset.py`](src/gpt/data/dataset.py) — corpus text → token id batches.
5. [`src/gpt/cli/train.py`](src/gpt/cli/train.py) — the training loop (warmup + cosine LR, gradient accumulation, atomic checkpointing).
6. [`src/gpt/cli/infer.py`](src/gpt/cli/infer.py) / [`evaluate.py`](src/gpt/cli/evaluate.py) — generation and held-out scoring.

## Scope: a right-sized subset, not full parity with the other `from_scratch` projects

This project deliberately skips what `custom-gpt-10m`/`50m`/`153m` have that a
distillation-only project doesn't need yet: a FastAPI serving layer (`gpt-serve`),
multi-source data auditing (`gpt-audit`), and LLM-as-judge QA reporting
(`gpt-qa-report`) — all built for a much larger, multi-source pretraining corpus this
project doesn't have. Add them later if this project's scope grows to need them; see
those projects' own `src/gpt/cli/` for the reference implementations.
