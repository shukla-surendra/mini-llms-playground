# Data & Training SOP — pretrain → post-train, for a model that can hold a conversation

Written 2026-08-31, right after building both corpora fresh (the sibling
`custom-gpt-153m` project's raw pool this project's pretrain data was originally
meant to reuse turned out to be gone — see "Why this was rebuilt from scratch"
below). Self-contained: everything needed to go from the two corpora already on
disk to a chat-capable checkpoint, without needing this session's history.

## Which data is which, exactly

| | Pretrain | Post-train |
|---|---|---|
| **Location** | `data/profiles/pretrain/{train,test}.txt` | `data/profiles/posttrain/{train,test}.txt` |
| **Content** | Cosmopedia-v2 (synthetic textbook-quality prose, streamed fresh from `HuggingFaceTB/smollm-corpus`) + your own 983 real book PDFs (`~/Downloads/books`, extracted via `tools/corpus-extractor`) | 249,589 real conversations across 6 sources: UltraChat (100k), SmolTalk (100k), No Robots (17,110), GSM8K (16,860), Dolly (12,173), OASST1 (3,446) |
| **Real token count** | **~1.135B** (988M Cosmopedia + 135,201,492 books, exact) | ~250-300M (not yet exactly tokenized — `make use-posttrain` will report the real number) |
| **Purpose** | Teach basic grammar, coherent prose, general world knowledge/reasoning — "learn language" | Teach it to actually hold a dialogue / follow an instruction — "learn to talk." **No prose is mixed in here on purpose** — this project's own `docs/TWO_PHASE_TRAINING.md` already diagnosed diluted chat-following as the reason an earlier mixed-corpus run underperformed. |
| **Gated sources skipped** | none | `lmsys/lmsys-chat-1m` (needs an accepted-terms HF token — not configured) |

This project's own 51.48M-param default architecture (`ModelConfig()`, no preset
needed) has a Chinchilla-optimal target of **~1.03B tokens** — the pretrain corpus
(~1.135B) sits almost exactly there, real data, not padded to hit a number.

## Why this was rebuilt from scratch (skip if you already know)

`data-pretrain`'s original design (`scripts/build_pretrain_split.py`) expected raw
source files copied from `custom-gpt-153m`'s own `data/raw/` — that directory (and
`custom-gpt-153m`'s own tokenized corpus) no longer exists on disk, a broader
version of a data-loss incident from earlier the same session. The books
themselves (`~/Downloads/books`, outside any project's `data/` dir) and the
extraction tool survived untouched — only project-internal `data/` directories
were affected. Rebuilt by streaming Cosmopedia-v2 directly (`scripts/
fetch_pretrain_corpus.py`, copied over from `custom-gpt-153m`) instead of relying
on a copy, and re-running `tools/corpus-extractor` fresh against the real books.

## Step-by-step: phase 1 (pretrain)

```bash
cd from_scratch/custom-gpt-50m-ddp
make setup                    # uv sync, one-time
make use-pretrain              # copies pretrain profile to canonical data/train.txt+test.txt,
                                # tokenizes for real (exact token counts print here)
```

Raise `batch_size` off its laptop/MPS default (`1`) before a real GPU run — it
exists specifically for MPS VRAM limits, not something a rented GPU needs:

```bash
GPT_BATCH_SIZE=32 GPT_GRAD_ACCUM=8 GPT_STEPS=<see below> make train-fresh
```

**Setting `GPT_STEPS` for phase 1**: `steps × batch_size × context_length` = total
tokens processed (this project's `context_length` default — check
`uv run gpt-config` for the exact active number). Target somewhere in the
~0.8B-1.1B range (16x floor to ~Chinchilla-optimal) — e.g. at
`batch_size=32, context_length=1024`: `steps = tokens_target / (32 × 1024)`.
Run `uv run gpt-config` first to see the exact numbers for whatever
`batch_size`/`context_length` you actually choose before committing to a `steps`
value — don't hand-copy this doc's arithmetic without checking it against the
active config.

**Monitor**: `make train-status` (if backgrounded via `make train-bg`) or watch
the foreground output directly. `logs/train_eval_history_<label>.csv` has every
eval's `train_loss`/`test_loss`/`test_perplexity`.

**Decide when phase 1 is "done"** using this project's own documented framework
(`docs/TRAINING_SCHEDULE.md`'s "Is a longer run still worth it?" three-question
check) — not a fixed step count picked in advance. Then:

```bash
make test          # runs the curated QA prompt set, renders an HTML report — read it
cp checkpoints/50m/*.pt archive/pretrain-snapshot-$(date +%Y%m%d)/checkpoints/  # snapshot before phase 2
```

## Step-by-step: phase 2 (post-train — this is what makes it "talk")

```bash
make use-posttrain              # switches active corpus, re-tokenizes — real token count prints here
GPT_STEPS=<phase1_final_step + N> GPT_BATCH_SIZE=32 GPT_GRAD_ACCUM=8 make train
```

**Do not use `train-fresh` here** — plain `make train` (default `resume=True`)
is what picks up phase 1's weights and optimizer state and keeps going, now on
chat data, per `_resume_into`'s architecture-only compatibility check (verified
in `trainer.py` before this doc was written — resume never checks which corpus
produced a checkpoint, only `embed_size`/`num_layers`/`context_length`).

**The one mistake that silently does nothing**: `GPT_STEPS` must be set *higher*
than wherever phase 1 stopped. The training loop is
`for step in range(start_step, train_cfg.steps)` — if phase 2's `GPT_STEPS`
isn't raised past phase 1's final step, the loop is empty and nothing trains, no
error printed.

Pick phase 2's length from the post-train corpus's own real size once
`make use-posttrain` reports it (aim for roughly 2-4 epochs over the chat
corpus — enough to actually shift behavior toward following turns, not so much
it forgets what phase 1 taught. `docs/DATASET.md`'s repetition-tolerance
guidance applies here too, same as the sibling projects).

**Verify it actually learned to talk**, once phase 2 finishes:

```bash
make infer                    # generate from the checkpoint directly
make serve                     # FastAPI server on :8000 — talk to it for real
make test                      # QA report again — compare against the phase-1 snapshot
```

## Running this under DDP (2 GPUs) instead of one

> For 3+ nodes, or the network/security-group requirements and provisioning
> options for real multi-machine hardware, see
> [`MULTI_NODE_DDP.md`](MULTI_NODE_DDP.md) — this section only covers the
> 2-node case inline.

This project's `trainer.py` already has real DDP support (`scripts/
ddp_smoke_test.py` is the CPU/gloo mechanism check — run it first, same principle
as `custom-gpt-350m-ddp`'s own smoke test, before trusting a real multi-node
launch). `make train`/`train-fresh` don't wrap `torchrun` themselves — launch it
directly, same pattern verified working in the sibling `-ddp` project:

```bash
# Single machine, 2 GPUs:
torchrun --nproc_per_node=2 -m gpt.cli.train

# 2 separate machines, 1 GPU each:
# master:
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=<master's private IP> --master_port=29500 -m gpt.cli.train
# worker:
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=<same master private IP> --master_port=29500 -m gpt.cli.train
```

`GPT_STEPS` is **not** world-size-aware in this project (unlike
`custom-gpt-350m-ddp`'s `GPT_TARGET_TOKENS` mechanism) — running the same `steps`
under `--nproc_per_node=2` doubles total tokens consumed, since each rank
processes its own batch independently. Halve `GPT_STEPS` by hand for the same
total-token budget across 2 ranks, or read `train()`'s own startup "Budget" print
(reports the actual world-size-scaled total) before committing GPU-hours.

At 51.48M params and a real GPU, given how much smaller this model is than the
`custom-gpt-350m-ddp` run that hit CUDA OOM at `batch_size=16`, `batch_size=32`
here is a reasonable starting point, not something to assume is safe — verify
memory headroom on **one** GPU first with a short real run before committing to
a 2-node launch, same lesson learned the hard way on that project (see its
`docs/RUN_LOG_2026-08-31.md`).

## Quick reference

| Command | Effect |
|---|---|
| `make use-pretrain` | Activate the pretrain corpus, tokenize, print real token count |
| `make use-posttrain` | Activate the post-train corpus, tokenize, print real token count |
| `make train-fresh` | Train from step 0, ignoring any checkpoint |
| `make train` | Train, auto-resuming from the latest checkpoint — this is how phase 2 continues phase 1 |
| `make train-stop` | Graceful stop (SIGINT) of a backgrounded run — saves `latest.pt` first |
| `make test` | Curated QA prompt set → HTML report |
| `make infer` / `make serve` | Talk to the checkpoint directly, or over a local API |
