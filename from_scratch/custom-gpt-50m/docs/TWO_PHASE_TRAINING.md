# Two-phase training: pretrain, then post-train

Added 2026-08-17 when restarting this project's training from step 0. The single-phase
run before this (chat conversations + books/repos/Wikipedia mixed into one corpus,
trained together from the start — see `archive/50m_run1_step913k_pre-153m-restart_2026-08-17/`)
plateaued through 91% of its planned 1,000,000 steps without reaching the quality bar.
`docs/TRAINING_QA.md`'s "Does arranging data in a particular way increase model
performance?" entry had already flagged the likely cause: extra (non-chat) documents
outnumbered chat conversations in that corpus, diluting chat-following behavior every
step, from the very first step, for the entire run.

This doc describes the alternative now available: separate the "learn language and
facts" objective from the "learn to hold a conversation" objective into two sequential
phases, instead of training on both at once.

## The two corpora

`data/profiles/<name>/{train,test}.txt` — one directory per corpus variant. Only one is
ever "active" (copied into the canonical `data/train.txt`/`test.txt` that `gpt-train`
actually reads) at a time.

| Profile | Built by | Content | Size |
|---|---|---|---|
| `pretrain` | `make data-pretrain` (`scripts/build_pretrain_split.py`) | Raw prose — cosmopedia-v1/v2 (synthetic textbooks), finemath-4plus (math reasoning), Hindi Wikipedia, open-web-math. Copied from the sibling `custom-gpt-153m` project's enriched raw pool, ~2.85B tokens. No chat formatting, no conversations. | ~11GB text |
| `posttrain` | `make data-posttrain` (`gpt-data --skip-download`, no `--extra-jsonl`) | Pure chat — the same 7 registered conversational sources as before (UltraChat, OASST1, Dolly, SmolTalk, No Robots, GSM8K, LMSYS if a token is set), **with the books/repos/Wikipedia extra documents left out** this time. 249,589 conversations. | ~800MB text |

Switch which one is active:

```
make use-pretrain     # copies data/profiles/pretrain/{train,test}.txt -> canonical, re-tokenizes
make use-posttrain    # copies data/profiles/posttrain/{train,test}.txt -> canonical, re-tokenizes
```

## Two ways to train from here

### A. Two-phase (pretrain, then post-train) — recommended for this restart

```
make use-pretrain
make train-fresh                      # phase 1: base language model, from step 0
# ... let it run to a good stopping point, check QA reports ...
cp checkpoints/50m/*.pt archive/<snapshot-name>/checkpoints/   # snapshot before phase 2
make use-posttrain
GPT_STEPS=<phase1_final_step + N> make train   # phase 2: resumes phase-1 weights,
                                                # continues training on chat data only
```

Why this works with zero code changes (verified in `trainer.py` before writing this):
checkpoint resume (`_resume_into`) only checks architecture compatibility
(`embed_size`/`num_layers`/`context_length`), not which corpus produced the checkpoint —
so swapping the active profile between phases and re-running `gpt-train` (the default
`resume=True`) picks up phase 1's weights and optimizer state and simply keeps training,
now on the new data. The LR schedule's cosine decay clamps at `min_lr` past its
configured `steps` ceiling, so phase 2 naturally continues at a low, fine-tuning-style
learning rate rather than crashing or zeroing out.

**The one thing you must set explicitly**: `GPT_STEPS` (default 1,000,000). The training
loop is `for step in range(start_step, train_cfg.steps)` — if phase 2's `start_step`
(phase 1's final step + 1) is already ≥ `train_cfg.steps`, the loop is empty and phase 2
silently does nothing. Always raise `GPT_STEPS` above wherever phase 1 stopped before
starting phase 2.

**What this is not**: true instruction-tuning with per-turn loss masking. This project
has no chat-template/masking machinery (`DATASET.md`: "raw next-token prediction over
the whole stream ... no chat template, no per-turn loss masking"). Phase 2 is "keep
training the base model, now on chat-formatted text, at a low LR" — a lightweight
approximation of SFT, useful but not equivalent rigor to real masked instruction-tuning.

### B. Single-phase on one profile only

Just `make use-pretrain` (or `use-posttrain`) once and `make train-fresh` — train the
whole budget on that one corpus, no phase switch. This is how the retired run trained
too, just on a cleaner (unmixed) corpus this time if you pick `posttrain` alone, or on
pure base-model prose if you pick `pretrain` alone.

## Recommendation for this restart

Start with **A (two-phase), phase 1 on `pretrain`**. Reasoning:

- The retired run's own diagnosis was dilution from mixing objectives in one pass — two
  separate phases removes that mechanism entirely rather than tuning around it.
- `pretrain`'s corpus (~2.85B tokens) is ~2.3x this project's own 16x-Chinchilla target
  for 51M params (~0.82B) — comfortable headroom, not wasteful (`TRAINING_QA.md`'s
  "excess data" entry: surplus tokens are neutral under this project's random-window
  sampling, not harmful).
- This mirrors how GPT/Llama-style models are actually built (pretrain broad language
  ability, then instruction-tune on top) — the *right* two-phase order, not the reverse.
- Phase 1 alone (no phase 2 yet) already gives a real, checkpointable, QA-reportable
  result to evaluate before committing to a specific phase-2 step budget — decide
  `GPT_STEPS` for phase 2 from what phase 1's QA reports actually show, not in advance.

Don't start with **B on `posttrain` alone** (pure chat, no pretraining) — 249,589
conversations (~250-300M tokens including the extra docs stripped out) is far short of
even the 16x floor for a from-scratch 51M-param model; the earlier mixed run only
reached its scale by padding with books/Wikipedia, which is exactly the padding this
restart is trying to get away from.
