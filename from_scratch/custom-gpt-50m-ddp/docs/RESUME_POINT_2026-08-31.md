# Resume point — 2026-08-31 night, stopped mid-prep for tomorrow's 2-node AWS run

Written because the user is starting a **new session tomorrow** with none of
today's conversation history — this doc is the complete handoff. Read this
first, before `DATA_AND_TRAINING_SOP.md` / `MULTI_NODE_DDP.md` (which cover the
general how-to; this covers *exactly what state things are in right now*).

## What's actually done

1. **A shared, cross-project data-prep tool now exists**: `tools/data-prep/`
   (repo root, sibling to `tools/corpus-extractor/`) — `sources.py` + `prepare.py`
   (moved from `custom-gpt-153m`, made import-path-generic, otherwise unchanged —
   they already took a `paths` object as their only project-specific input) and
   a new `build_pretrain_corpus.py` (generalized replacement for every project's
   own hardcoded `build_pretrain_split.py` — takes `--source`/`--jsonl-source`
   paths and `--out-train`/`--out-test` as CLI args, memory-safe chunk-streaming
   implementation, see its own docstring for why the streaming rewrite was
   necessary — two real OOM kills, exit 137, happened building today's corpus
   before it was fixed).
2. **`custom-gpt-50m-ddp/data/raw` is now a symlink** to `../../_shared_data/raw`
   — matching the established convention already used by `custom-gpt-153m`
   (documented in `from_scratch/models.md`'s "Shared raw data" section, dated
   2026-08-18). Today's earlier from-scratch re-downloads (a second Cosmopedia-v2
   stream, a second 6-source chat download, a second book extraction) were all
   **redundant** with data already sitting in `_shared_data/raw/` — that
   duplicate work has been deleted; nothing was lost.
3. **Post-train corpus rebuilt, real and complete**: `data/profiles/posttrain/`
   — **422,895 conversations** across **all 8** registered chat sources
   (UltraChat, OASST1, Dolly, SmolTalk, No Robots, GSM8K, LMSYS-Chat-1M,
   OpenHermes-2.5 — all already cached in `_shared_data/raw/`, `skip_download=True`
   used, zero new downloads). `train.txt` (1.05 GB) / `test.txt` (117 MB) /
   `test_prompts.txt` (50 held-out prompts for QA). This is meaningfully richer
   than the 6-source, 249,589-conversation version built earlier today — that
   earlier one is superseded, don't use it.
4. **Pretrain corpus rebuilt, real and complete**: `data/profiles/pretrain/` —
   all 5 available raw prose sources (Cosmopedia v1, Cosmopedia v2, finemath-4plus,
   Hindi Wikipedia, open-web-math) **plus real book text** (713 whole-book
   records from `_shared_data/raw/books/dataset.jsonl`, ~476 MB — NOT the
   1024-token-chunked version `BOOKS_CORPUS_INTEGRATION.md` describes building at
   one point; this file on disk today is whole-book records instead. Fine as-is:
   `get_batch()` samples random windows from the concatenated token stream
   regardless of document granularity, so this doesn't need re-chunking before
   use). **`train.txt` is 30,159 MB (~30 GB), `test.txt` is 305 MB** — real doc
   counts printed during the build:
   ```
   HuggingFaceTB/cosmopedia            14,002,822 docs
   HuggingFaceTB/smollm-corpus/v2      52,051,053 docs
   HuggingFaceTB/finemath-4plus        33,409,428 docs
   wikimedia/wikipedia/hi               1,143,378 docs
   open-web-math/open-web-math         16,654,555 docs
   books (whole-book records)                 713 docs
   ```
   **This is far larger than this model needs** (51.48M params, Chinchilla-optimal
   ≈1.03B tokens ≈ a few GB of text, not 30 GB) — that's fine, not wasteful
   (surplus tokens are neutral under random-window sampling per this project
   family's own established finding), but it means **do not try to tokenize or
   consume the whole 30 GB** — see "What's NOT done yet," next.

## What's NOT done yet — the actual next steps, in order

1. **Tokenize both profiles and get real token counts.** Neither profile has
   been tokenized yet against this specific corpus build. Run:
   ```bash
   cd from_scratch/custom-gpt-50m-ddp
   make use-pretrain     # tokenizes data/profiles/pretrain -> data/train.bin/test.bin, prints real token count
   ```
   **Expect this to take a while and use real disk/CPU** — 30 GB of text through
   a BPE tokenizer is not instant. If it's impractically slow given the model
   only needs ~1B tokens anyway, consider truncating `data/profiles/pretrain/train.txt`
   to a smaller prefix (e.g. `head -c <bytes> train.txt > train_capped.txt`,
   sized around 4-6 GB to comfortably cover ~1B+ tokens with headroom) **before**
   tokenizing, rather than tokenizing all 30 GB and only using a fraction of the
   resulting `.bin` — that would waste real time tokenizing text that's never
   read. This wasn't done tonight for lack of time, not because it's wrong to do.
2. **Update `infra/aws-gpu-node-multi/50m-ddp.tfvars`'s `target_tokens`** once
   step 1 gives a real number (currently a `1000000000` placeholder — update the
   comment above it too, which currently says "placeholder below, update after
   `make use-pretrain` reports the exact token count").
3. **Verify `batch_size=16`/`grad_accum_steps=16` on ONE real GPU before trusting
   them** — these are an *unverified starting guess* in `50m-ddp.tfvars`, not a
   measured-safe number the way `custom-gpt-350m-ddp`'s `batch_size=4` was
   (see that project's own OOM saga, `docs/RUN_LOG_2026-08-31.md`, for exactly
   why this matters and how cheap the single-GPU check is). Don't skip this step
   just because 51M params is much smaller than 347M — "much smaller, probably
   fine" was exactly the assumption that turned out wrong once already this week.
4. **Upload corpus to S3** once tokenized — no `make upload-tokenizer` step
   needed for this project (GPT-2's built-in `tiktoken` encoding, no custom
   `tokenizer.json` to sync — already reflected in `50m-ddp.tfvars`'s empty
   `tokenizer_prefix`).
5. **Deploy real instances** — `infra/aws-gpu-node-multi/` is currently sitting
   in the **`50m-ddp` Terraform workspace** (not `default`, which holds
   `custom-gpt-350m-ddp`'s state — confirm with `terraform workspace show`
   before running anything). Plan first:
   ```bash
   cd infra/aws-gpu-node-multi
   terraform workspace select 50m-ddp   # if a new session starts in "default"
   terraform plan  -var-file=50m-ddp.tfvars
   terraform apply -var-file=50m-ddp.tfvars   # real billing starts here — confirm with the user first
   ```
6. **Launch training** following `docs/DATA_AND_TRAINING_SOP.md`'s phase-1
   (pretrain) steps, then `MULTI_NODE_DDP.md` for the actual 2-node `torchrun`
   commands.
7. **QA reports on checkpoints, both phases** — the user explicitly wants
   pretrain-phase and post-train-phase QA reports generated at different
   checkpoints. **Not yet verified**: whether `make test`/`cli/qa_report.py`
   already reads the *active* profile's `test_prompts.txt` correctly for
   whichever phase is currently active (pretrain has no `test_prompts.txt` at
   all — only posttrain's chat data produces held-out prompts, since
   `prompt_from_turns()` needs a conversation structure a prose document doesn't
   have). Check `cli/qa_report.py`/`qa_prompts.py` against this before assuming
   QA reports "just work" identically in both phases — this is a real open
   question, not confirmed either way tonight.

## Known-good facts to build on (don't re-derive these tomorrow)

- `custom-gpt-50m-ddp` genuinely has real DDP support (`DistributedDataParallel`
  in `trainer.py`, `RANK`/`WORLD_SIZE`/`LOCAL_RANK` handling in `cli/train.py`,
  a working `scripts/ddp_smoke_test.py`) — confirmed directly in code, not
  assumed from the project name.
- Model: 51,475,968 params, `context_length=1024, embed_size=512, num_heads=8,
  num_layers=8`, GPT-2 tiktoken vocab (50,257) — classic family, same tokenizer
  as `custom-gpt-10m`/`custom-gpt-153m`.
- `_shared_data/raw/` (repo root's `from_scratch/`, gitignored) is the
  established, documented shared-data convention since 2026-08-18 — check there
  before ever re-fetching anything for any project in this family.
- AWS: `g5.xlarge` (A10G) in `us-east-1c` (`subnet-0ddd28a9cc6a2f624`) is the
  combination confirmed to actually get EC2 capacity on 2026-08-31 — a
  reasonable default to keep reusing, not a guarantee it'll work again.
- `infra/aws-gpu-node-multi/` now serves **both** DDP projects via Terraform
  workspaces (`default` = 350m-ddp, `50m-ddp` = this project) — same module
  code, separate state, zero infra duplication. `50m-ddp.tfvars` is the
  variables file for the second workspace; don't confuse it with the bare
  `terraform.tfvars` (350m-ddp's).
