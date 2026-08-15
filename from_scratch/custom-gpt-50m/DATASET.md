# Dataset

**This project (`custom-gpt-50m`, 51,475,968 params, `context_length=1024`) trains on
the exact same corpus as the sibling [`custom-gpt-10m`](../custom-gpt-10m/DATASET.md)
project** — `data/` here is a symlink to `custom-gpt-10m/data`, not a separate copy.
Everything below describes the real corpus currently on disk, not what a fresh `make
data` would produce from the registry defaults alone — a lot of what's here was added
after the registry's original 5-source design (repos, books, Wikipedia, GSM8K — see
"How this corpus actually got built" below), and three real parsing bugs were found and
fixed along the way. Full narrative, in the order it happened:
[`docs/BOOKS_CORPUS_INTEGRATION.md`](docs/BOOKS_CORPUS_INTEGRATION.md).

## At a glance

| | |
|---|---|
| Chat sources | 7 registered (6 real, 1 correctly gated-and-excluded) — see table below |
| Extra (non-chat) documents | Books, 3 local repos' source, Simple English Wikipedia — see table below |
| Built corpus | `data/train.txt` ≈1.19GB; `data/test.txt` ≈133MB |
| Tokenized (GPT-2 `tiktoken`, `encode_raw`) | **280,330,103 train tokens, 31,367,398 test tokens** (311,697,501 total) |
| Split | 90% train / 10% test, chat conversations and extra documents each split independently at the same ratio, then pooled and reshuffled together, fixed seed 42 |
| Document separator | `<|endoftext|>` — GPT-2's real reserved special token (see "Document boundaries," below — this is not the same as a plain blank line) |
| Format | plain text, `Role: message` lines for chat, raw prose/code for extra documents, every document (chat or extra) separated the same way |
| Training objective | raw next-token prediction over the whole stream — **base model, not instruction-tuned** (no chat template, no per-turn loss masking) |

## The 7 registered chat sources

| Dataset | Access | Conversations kept | Notes |
|---|---|---|---|
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | public | 100,000 | Synthetic — two ChatGPT instances conversing. Bulk volume. |
| [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) | public | 3,446 | Real raw schema is a flat reply-tree table (`message_id`/`parent_id`/`rank`), not a ready-made conversation list — **was silently parsing to 0 before a dedicated tree-reconstruction fix** (`load_oasst_conversations()`, walks each tree picking the best-ranked reply at each branch). English-only. |
| [`zidankhan/databricks-dolly-15k`](https://huggingface.co/datasets/zidankhan/databricks-dolly-15k) | public | 12,173 | The actual HF repo only ever shipped a raw `.jsonl`, no parquet — **was silently un-downloadable before a generic `refs/convert/parquet` fallback fix** (Hugging Face auto-converts every public dataset to parquet on that ref regardless of the source repo's native format). Also **was silently 0 conversations** even after that fix, because it's single-turn (exactly 2 turns: one prompt, one reply) and the old `min_turns=3` default discarded 100% of single-turn data — fixed by lowering the default to `2`, the real minimum for "a conversation happened." |
| [`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | public | 100,000 | Compact multi-domain mixture (reasoning, rewriting, summarization). |
| [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | public | 17,110 | Entirely human-written, no model anywhere in the loop — 10 categories (Generation, Open QA, Brainstorm, Chat, Rewrite, Summarize, Coding, Classify, Closed QA, Extract). **License: CC BY-NC 4.0 — non-commercial use only.** |
| [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | public | 16,860 | Grade-school math word problems with explicit step-by-step worked solutions (calculator-annotated arithmetic, e.g. `48/2=24`). Added specifically because this project's own QA reports showed the model attempting zero real arithmetic on simple word problems — targeted at that gap, not general growth. Both `main` and `socratic` train splits. |
| [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | **gated** | 0 | Correctly excluded — no `HF_TOKEN` configured in this environment. Real, unfiltered Chatbot Arena traffic; would add genuine messy-real-user-prompt diversity if a token were ever added. |

**Total: 249,589 chat conversations.**

Licenses: UltraChat MIT · OASST1 Apache-2.0 · Dolly CC BY-SA 3.0 · SmolTalk Apache-2.0 ·
No Robots **CC BY-NC 4.0 (non-commercial)** · GSM8K MIT · LMSYS has its own gated dataset
license (not currently in use here).

## The extra (non-chat) documents

Not conversational, so these don't go through the chat registry/schema parsers at all —
each was produced as a flat `{"text": ..., "token_count": ..., ...}` JSONL by
[`tools/corpus-extractor`](../../tools/corpus-extractor/) (a standalone Rust CLI, or —
for Wikipedia — a one-off Python script replicating its exact chunking behavior), then
merged in via `gpt-data --extra-jsonl <path>` (repeatable — pass it once per source).

| Source | Documents | Chunk size |
|---|---|---|
| Books (`~/Downloads/books`) | 48,912 chunks | 1024 tokens, 100 overlap |
| `eng-skills` repo | 1,216 chunks | 1024 tokens, 100 overlap |
| `OxideOS` repo | 1,051 chunks | 1024 tokens, 100 overlap |
| `platform-lab` repo | 2,231 chunks | 1024 tokens, 100 overlap |
| Simple English Wikipedia | 222,029 chunks | 1024 tokens, 100 overlap |

**Total: 275,439 extra documents — 71,245,373 tokens from Wikipedia alone, more than
either books or the three repos combined.**

### Books, in detail

665 real PDF files (~4.4GB) in `~/Downloads/books`. 80 additional `.epub`-named items in
that same folder are Apple Books' unpacked local-storage directory bundles, not real zip
EPUBs — `corpus-extractor` can't read that format, excluded from this pass. 610/665 PDFs
extracted successfully (55 failed — malformed PDFs with unusual internal structure,
cleanly skipped and logged after a real `pdf-extract` panic-safety fix, not silently
corrupting the run); 392 distinct books actually survived quality filtering into the
final corpus. Individual titles aren't listed here by choice — topic breakdown instead,
by keyword classification of filenames (a blunt instrument — most titles didn't match
any specific keyword and landed in "general nonfiction," so treat these as a floor, not
an exhaustive taxonomy):

| Topic (keyword-detected) | Books |
|---|---|
| Psychology / self-help / communication / persuasion | 44 |
| Language learning (IELTS, French, German, grammar) | 21 |
| Business / career / finance / productivity | 16 |
| Programming / tech / AI / data | 14 |
| Fiction / novels | 7 |
| Science / medicine / health | 5 |
| General nonfiction (no specific keyword matched) | 285 |

### The three repos, in detail

All three are `.rs`/`.md`/`.py` files only, walked `.gitignore`-aware (so `target/`,
`.venv/`, build artifacts are already excluded, not part of what got extracted). Real
top-level structure of each, for what the chunk counts actually represent:

- **`eng-skills`** (1,216 chunks: 1,211 md, 5 py) — an English-fluency and
  engineering-communication training repo, not a code project; the near-total `.md`
  share reflects that. Top-level: `Vocabulary-Collections/` (vocab, phrasal verbs,
  idioms, technical/architectural English, business communication, grammar, speaking
  toolkit), `Communication-Mastery/` (thinking frameworks, explanation frameworks,
  storytelling, interview/meeting/architecture communication), `Project_Management/`
  (PMBOK/PM literacy for engineers), `Book-Summaries/`.
- **`OxideOS`** (1,051 chunks: 886 rs, 165 md) — an OS-kernel project, Rust-dominated as
  expected. Top-level: `kernel/`, `userspace/`, `tools/`, `docs/`.
- **`platform-lab`** (2,231 chunks: 1,766 md, 335 py, 130 rs) — a hands-on
  platform/MLOps/cloud practice monorepo, the most topically diverse of the three.
  Top-level: `mlops_aiops/` (MLflow, Evidently, Feast, vLLM, monitoring pipelines),
  `cloud-practice/` (AWS/Terraform), `k8s_explorer/` + `k8s_mlops/` (Kubernetes),
  `genai_lab/` (MCP, LangGraph, RAG, vector DBs), `fundamentals/` (DSA, system design,
  LLD, security, behavioral prep), `rust_dsa_practice/`, `local_llms/`,
  `pytorch_exploration/`.

### Wikipedia, in detail

`wikimedia/wikipedia`, config `20231101.simple` (Simple English Wikipedia, not full
English Wikipedia — deliberately: simpler prose, smaller size, better match for a small
model). 241,787 articles total; 45,756 stub articles (<200 chars) dropped before
chunking. CC-BY-SA-3.0/GFDL. Added specifically because QA reports showed factual errors
(never naming Paris for "capital of France") — targeted at that gap, not general growth.

`private_profile/` (the fourth repo in this workspace) was deliberately **not**
included — it holds real employer/client identifiers by design; general-purpose training
data is the wrong place for that regardless of local-only use.

## Chunk size — why 1024, why overlap, why token-count not character-count

`--chunk-tokens 1024` is not an arbitrary number — it's chosen to **exactly match this
project's `context_length=1024`**. The reasoning: `dataset.py`'s `get_batch()` samples
random `context_length`-token training windows from the flat token stream regardless of
where document boundaries fall, so a book/repo/Wikipedia chunk doesn't need to be any
particular size for training itself to work — but sizing chunks to roughly one full
context window means a single training window is more likely to see one coherent
document mostly or entirely, rather than being dominated by fragments of many unrelated
short ones. (`custom-gpt-10m`'s sibling project uses the same books/repos, chunked at
`--chunk-tokens 512` instead, matching *its* `context_length=512` — same reasoning,
different number, because it's a different context window.)

Chunking is done in **GPT-2 tokens**, via `tiktoken`'s `r50k_base`/`gpt2` encoding — the
exact tokenizer this project trains against — not characters or words. This matters
because token-to-character ratio varies by content (dense technical prose tokenizes
differently than code, which tokenizes differently than casual English), so a
character-based chunk target would produce inconsistent actual token counts; a
token-based one doesn't.

`--chunk-overlap 100` means consecutive chunks from the *same* source document share
their last/first 100 tokens rather than cutting cleanly at a hard boundary — so a random
training window that happens to land near a chunk seam still sees continuous, coherent
context on both sides, instead of a context sentence being truncated with no lead-in.

## Document boundaries — why `<|endoftext|>`, and a real gotcha it took to get right

Every document — chat conversation or extra document, old data or new — is joined with
`<|endoftext|>`, GPT-2's own real reserved special token, not a plain `"\n\n"`. The
mechanism this matters for: `get_batch()`'s random windows regularly straddle two
unrelated documents (a chat exchange and a Wikipedia article, two different books,
etc.) — a soft separator like a blank line is a weak "this is unrelated, start fresh"
signal; a token GPT-2's own pretraining already primed the embedding space to recognize
as a hard boundary is a strong one.

**A real assumption that turned out wrong, caught by testing before trusting it**:
simply using the literal string `"<|endoftext|>"` as the join separator is *not* enough
on its own. `tiktoken`'s `encode(text, disallowed_special=())` alone only stops the
tokenizer from raising an error when it sees that string — without also passing
`allowed_special={"<|endoftext|>"}`, the literal string silently tokenizes as 7 ordinary
subword pieces (`<`, `|`, `end`, `of`, `text`, `|`, `>`), not the single real special
token id `50256`. Confirmed directly: `encode("<|endoftext|>", disallowed_special=())` →
7 tokens; adding `allowed_special` → `[50256]`, one token. Both `dataset.py`'s
`encode_raw()` (training) and `inference/generate.py`'s prompt encoding (inference-time
consistency) now pass `allowed_special` correctly.

## How this corpus actually got built — the full process, in order

1. **Chat sources**: `gpt-data` downloads each registered source's parquet (falling back
   to Hugging Face's auto-converted `refs/convert/parquet` ref if a repo never shipped
   native parquet), parses rows into `(role, text)` turns via schema-specific logic
   (`extract_turns_conversation`/`extract_turns_instruction`/`load_oasst_conversations`),
   applies per-turn quality filters (length, printable/ASCII ratio, alphabetic density,
   redaction-placeholder rejection), and collects one `[(role, text), ...]` list per
   conversation.
2. **Extra documents**: each source (books, each repo, Wikipedia) is extracted
   *separately* into its own `{"text": ...}` JSONL — `tools/corpus-extractor` for
   local folders (PDF/EPUB/txt/md/rs/html/js/py, `.gitignore`-aware directory walking,
   panic-safe per-file extraction), a matching one-off script for the remote Wikipedia
   dataset — all at the same 1024-token/100-overlap chunking.
3. **Pooling and splitting**: chat conversations are shuffled and split 90/10 (seed 42);
   each extra-document source is independently shuffled and split the same way; the two
   resulting train pools (and the two test pools) are then pooled together and
   reshuffled once more — so a book excerpt and a chat conversation can end up adjacent
   in the final file, not all-chat-then-all-extra in a block.
4. **Writing**: every document, from every source, joined with `<|endoftext|>` and
   written to `data/train.txt`/`data/test.txt`.

Held-out prompts (`data/test_prompts.txt`, used by `make eval`/`make test`) are derived
only from chat conversations — a book/Wikipedia/code excerpt has no natural "held-out
Assistant reply" to build a QA prompt from.

## A composition trade-off worth knowing, not hiding

**Extra documents (275,439) now outnumber chat conversations (249,589) by raw document
count.** This corpus has moved substantially toward general prose/code/reference content
and away from being "mostly chat-turn-shaped." That's a deliberate, explicit trade-off,
not an accident — but it means the model sees proportionally less of the
`"User: ...\nAssistant: ..."` pattern per training step than earlier, chat-only versions
of this corpus did. Whether that helped or hurt actual QA-style behavior is something to
check empirically against `reports/qa_report_*.html` once training has run a while, not
something to assume either way.

## Rebuilding it

```bash
gpt-data --list                          # print the chat-source registry from code
make data-public                         # rebuild chat sources only (no HF_TOKEN needed)
uv run gpt-data --extra-jsonl <path> [--extra-jsonl <path> ...]   # + extra documents
make audit                               # verify: noise rate, ASCII ratio, train/test overlap
```

Rebuilding the extra-document JSONLs themselves (if source folders change):

```bash
cd ../../tools/corpus-extractor
./target/release/corpus-extractor --input <folder> --output <out-dir> \
  --extensions pdf,epub,rs,md,py --chunk-tokens 1024 --chunk-overlap 100 --no-split
```

Full detail on every fix, every decision, and the exact order things happened in:
[`docs/BOOKS_CORPUS_INTEGRATION.md`](docs/BOOKS_CORPUS_INTEGRATION.md). Registry source
of truth: [`src/gpt/data/sources.py`](src/gpt/data/sources.py).
