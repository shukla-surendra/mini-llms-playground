# Publishing This Model to the Hugging Face Hub

Covers [`scripts/upload_to_hf.py`](../scripts/upload_to_hf.py) — what it uploads, and how
to actually get a token and verify the upload afterward. Why a custom model needs a raw-
files upload instead of `push_to_hub()`, and what makes a model card genuinely
self-contained, are covered in
[Chapter 31 — Publishing a Model: The Hugging Face Hub Workflow](../../../docs/llm-engineering/31_publishing_a_model_the_hugging_face_hub_workflow.md).
This doc covers only this project's exact file list, token setup, and verification steps.

## What gets uploaded, and why each file matters

| File | Why it's needed |
|---|---|
| `tinystories_gpt_checkpoint.pt` | The trained weights themselves — useless without everything below |
| `tokenizer.json` | The **exact** custom BPE vocabulary this checkpoint was trained with. Per [`CONTINUING_TRAINING_ON_NEW_DATA.md`](CONTINUING_TRAINING_ON_NEW_DATA.md#the-one-hard-requirement-the-tokenizer-must-stay-the-same), using any other tokenizer — even one with the same `vocab_size` — silently produces garbage, since token IDs would no longer mean the same things to the model |
| `model.py` | The `TinyStoriesGPT` class definition — without this, the checkpoint's `state_dict` (a dict of tensor weights) has no architecture to load into |
| `inference.py` | Generation logic (sampling, temperature, repetition penalty) — usable standalone once someone has the checkpoint |
| `api_server.py` | Lets someone stand up the same FastAPI server this project uses, from the downloaded model alone |
| `pyproject.toml` | Exact dependencies needed to run any of the above (`uv sync` / `pip install .` both work from it) |
| `model_card.md` → uploaded as `README.md` | The model page HF actually renders — see below |

## Getting a token, and where it lives

1. Go to <https://huggingface.co/settings/tokens>
2. Create a new token with **write** permission (read-only tokens can't upload)
3. `cp .env.example .env`, then edit `.env` and paste your real token in place of the
   placeholder — `.env` is gitignored (see the repo root's `.gitignore`), so this never
   gets committed. `scripts/upload_to_hf.py` loads it automatically via `python-dotenv`
   at the top of the script (`load_dotenv(PROJECT_DIR / ".env")`) — no manual `export`
   needed each session. An `HF_TOKEN` already set in your shell, or an explicit
   `--token`, both take priority over `.env` if you ever need to override it for one run.

## Running it

```bash
make publish REPO_ID=your-username/tinystories-gpt-6m
```

Or directly, bypassing the Makefile (still reads `.env` the same way):

```bash
uv run scripts/upload_to_hf.py --repo-id your-username/tinystories-gpt-6m
```

- `--repo-id` (or `REPO_ID=` for `make publish`) is required —
  `<your-username>/<model-name>`, created automatically if it doesn't exist yet
  (`create_repo(..., exist_ok=True)`).
- `--private` keeps the repo private (default is public) — pass it directly to the
  `uv run` form; there's no Makefile shortcut for this flag yet.
- `--checkpoint` defaults to `tinystories_gpt_checkpoint.pt` (the best-validation-loss
  checkpoint, per [`HOW_MUCH_TRAINING_IS_ENOUGH.md`](HOW_MUCH_TRAINING_IS_ENOUGH.md#signal-2-a-single-non-improving-evaluation-is-noise-not-a-stop-sign))
  and automatically falls back to `tinystories_gpt_checkpoint_latest.pt` if the primary
  one isn't found.
- `--tokenizer` defaults to `data/tokenizer.json` — **if you've been experimenting with
  [`CONTINUING_TRAINING_ON_NEW_DATA.md`](CONTINUING_TRAINING_ON_NEW_DATA.md)'s
  `--reuse-tokenizer` workflow, this still points at the same file**, since the whole
  point of that workflow is keeping one tokenizer across every dataset round — there's
  never a second tokenizer file to worry about picking correctly.

## The model card (`model_card.md` → `README.md` on the Hub)

A separate, real HF-formatted model card (YAML frontmatter with `license`, `tags`,
`datasets`, `pipeline_tag`) — **not** a copy of this project's own `README.md`; see
Chapter 31 for why those two documents have to diverge. Edit
[`../model_card.md`](../model_card.md) directly if you want to change what appears on the
model page — it's a real file in this repo, not generated inline by the script. It
includes this project's actual loss/perplexity numbers and a real generated sample, not
placeholders.

## Verifying it worked

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('your-username/tinystories-gpt-6m', 'tinystories_gpt_checkpoint.pt')
print('downloaded:', path)
"
```

Or just visit `https://huggingface.co/your-username/tinystories-gpt-6m` — the model card's
YAML frontmatter should produce a rendered page with the right tags/description, and a
file listing showing all six uploaded files. The model card's own usage snippet
(downloads the checkpoint + tokenizer + `model.py`, reconstructs the model, generates
text) is the real, runnable end-to-end test that a fresh download actually works — worth
running once after a new upload, since it exercises exactly what an outside user would do.

## Re-uploading after further training

Nothing about this script is one-shot — running it again with the same `--repo-id`
overwrites the existing files with whatever's currently on disk. This means after any
additional training round (per
[`CONTINUAL_TRAINING_LOW_RESOURCE.md`](CONTINUAL_TRAINING_LOW_RESOURCE.md)), re-running
the same upload command publishes the improved checkpoint to the same Hub repo — no flag
needed to indicate "this is an update."
