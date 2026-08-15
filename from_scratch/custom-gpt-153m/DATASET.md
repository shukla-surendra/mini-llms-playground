# Dataset

Quick-reference summary of what this model trains on. For the full pipeline (schema
parsing, quality filters, reproducibility, verification commands), see
[`docs/DATASETS.md`](docs/DATASETS.md) — the registry itself lives in code at
[`src/gpt/data/sources.py`](src/gpt/data/sources.py), which is the single source of
truth; this file and `docs/DATASETS.md` both just describe it.

**This project (`custom-gpt-153m`, 152,791,296 params, context 1024) trains on the exact
same corpus as the sibling [`custom-gpt-10m`](../custom-gpt-10m/DATASET.md) and
[`custom-gpt-50m`](../custom-gpt-50m/DATASET.md) projects** — `data/` here is a symlink
to `custom-gpt-10m`'s copy, and only model size/context length differ between the three.

## At a glance

| | |
|---|---|
| Sources | 5 Hugging Face conversation/instruction datasets, merged |
| Cap per source | 100k conversations (`--max-per-dataset`) |
| Built corpus | `data/train.txt` 982MB / 10,241,829 lines; `data/test.txt` 109MB / 1,143,073 lines |
| Tokenized (GPT-2 `tiktoken`) | ~173.7M train tokens, ~19.3M test tokens |
| Split | 90% train / 10% test, by **conversation** (never split mid-conversation), fixed seed 42 |
| Format | plain text, one `Role: message` per line, blank line between conversations |
| Training objective | raw next-token prediction over the whole stream — **base model, not instruction-tuned** (no chat template, no per-turn loss masking) |

## The five sources

| Dataset | Access | Rough size | Role in the mix |
|---|---|---|---|
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | public | ~200k dialogues | bulk conversational volume (synthetic, two ChatGPT models talking) |
| [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) | public | ~161k messages | genuine human-authored phrasing, crowd-sourced conversation trees |
| [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | public | ~15k pairs | small, hand-written, single-turn instruction/response |
| [`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | public | compact mixture | task diversity (reasoning, rewriting, summarization) |
| [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | **gated** — needs `HF_TOKEN` | 1M conversations | real, unfiltered user prompts from Chatbot Arena |

Licenses: UltraChat MIT, OASST1 Apache-2.0, Dolly CC BY-SA 3.0, SmolTalk Apache-2.0,
LMSYS-Chat-1M has its own dataset license (accept terms on the HF page before use — it
contains unfiltered real user content).

## Why this mix

Synthetic sets (UltraChat, SmolTalk) give volume and fluency; human sets (OASST1, Dolly)
give authentic phrasing and hand-checked quality; LMSYS gives real, messy user traffic —
typos, abrupt phrasing, incomplete questions — that curated sets don't have. Each turn is
independently quality-filtered (length, printable/ASCII ratio, alphabetic density,
redaction-placeholder rejection, code-fence cap) before merging; see
[`docs/DATASETS.md`](docs/DATASETS.md#quality-filters) for exact thresholds.

## Rebuilding it

```bash
gpt-data --list          # print the registry from code
make data                # all five sources (needs HF_TOKEN for LMSYS)
make data-public         # four public sources only, no token
make audit                # verify: noise rate, ASCII ratio, train/test overlap
```
