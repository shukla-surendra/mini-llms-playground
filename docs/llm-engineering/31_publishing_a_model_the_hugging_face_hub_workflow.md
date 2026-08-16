# Publishing a Model: The Hugging Face Hub Workflow

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 4 — Serving: Turning a
Trained Model Into Something You Can Talk To (appended after the original numbered
catalog — see [Chapter 0](00_roadmap.md)'s reading-order note). Builds on
[Chapter 27](27_checkpointing_and_resuming_training.md)'s checkpoint contents — this
chapter is about what it takes to hand a trained checkpoint to someone who has never seen
this repo's code.

## In Plain English

A model trained with `transformers`-library classes (`AutoModelForCausalLM` and similar)
gets a one-line `model.push_to_hub()`, because the library already knows how to serialize
itself in a format the Hub's ecosystem understands automatically. A model built as a
plain, custom `nn.Module` — the case for every from-scratch model in this repo — has no
such shortcut: publishing it means uploading the raw pieces a stranger needs to
reconstruct and run it themselves, since there's no framework class on the other end
already waiting to load it.

## The First-Principles Explanation

### What a custom model publication actually needs, and why each piece is necessary

A `state_dict` alone (per [Chapter 27](27_checkpointing_and_resuming_training.md)) is
just a dictionary of tensors — useless to a downloader without three more things:

1. **The exact tokenizer used during training.** Per
   [Chapter 28](28_catastrophic_forgetting_and_continual_training.md)'s tokenizer-mismatch
   lesson, even a same-`vocab_size` tokenizer with a different token-ID mapping produces
   garbage — the tokenizer isn't optional metadata, it's load-bearing for the checkpoint
   to mean anything.
2. **The model-definition code itself.** Without the class definition, a `state_dict` has
   no architecture to load into — `transformers` avoids shipping this per-model because
   its architectures are already implemented library-side; a custom model has no such
   library to lean on, so the class definition has to travel with the weights.
3. **Enough of the surrounding code (generation logic, dependency manifest) that someone
   can actually run inference**, not just load tensors into memory. Publishing weights a
   downloader can load but not use isn't a meaningfully complete publication.

### The model card is a different document from the project README, not a copy of it

A project `README.md` is written for someone browsing the repository — it assumes
repo-navigation context, uses relative links to a `docs/` folder, and orients around "how
do I run this code." A Hub model card is written for someone who has landed directly on
the model's Hub page with **none** of that context — it needs to be self-contained: real
training results (actual numbers, not placeholders), a runnable usage snippet that
downloads and runs the model with nothing else assumed, and YAML frontmatter (`license`,
`tags`, `datasets`, `pipeline_tag`) that the Hub's site uses for search, filtering, and
its auto-rendered inference widget. Writing one document to serve both audiences tends to
under-serve whichever one didn't write it — treating them as genuinely separate documents
with separate audiences produces a better result for both.

### Publishing is not one-shot

Nothing about a well-built upload script is single-use — running it again against the
same repo ID with newer files on disk overwrites what's there. This means "publish an
updated checkpoint after further training" is the same command as the original publish,
not a distinct "update" operation requiring its own flag — the Hub repo simply reflects
whatever was most recently uploaded to it.

## Grounded in This Repo's Code

[`from_scratch/custom-gpt-6m/scripts/upload_to_hf.py`](../../from_scratch/custom-gpt-6m/scripts/upload_to_hf.py)
uploads exactly the raw-files set the reasoning above requires — the checkpoint,
`tokenizer.json` (this project's own custom-trained BPE vocabulary, per
[Chapter 9](09_tokenization.md)), the `model.py` class definition, `inference.py`'s
generation logic, `api_server.py` so a downloader can also stand up the same serving
endpoint, a dependency manifest, and `model_card.md` uploaded as the Hub repo's
`README.md` — a real, separate file from this project's own repo-facing `README.md`, not
generated inline by the script. The equivalent script in
[`from_scratch/custom-gpt-153m/scripts/upload_to_hf.py`](../../from_scratch/custom-gpt-153m/scripts/upload_to_hf.py)
follows the identical pattern, minus a tokenizer file — that project reuses GPT-2's public
tokenizer as-is rather than training its own, so there's no project-specific tokenizer
file to publish.
[`from_scratch/custom-gpt-6m/docs/PUBLISHING_TO_HUGGING_FACE.md`](../../from_scratch/custom-gpt-6m/docs/PUBLISHING_TO_HUGGING_FACE.md)
covers this project's exact file list, token setup, and verification steps.

## Deep-Dive: Why the Model Card's Self-Containment Requirement Is Stricter Than It Looks

A project README can say "see `docs/DATASET_AND_TOKENIZER.md` for why this dataset was
chosen" and rely on that file existing one directory away. A model card cannot do this
meaningfully — a Hub visitor has the model card and whatever files were uploaded alongside
it, not the rest of the source repository. Every claim the model card makes about how to
use the model has to be actually verifiable from what's in the Hub repo itself: a usage
snippet that downloads the checkpoint, tokenizer, and model code, reconstructs the model,
and generates text is the real test of self-containment — if that snippet doesn't work
using only what a fresh clone of the Hub repo provides, the model card has silently
assumed context that isn't actually there for its actual audience.

## Try It Yourself

- List every file `upload_to_hf.py` uploads in this repo and, for each one, answer
  concretely: what breaks for a downloader if this specific file were missing?
- Compare a project's own `README.md` to its `model_card.md` side by side — identify at
  least two things the model card includes that the README doesn't need (or vice versa),
  and connect each to the difference in audience described above.

## Common Misconceptions

- **"Any PyTorch model can use `push_to_hub()`."** That convenience is a `transformers`-
  library feature for its own model classes; a custom `nn.Module` has no built-in
  equivalent and needs an explicit raw-files upload instead.
- **"The model card can just be a copy of the README."** They serve different audiences
  with different available context — a model card has to be self-contained in a way a
  repo-internal README, which can rely on relative links to the rest of the repo, doesn't
  need to be.
- **"Publishing an updated checkpoint requires a special 'update' workflow."** It doesn't —
  re-running the same publish command with newer files on disk overwrites the Hub repo's
  contents; there's no separate update path to learn.

## Practice Questions

1. Why does a custom `nn.Module` need its class definition uploaded alongside its weights,
   when a `transformers`-based model doesn't need to publish its architecture code at all?
2. A model card links to `docs/ARCHITECTURE.md` in the source repo for "why this
   parameter count." Explain concretely what a Hub visitor experiences when they click
   that link, and why that's a design mistake specific to model cards.
3. What does re-running a publish script with a newer checkpoint on disk actually do to
   the files already present in the Hub repo?

## Key Terms

- **Raw-files publication**: uploading a custom model's weights, tokenizer, class
  definition, and enough runnable code directly, as the substitute for a
  `transformers`-native `push_to_hub()` call.
- **Model card**: a Hub-rendered, self-contained document (YAML frontmatter plus usage
  instructions) written for a visitor with no access to the source repository — distinct
  from a project's own `README.md`.
- **Self-containment (model card)**: every instruction in the model card must be
  verifiable using only what's actually uploaded to the Hub repo, not context assumed from
  the rest of the source repository.
