# Models — comparison and status

Six from-scratch GPT projects in this directory, each its own independent codebase
(own `pyproject.toml`, `Makefile`, `src/gpt/`), ordered by parameter count. This file
is a living index — update it in place as configs, data, or training status change,
rather than writing a new doc per snapshot. Numbers below were pulled directly from
each project's `gpt-config`, checkpoint metadata, and `data/train.bin.json` fingerprint
on 2026-08-17 — not estimated.

## Terminology: raw data vs. corpus vs. tokenized corpus

Three distinct stages, each with its own name in this codebase's docs — worth being
precise about since "corpus" gets used a lot below and means something narrower than
"raw data":

1. **Raw sources** (`data/raw/<source>/text.txt` or parquet) — exactly what got fetched
   from HuggingFace, one folder per dataset, untouched and unmixed, no train/test split.
   A reusable pool, not something anything trains on directly.
2. **Corpus** (`data/train.txt` + `data/test.txt`) — the *assembled, split, train-ready*
   plain text: raw sources concatenated (documents joined by `\n\n` or `<|endoftext|>`),
   split into train/test at document boundaries. This is what "corpus" means everywhere
   in this file and in each project's own `DATASET.md` — a distinct, derived artifact,
   not the raw pool itself (which is usually larger).
3. **Tokenized corpus** (`data/train.bin` + `data/test.bin`) — the corpus converted to
   token ids, flat `uint16` arrays. This is what `get_batch()` actually memory-maps and
   samples random windows from during training — not the `.txt` corpus directly.

The "Corpus (tokens)" column in the table below refers to stage 3 (what's actually
tokenized and trainable today) unless a section explicitly says otherwise.

## Shared raw data

`from_scratch/_shared_data/raw/` (gitignored) holds the raw fetched sources — pretraining
text (cosmopedia-v1/v2, finemath-4plus, Hindi Wikipedia, open-web-math) and the 8
registered chat/instruction sources (UltraChat, OASST1, Dolly, SmolTalk, No Robots,
GSM8K, LMSYS-Chat-1M, OpenHermes-2.5) — moved out of `custom-gpt-153m/data/raw/` on
2026-08-18, since that project's raw pool had already become the de facto source every
other project (50m, 350m) was copying or reading from directly. `custom-gpt-153m/
data/raw` is now a **symlink** to the shared folder (`../../_shared_data/raw`), so its
own scripts (`sources.py`, `fetch_pretrain_corpus.py`, `build_pretrain_split.py`,
`tokenize_direct_from_raw.py`) needed zero changes — they still open `data/raw/...`
paths, the OS resolves the rest.

This deliberately reverses this workspace's own earlier move *away* from shared/symlinked
data (see `custom-gpt-153m/DATASET.md`'s note that `data/` used to symlink into
`custom-gpt-10m` and was made independent) — the difference this time is the symlink
points at explicit, documented, workspace-level shared infrastructure (this folder),
not silently at a sibling project's own data. Other projects reusing this data going
forward should reference `../_shared_data/raw/...` directly rather than reaching into
`../custom-gpt-153m/data/...` (50m's and 350m's existing copies/tokenized `.bin` files
from earlier this session predate this move and were left as-is, not retroactively
changed).

## At a glance

| Model | Params | Context | Embed | Heads | Layers | FFN/Attn | Tokenizer | Vocab | Corpus (tokens) | 16x target | 20x (Chinchilla) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [6m](#custom-gpt-6m) | 5,853,184 | 256 | 256 | 8 | 6 | standard MLP | custom char/story (`meta.json`) | 4,096 | 22.4M | 93.6M | 117.1M | **Done** — step 5,097/5,000 |
| [10m](#custom-gpt-10m) | 9,979,040 | 512 | 160 | 8 | 6 | standard MLP | GPT-2 (`tiktoken`) | 50,257 | 280.3M | 159.7M | 199.6M | **In progress** — step 621,827/1,000,000 |
| [50m](#custom-gpt-50m) | 51,475,968 | 1024 | 512 | 8 | 8 | standard MLP | GPT-2 (`tiktoken`) | 50,257 | 2.458B | 823.6M | 1.030B | **Restarted, paused** — step 5,399/1,000,000 (v2) |
| [153m](#custom-gpt-153m) | 152,791,296 | 1024 | 768 | 12 | 16 | standard MLP | GPT-2 (`tiktoken`) | 50,257 | 2.989B | 2.445B | 3.056B | **Not trained** — corpus complete (~20x/param), training not started |
| [200m](#custom-gpt-200m) | 201,769,344 | 2048 | 896 | 14 | 18 | SwiGLU + RoPE + RMSNorm | `oxide-bpe-32k` (own, trained) | 32,768 | 9.7M (placeholder) | 3.228B | 4.035B | **Not trained** — architecture + tokenizer ready, real corpus not built |
| [350m](#custom-gpt-350m) | 347,360,256 | 2048 | 1024 | 16 | 25 | SwiGLU + RoPE + RMSNorm | `oxide-bpe-32k` (own, trained on 153m's corpus) | 32,768 | 3.083B | 5.558B | 6.947B | **Not trained** — tokenizer + corpus ready (~8.9x/param), training not started |

"16x/20x target" = tokens-per-parameter this project's own docs use as the sizing rule
(20x = Chinchilla-optimal, 16x = this project's usual "fits inside a bounded GPU budget"
compromise — see `custom-gpt-153m/DATASET.md` and `docs/MODEL_SIZING_GUIDE.md` in each
project for the reasoning). "Corpus (tokens)" is what's actually tokenized into
`data/train.bin` today, which for 153m in particular lags behind the raw pool sitting
in `data/raw/` (see its section below).

## Two architecture families

- **6m/10m/50m/153m** — the "classic" family: standard post-embedding position
  embeddings, LayerNorm, GELU-MLP, GPT-2's `tiktoken` BPE (50,257 vocab, shared and
  interchangeable across these four — a checkpoint from one is architecture-compatible
  with another only if `embed_size`/`num_layers`/`context_length` also match, but the
  *tokenizer* is always identical). 6m is a further outlier — its own tiny custom
  tokenizer (4,096 vocab) and a story-completion dataset (`meta.json`), not the shared
  chat+web corpus the other three use.
- **200m/350m** — the "modern" family: RoPE position encoding, RMSNorm, SwiGLU-gated
  MLP, and a project-trained custom 32,768-vocab BPE tokenizer (`oxide-bpe-32k`) that is
  **not** compatible with the classic family's GPT-2 tokenizer or with each other's
  weights unless both are trained on the same `tokenizer.json`. Both now have a trained
  tokenizer (350m's trained 2026-08-18, on a sample of 153m's corpus), but each has its
  **own separate `tokenizer.json`** — 200m's and 350m's token ids are not
  interchangeable with each other either, despite sharing the same vocab size and
  training recipe.

Practical implication: raw **text** can be freely reused across any of the six projects
(a `.txt` file doesn't care which tokenizer reads it later), but tokenized `.bin` files
cannot — moving data from a classic-family project into 200m/350m always requires a
re-tokenize pass with the target project's own tokenizer.

## Context length, concretely

Three context lengths appear across the six projects — 512 (10m), 1024 (50m/153m), 2048
(200m/350m). "Context length" is the number of tokens the model can attend to at once;
anything before the window's start is invisible to it, not summarized or remembered —
just gone. What that actually means in terms of real content is easy to lose track of as
an abstract number, so here it is against one real, unmodified document already sitting
in `custom-gpt-153m`'s corpus (`data/raw/HuggingFaceTB__finemath__finemath-4plus/text.txt`,
a real math-review blog, 8,640 GPT-2 tokens total — encoded/decoded with `tiktoken`'s
`gpt2` encoding, the same tokenizer 6m/10m/50m/153m all use).

**First 512 tokens** (2,118 characters) — everything a 10m-preset model can ever see at
once, no matter how long the source document actually is:

> Article posted June 2, 2012 at 07:33 AM GMT-5 • comment • Reads 539 Today is Sunday,
> June 10, and I'm doing this blog a week late because the website crashed last week. It
> also means that there are exactly 2 school days until Finals start! [...] Determine
> which three numbers could be the sides of a right triangle. A. 64,73,98 B. 64,72,96 C.
> 65,72,97 The answer is C because of the Pythagorean theorem. [...] Article posted June
> 10, 2012 at 07:40 PM GMT-5 • comment • Reads 291 For this weeks blog I was asked to
> post two review questions for the final exam. [...] My first question is: What does
> MAPA COCI stand for? A hint

Cuts off mid-sentence, mid-question — the 513th token (whatever it turns out to be) is
simply never seen by a 512-context model reading this document from the start.

**Tokens 513–1024** — what a 1024-context model (50m, 153m) sees that a 512-context
model (10m) cannot, in the same read-through:

> [...] for this question is to remember lines in triangles and their concurrent
> points! My next question is: How do you find the circumference of a circle with a
> radius of 10 cm? Answers: [...] The problem I am working on for this blog is # 12 from
> the green version on test # 7. It asks for you to find the area of a parallelogram
> that has sides of 8 and 6. [...] It asks you find the area of a deck that surrounds a
> hot tub if the hot tub has a diameter of 6 meters and the deck is 2 meters wide. [...]
> The way to figure out this problem is to find the area of the hot tub and then
> subtract that from the area of the deck. The only formula that you need to use on this
> problem is a=pie(r) squared. The first thing to do

A full extra worked problem (the parallelogram) plus the start of a third — content a
512-context model never gets the chance to condition on.

**Tokens 1025–2048** — the additional reach 200m/350m get over 50m/153m:

> [...] is sub the number 3 into that formula and you end up with 9 pie. [...] Your
> final answer should be 16 pie. Article posted June 10, 2012 [...] Which of the
> following are the slopes of two perpendicular lines? [...] THEREFORE D IS THE
> ANSWER! [...] Article posted June 10, 2012 at 10:34 PM [...] Consecutive sides of a
> rectangle are congruent. a. sometimes b. always c. never the answer is SOMETIMES
> [...] For Ch 12, the work was on transformations. [...] Our final exam is Wednesday!
> Ahhh!

Three more full review-question-and-answer posts — and even at 2048 tokens, this single
document still has **6,592 tokens left over, unseen by any of the six models** at their
configured context length. The takeaway isn't "bigger is strictly better" (longer
context costs quadratically more attention compute per token) — it's that context length
is a hard, literal wall on how much of a real document a model can condition its
prediction on at once, not a soft budget.

### Is context length a limit on input, or on output?

**Both — jointly, not separately.** It's one shared window: `context_length` is the
total number of token *positions* the model has ever learned to attend across (its
position embeddings / RoPE table only go up to that number). Every token that ever needs
to be attended to — the prompt **and** every token generated so far — has to fit inside
that same window at once. There's no separate "input budget" and "output budget."

This project's real inference path (`generate.py`'s KV-cache/`sdpa` loop, what every
trained checkpoint actually uses) makes the arithmetic explicit:

```
steps = min(max_new_tokens, context_length - prompt_length)
```

`max_new_tokens` is a request — how many tokens *you'd like*. `context_length -
prompt_length` is what's actually *available* — whatever room the prompt didn't already
use. The smaller of the two wins. Concretely, at `context_length = 1024`:

| Prompt length | Room left for generation | What happens |
|---|---|---|
| 50 tokens | 974 tokens | Generates up to `max_new_tokens`, capped at 974 |
| 800 tokens | 224 tokens | Generation gets cut short once the window fills, even if `max_new_tokens` asked for more |
| 1024 tokens | 0 tokens | **No new tokens at all** — the window is already full before generation starts |
| 1200 tokens (over the limit) | — | The prompt itself gets silently truncated to the last 1024 tokens *before* generation begins (`ids = ids[:, -context_length:]`) — the earliest 176 tokens of the prompt are dropped and never seen, not an error |

So: yes, context length limits how much you can supply as input — but not on its own.
A long prompt doesn't get rejected, it either eats into the generation budget or (past
the hard cap) gets silently truncated from the front. And a short prompt doesn't
guarantee a long generation either — that's still bounded by whatever `max_new_tokens`
you actually pass. The one deliberately hard, non-negotiable number is `prompt_length +
generated_so_far ≤ context_length`, always.

## Beyond research/exploration: realistic production niches per size

None of these six are candidates for general-purpose assistants, agentic/tool-use
systems, or anything requiring multi-step reasoning — that's a capacity ceiling, not a
data problem (see the SRE-agent discussion this doc's history has on that: reasoning and
compositional tool-use are scale-dependent capabilities that don't reliably show up
below roughly the 1B-3B range, no matter how good the training data is). What *does*
scale down are narrow, single-purpose, latency/cost/privacy-driven jobs where a bigger
model would be correct but overkill. Grounded in what models of comparable real size
have actually been deployed for, not aspirational:

- **6m** (5.85M) — below general text generation entirely. Realistic niche: an
  on-device, offline, narrow-vocabulary component — predictive text for a fixed tiny
  domain (a game's flavor-text/NPC-bark generator, a keyboard's next-char suggester on
  hardware too constrained for anything bigger), or a classifier head repurposed from
  the same architecture rather than a generator. Not a chatbot in any form.
- **10m** (9.98M) — roughly `TinyStories`-scale, where actual research (the TinyStories
  paper) shows coherent short-form text is achievable at this size *for a narrow enough
  domain*. Realistic niche: a heavily-constrained, fixed-intent responder (a smart-home
  device's handful of canned replies, template-filling rather than open generation) —
  never open-domain chat, always with output validation downstream.
- **50m** (51.5M) — below GPT-2-small (124M). Realistic niche: latency-critical
  autocomplete where a human always reviews before anything ships — commit-message
  suggestions, search-query completion, ticket-template stubs. The pattern here isn't
  "the model is smart," it's "the model is fast/cheap/offline-capable and a human is the
  actual quality gate."
- **153m** (152.8M) — comparable to GPT-Neo-125M/OPT-125M, both of which have seen real
  (if narrow) production use: lightweight content-moderation pre-filters, low-latency
  "smart reply"-style draft suggestions (the pre-LLM generation of Gmail Smart Reply ran
  on models in this range), and — a genuinely current pattern — **draft models in
  speculative decoding**, where a small fast model proposes tokens a larger model
  verifies/corrects, speeding up big-model inference without the small model needing to
  reason on its own.
- **200m** (201.8M) — same tier as 153m, plus enough headroom for narrow-domain
  generation from structured input (templated product descriptions, changelog-entry
  drafting from a diff) when the domain is tightly bounded and a human or validator
  checks the output.
- **350m** (347.4M) — GPT-2-medium territory (355M). The most realistic "general-ish"
  niche of the six, but still only viable paired with **retrieval (RAG)** doing the
  actual knowledge/fact work — the model's job becomes phrasing and formatting retrieved
  content into fluent text, not recalling or reasoning over facts itself. Also the best
  candidate of the six for the speculative-decoding draft-model role at real production
  latency budgets.

Common thread across all six: the honest production angle isn't "smart enough to act
autonomously," it's **cheap, fast, offline/on-device, and narrow** — value comes from
constraints a frontier model can't match (zero network round-trip, fixed hardware
budget, predictable latency), not from capability a frontier model already has.

### Where you'd actually deploy each one

The niche above is the *pattern*; this is a concrete place to point a trained
checkpoint. Every one of these six projects already ships `gpt-serve` (a FastAPI
inference server, `make serve`) — that's the one piece every option below shares, the
serving layer is not something to build from scratch.

| Model | Concrete deployment target |
|---|---|
| 6m | A fixed-vocabulary offline component — e.g. hooked into a Unity/Godot game's NPC-dialogue or flavor-text system via a local `gpt-serve` call, no network round-trip needed. Not a standalone product. |
| 10m | A Raycast/Alfred/PowerToys-style launcher plugin for canned quick-replies, or a local smart-home hub's fixed-intent responder (`gpt-serve` running on the hub itself, e.g. a Raspberry Pi) — small, closed set of intents, always. |
| 50m | `gpt-serve` behind a VS Code/IDE extension for commit-message or code-comment-stub suggestions — draft-only, developer always edits before committing. Also viable as search-query/ticket-template autocomplete behind a personal app's search bar. |
| 153m | `gpt-serve` behind a Slack/Discord bot for "smart reply"-style draft suggestions in a personal or small-team chat. The more technically interesting option: a **speculative-decoding draft model** paired with a larger local model via `vllm`/`llama.cpp` — this workspace's own `platform-lab/genai_lab/` already has local-LLM tooling (Ollama, LangGraph) this could plug into directly. |
| 200m | `gpt-serve` generating templated content from structured input — e.g. changelog-entry drafts from a `git diff` in one of this workspace's own repos, or product-description drafts from a spec sheet — always with a human-review step before anything ships. |
| 350m | `gpt-serve` paired with a local vector DB for retrieval-augmented Q&A over a narrow personal knowledge base — `platform-lab/genai_lab/`'s existing RAG/vector-DB experimentation is the natural place to wire this in, querying e.g. this workspace's own accumulated docs/notes rather than open-domain facts. |

None of these are "ship it to strangers on the internet" products — they're personal/
small-team tools where the model's narrowness and the deployer's own review are both
load-bearing parts of the design, not gaps to route around.

## Scaling up to ~1B params — what it would actually take

Grounded in this session's own **measured** GPU numbers (`infra/gcp-gpu-node/docs/
training_sop.md`), not guessed — 50m measured 13.3 steps/sec (~56,900 tok/s), 153m
measured 1.45 steps/sec (~23,760 tok/s), both on a single L4. tok/s falls roughly with
model size as expected (more FLOPs/token), which lets a ~1B-param extrapolation be
built the same way this project builds every other estimate: from the roofline, not
a guess.

**Compute FLOPs/token ≈ 6 × params** (standard transformer estimate). L4's dense bf16
peak is ~121 TFLOPS; 153m already runs at 18.2% of that (its measured MFU). A ~1B model
is more compute-bound than 153m (better compute:memory-access ratio per FLOP), so
assume MFU improves somewhat — 20-25% is a reasonable, still-conservative range:

| GPU | Peak bf16 (dense) | MFU assumed | Effective tok/s | 16x budget (16B tokens) wall-clock | Cost (on-demand) | Cost (spot) |
|---|---|---|---|---|---|---|
| L4 (this session's GPU) | 121 TFLOPS | 20-25% | ~4,000-5,000 | **~37-46 days** | ~$625-780 ($0.70/hr) | ~$270-335 (~$0.30/hr) |
| A100 80GB | 312 TFLOPS | 20-25% | ~10,400-13,000 | **~14-18 days** | ~$1,320-1,700 ($3.93/hr) | ~$765-980 ($2.27/hr) |

**These are projections, not measurements** — the same "verify before committing real
money" discipline this project already uses everywhere else applies here too:
`gpt-benchmark --sweep-batch` on whatever GPU is actually rented, before any multi-week
run, exactly as was done for 50m and 153m this session.

**The real constraint is wall-clock time, not dollar cost.** On L4, the cheapest option,
a full Chinchilla-16x run is over a month of *continuous, uninterrupted* single-GPU
training — which makes spot-preemption-safe resume (already built and proven in this
project's checkpoint/resume path) load-bearing rather than optional, and makes A100-class
hardware genuinely worth its higher hourly rate purely to compress a 6-week commitment
into 2.

**Data is not the blocker it looks like at first.** 16B tokens (16x) sounds far past
153m's enriched pool (~7B after this session's round-2 fetch), but per this same
project's own repetition-tolerance rule (used already to justify 153m's 16x-over-20x
compromise, and 350m's reused-pool situation): 16B / 7B ≈ **2.3 epochs** over the
existing pool — comfortably inside the "~4 epochs costs little" tolerance. **No new
fetching is strictly required** to reach the 16x floor at 1B params; the existing pool
just gets cycled through roughly twice.

**Where it would go once trained**: same `gpt-serve` deployment pattern as 350m above —
paired with retrieval doing the factual heavy lifting, the model's own job is fluent
phrasing over retrieved content, now with enough headroom to also serve as a genuinely
capable speculative-decoding draft model for something much larger.

---

## custom-gpt-6m

5,853,184 params · ctx 256 · embed 256 · 8 heads · 6 layers · standard MLP · custom
4,096-vocab tokenizer · `checkpoints/6m/causal/`

Smallest and only **fully completed** run in this set: step 5,097 against a 5,000-step
target, `final.pt` present, 41.76M tokens processed. Trains on a synthetic short-story
dataset (100,000 train stories / 2,000 val, 22.4M/0.4M tokens) rather than the chat+web
corpus the larger classic-family models use — a from-scratch sanity-check project more
than a capability target.

## custom-gpt-10m

Real preset: 9,979,040 params · ctx 512 · embed 160 · 8 heads · 6 layers · standard MLP
· GPT-2 tokenizer · `checkpoints/10m/`. (`gpt-config` with no `GPT_PRESET` set shows the
50m architecture instead — that's just this file's dataclass field defaults, not what
was actually trained; the saved checkpoint's file size, ~120MB, confirms the real
9.98M-param preset was used via `GPT_PRESET=10m`.)

**In progress**: step 621,827 of a 1,000,000-step budget, `best_test_loss` 2.88, ~15.6
hours of training time logged. Corpus: 280.3M tokens (chat + books + Wikipedia +
repos — the corpus 50m and 350m originally symlinked from before each went
independent). At 280.3M tokens against a 9.98M-param model that's ~28x/param, well past
even the 20x Chinchilla mark — this model has more data headroom than any other in the
set relative to its size.

## custom-gpt-50m

51,475,968 params · ctx 1024 · embed 512 · 8 heads · 8 layers · standard MLP · GPT-2
tokenizer · `checkpoints/50m/`

**Two runs, tracked separately:**

- **v1 (retired)** — trained to step 914,132/1,000,000 on the original mixed
  chat+books+Wikipedia+repos corpus, `best_test_loss` 2.5040. QA reports plateaued
  without reaching the quality bar through 91% of the run; root cause flagged in
  `docs/TRAINING_QA.md` as corpus dilution (extra non-chat documents outnumbering chat
  conversations from step 1). Archived in full — weights, every QA report, eval
  history, and the original corpus — at
  `archive/50m_run1_step913k_pre-153m-restart_2026-08-17/`.
- **v2 (current, paused)** — restarted from step 0 on a new corpus copied from
  `custom-gpt-153m`'s enriched raw pool (cosmopedia-v1/v2, finemath-4plus, Hindi
  Wikipedia — 2.458B tokens, ~19.9 tok/param). Two-phase training is now supported
  (see `docs/TWO_PHASE_TRAINING.md`): this v2 run is phase 1 (pretrain); a
  `data/profiles/posttrain/` corpus (249,589 pure chat conversations, no extra-doc
  dilution) is already built and ready for phase 2 once phase 1 reaches a good stopping
  point. Currently paused at step 5,399 for manual resume — see that doc for the exact
  resume command.

## custom-gpt-153m

152,791,296 params · ctx 1024 · embed 768 · 12 heads · 16 layers · standard MLP ·
GPT-2 tokenizer · `checkpoints/153m/`

**Not trained** — the one checkpoint present is a benchmark/speed-test artifact (`step:
-1`, 0 processed tokens, ~65s total — from an earlier GCP speed-measurement session,
not a real training run). `data/train.bin`/`test.bin` were rebuilt 2026-08-18 from the
now-complete enriched raw pool (`scripts/build_pretrain_split.py`, memory-safe chunked
version — an in-memory version of this script was OOM-killed once on this machine at
~13GB combined input, see the script's own docstring): **2.958B train tokens + 29.9M
test ≈ 2.989B total**, right at the 20x Chinchilla target (3.056B). Five sources:
cosmopedia-v2 (1.2B), cosmopedia-v1 (~488M), finemath-4plus (600M), Hindi Wikipedia
(150M), open-web-math (300M, new). `data/finetune_corpus/` holds a
separately-preserved, already-built pure-chat corpus (1.32B tokens across the 7
registered sources, including a working LMSYS-Chat-1M — unlike 50m, this project has a
real `HF_TOKEN`) for a future post-training phase, mirroring 50m's two-phase setup.
Ready to train as configured — nothing blocking a real run at this point.

## custom-gpt-200m

201,769,344 params · ctx 2048 · embed 896 · 14 heads · 18 layers · SwiGLU+RoPE+RMSNorm ·
`oxide-bpe-32k` tokenizer (**trained**, `tokenizer/tokenizer.json` present) ·
`checkpoints/200m/`

**Not trained** — checkpoint present is a benchmark artifact identical in shape to
153m's (`step: -1`, 0 tokens). A stray `checkpoints/tiny/` directory also exists from
an unrelated small-preset test run, not this project's real target size. `data/`
currently holds only a 9.7M-token placeholder corpus — real data collection for this
size hasn't started yet. Architecturally the direct scale-up sibling of 350m (same
SwiGLU/RoPE/RMSNorm family, own tokenizer already trained), so once its own data is
sourced it's the natural rung between 153m and 350m rather than a dead branch.

## custom-gpt-350m

347,360,256 params · ctx 2048 · embed 1024 · 16 heads · 25 layers · SwiGLU+RoPE+RMSNorm
· `oxide-bpe-32k` tokenizer (**not yet trained** — `tokenizer/` directory is empty) ·
`checkpoints/350m/` (does not exist yet)

**Data and tokenizer are now ready; training has not started.** `data/` still has the
old symlink farm into `custom-gpt-10m`'s corpus for the extra staging dirs
(`books_staging`, `hf_cache`, `raw`, `repos_staging`, `wikipedia_staging`) — unused, left
as-is — but `data/train.bin`/`test.bin` are now real, independent files. Its own
`DATASET.md` is still stale (describes 153m's old 152.8M-param/2.46B-token numbers
verbatim, a leftover copy) — the real Chinchilla targets recomputed above (5.558B/16x,
6.947B/20x against the actual 347.36M params) supersede it; not yet corrected in the
doc itself.

**What was actually built (2026-08-18), reusing `custom-gpt-153m` rather than fetching
350m's originally-documented separate 220B-token fineweb-edu mix:**

1. `oxide-bpe-32k` trained (`make tokenizer` / `gpt-train-tokenizer`) on a 2GB sample of
   153m's rebuilt `data/train.txt` — 122s, 32,768 vocab, 3.11 chars/token measured on a
   short sample sentence. `tokenizer/tokenizer.json` now exists.
2. `scripts/tokenize_from_153m.py` (new, one-off) tokenizes 153m's `train.txt`/`test.txt`
   **directly into this project's own `data/train.bin`/`test.bin`**, streaming rather
   than copying the 13GB source text first — disk was down to ~19GB free at the time,
   not enough headroom for a full text copy plus its own `.bin`. Result: **3,052,253,747
   train tokens + 30,815,351 test ≈ 3.083B total** (5,822 MB + 59 MB as `uint16`).

That lands at **~8.87 tokens/param** against this model's real size — short of even the
16x floor (5.558B), so a real training run here means either multiple epochs over the
reused pool (repetition up to ~4 epochs costs little per this project's own docs) or
fetching more data later, not a single-pass Chinchilla-optimal run out of the gate.
`make train-fresh` (or equivalent) is the next real step whenever training starts.
