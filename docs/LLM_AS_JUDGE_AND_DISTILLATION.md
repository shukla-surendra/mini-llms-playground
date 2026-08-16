# LLM-as-judge, and distilling a bigger model

Two related uses of a locally-hosted teacher model: **grading** your small model's
output, and **training on** the teacher's output. They share a setup and have very
different economics and legal footing.

> **On the model name.** There is no "Gemma 4" that I am aware of — Google's line runs
> Gemma, Gemma 2, Gemma 3 (1B / 4B / 12B / 27B). This document uses **`gemma3:4b`**,
> the 4-billion-parameter Gemma 3. Run `ollama list` and adjust the tag if yours
> differs; nothing here depends on the specific version.

---

# Part 1 — LLM-as-judge

## When it earns its place

You already have three evaluation layers, and a judge sits above all of them:

| tool | measures | fails at |
|---|---|---|
| `test_loss` | next-token prediction | says nothing about whether an answer is *right* |
| `gpt-eval` | well-formedness (non-empty, non-repetitive, no role leak) | **saturates** — pins near 100 once garbage stops |
| `gpt-score` | correctness on prompts with a checkable answer | cannot grade open-ended answers |
| **LLM judge** | open-ended quality — helpfulness, coherence, relevance | slow, biased, needs validating |

The gap a judge fills is precise: `gpt-score` can mark "capital of France → Paris", but
it cannot mark "give me two tips for sleeping better". Roughly two thirds of the QA
prompt set is open-ended and currently only readable by you.

## Setup

```bash
ollama serve &                 # the API listens on :11434
ollama pull gemma3:4b          # ~3.3 GB
ollama list
curl -s http://localhost:11434/api/tags | head
```

A 4B judge grading a ~50M student is a wide enough gap to be meaningful. Judging a
*capable* model with a 4B judge is not — see the reliability section.

## Design: pairwise, not absolute

The instinct is to ask "score this answer 1–5". Don't start there. **Pairwise
comparison is substantially more reliable** than absolute scoring, for a mechanical
reason: a small judge has no stable internal calibration of what "4 out of 5" means, so
absolute scores drift between prompts and between runs. "Which of these two is better"
requires no calibration — it is the same question every time.

That also matches what you actually want to know: *is checkpoint B better than
checkpoint A*, not *is checkpoint B objectively a 3.7*.

**Control for position bias.** Judges systematically favour whichever answer appears
first. Ask each comparison twice with the order swapped and only count a win when the
judge picks the same answer both times; disagreement is a tie. This roughly halves
throughput and is not optional — without it you are measuring presentation order.

**Force structured output.** Ollama supports a JSON schema via the `format` field.
A judge that replies in prose needs parsing, and parse failures silently become ties.

## A working judge

`gpt-judge` (see `cli/judge.py` in each project) implements the above:

```bash
ollama serve &
gpt-judge --a best --b latest --cpu          # compare two checkpoints
gpt-judge --a latest --judge gemma3:12b      # a stronger judge if you have the RAM
gpt-judge --a latest --limit 20              # quick look
```

It generates both models' answers to the open-ended prompts, asks the judge each pair
in both orders, and reports win/loss/tie with the position-bias-disagreement rate
surfaced rather than hidden.

## Reliability — read before trusting a number

A 4B judge has known, measurable failure modes:

- **Verbosity bias.** Longer answers are preferred, largely independent of content.
  Your model's answers vary a lot in length, so this is a live confound.
- **Position bias.** Handled above by double-asking, but the disagreement rate is
  itself the signal — if the judge flips on 40% of pairs, its opinion is noise.
- **Self-preference.** Judges favour text that looks like their own output.
- **Small judges are weak at correctness.** A 4B model will confidently mark a wrong
  factual answer as good. Keep `gpt-score` for anything checkable; use the judge only
  where nothing checkable exists.

**Validate before you rely on it.** Take 20 pairs, judge them yourself blind, and
compare. If the judge agrees with you under ~70% of the time it is not measuring
quality, and a bigger judge (`gemma3:12b`, `qwen2.5:14b`) or a rubric change is needed
before the numbers mean anything.

---

# Part 2 — Distillation

## Three kinds, and only one is available to you

| kind | what transfers | viable here? |
|---|---|---|
| **Sequence-level** (synthetic data) | the teacher's *outputs*, used as training text | **yes** — the practical route |
| **Logit / soft-label** | the teacher's full next-token distribution | **no** — see below |
| **Feature / hidden-state** | intermediate activations | no — needs architecture alignment |

**Why logit distillation is closed off: the tokenizers do not match.** Soft-label
distillation trains the student to reproduce the teacher's probability distribution
over the *same vocabulary*. Gemma uses a ~256K SentencePiece vocabulary; your models
use GPT-2 BPE (50,257) or the 200m project's own 32,768. There is no faithful mapping
between them.

Adopting Gemma's vocabulary to fix that is worse than it sounds. At `E=896`, a 256K
embedding table is **229M parameters — larger than the entire 200M model**, before a
single transformer block. The tokenizer choice and the distillation method are
coupled, and this repo has already chosen a small vocabulary for good reasons
(`custom-gpt-200m/docs/ARCHITECTURE.md`).

## The throughput reality

Sequence-level distillation means generating text with the teacher and training on it.
Do the arithmetic before planning around it. `gemma3:4b` on an M4 Pro generates roughly
25–45 tokens/sec:

| target | tokens | time at ~35 tok/s |
|---|---|---|
| instruction-tuning set | 1M | ~8 hours |
| larger fine-tune set | 10M | ~3.3 days |
| small pretraining corpus | 1B | **~330 days** |

**So local distillation is viable for fine-tuning data and not for pretraining data.**
That is the single most useful line in this document. A 1–10M-token, high-quality,
teacher-generated instruction set is a genuinely good use of a weekend; generating a
2.5B-token pretraining corpus locally is not a plan.

For pretraining scale, use a corpus **someone else already distilled**:
`HuggingFaceTB/cosmopedia-v2` is 28B tokens of synthetic textbooks generated by
Mixtral, and it is already in this repo's data plan
(`custom-gpt-153m/DATASET.md`). That is distillation — just not one you pay the
compute for.

## What to actually generate

If you do run a local distillation pass, spend the tokens where the corpus is weakest
rather than on more general prose:

- **Worked reasoning traces** — problems *with* step-by-step solutions. This is what
  the GSM8K addition was reaching for and what the capability probes keep failing.
- **Instruction/response pairs** in your exact `User:`/`Assistant:` format, so the
  format signal is reinforced rather than diluted.
- **Clarifying-question examples** — vague request → assistant asks for specifics. The
  behaviour a base model cannot learn from web text.

Then fine-tune on it *after* pretraining, not mixed in — same argument as
`custom-gpt-153m/DATASET.md` makes for the chat corpus.

---

# Part 3 — Is distillation legal?

**Not legal advice.** Licenses change, and the current text governs, not this file.
Read the license shipped with the model you actually pull.

## The short answer for Gemma: yes, with conditions

Google's **Gemma Terms of Use** explicitly contemplate distillation. The terms define a
"Model Derivative" to include a model created by **transferring knowledge from a Gemma
model to another model**, including by distillation — and permit creating them. This is
unusually clear; most licenses leave it ambiguous.

The conditions that come with it, in substance:

1. **Pass-through.** If you distribute your distilled model, you must supply recipients
   with the Gemma Terms and the **Prohibited Use Policy**, and impose terms at least as
   restrictive.
2. **Notice.** Include a statement that the model is derived from Gemma.
3. **Use restrictions follow the derivative.** The Prohibited Use Policy binds your
   model too — you cannot distil away the restrictions.

Gemma is **not** OSI open source, despite "open weights". It is a custom license with
usage restrictions, which is why the obligations above exist at all.

## Where it is genuinely risky

| teacher | distillation |
|---|---|
| **Gemma** (open weights) | permitted, with the pass-through conditions above |
| **Llama 3.x** (open weights) | permitted; the community license adds naming and attribution requirements for derivatives |
| **Qwen, Mistral** (Apache-2.0 variants) | generally the most permissive — check the specific release |
| **OpenAI / Anthropic / Gemini *API* output** | **typically prohibited.** Provider terms generally forbid using outputs to develop competing models |

That last row is the real trap. The legal risk is not in distillation as a technique —
it is in *which teacher* you take output from. Open-weight models under licenses that
address derivatives are the safe path; API outputs from a commercial provider are the
risky one, and the restriction is contractual rather than about copyright.

## For what you are doing

Training a personal model locally, not distributing it, and not offering a competing
service is the lowest-risk case there is. The obligations that matter — notice,
pass-through, the prohibited-use policy — attach mainly to **distribution**. If you
later publish weights to Hugging Face, that is the moment to re-read the teacher's
license and add the required notices, and it is much easier to do that if you recorded
which teacher generated which data at the time.

**Practical habit worth adopting now:** record the teacher model, version, and date
alongside any generated dataset, the same way `.bin.json` records the tokenizer. You
cannot reconstruct provenance later, and provenance is exactly what a license
obligation asks you for.
