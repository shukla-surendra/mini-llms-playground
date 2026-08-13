# The Fine-Tuning Landscape: Full FT, PEFT, Prompting, and RAG

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 3 — Fine-Tuning. Builds on
[Chapter 7](07_history_how_we_got_here.md#generation-6-from-predicts-text-to-follows-instructions-2022-present)'s
observation that a raw pretrained model can predict text well without being useful as an
assistant. This chapter is the map of every real technique that closes that gap — what
each one actually changes, and how to pick between them.

## In Plain English

"Fine-tuning" gets used loosely to mean almost any way of making a pretrained model more
useful for a specific job. In reality there are at least four genuinely different levers,
and they're not interchangeable — they trade off compute cost, how much data you need,
and how permanently the change sticks, in very different ways. Picking the wrong one for
your actual constraint (e.g., reaching for full fine-tuning when you have no GPU) is a
real, common, avoidable mistake.

## The First-Principles Explanation

### The four levers, precisely distinguished

```
1. PROMPTING (no training at all)
   Change: nothing about the model. You just phrase the input differently
   (instructions, examples, formatting) to steer behavior.
   Cost: ~zero compute, no data collection.
   Permanence: none — every conversation starts from zero, you re-supply
   the steering every time.

2. RAG — Retrieval-Augmented Generation (no training, but adds infrastructure)
   Change: nothing about the model's weights. Instead, relevant documents
   are retrieved (via a vector database / search) and inserted into the
   prompt before generation, giving the model access to information it
   was never trained on.
   Cost: no training compute, but real infrastructure (an index, a
   retrieval pipeline) to build and operate.
   Permanence: none in the weights — swap the document index and the
   model's effective "knowledge" changes instantly, no retraining.

3. FULL FINE-TUNING (every weight updated)
   Change: literally every parameter in the model, via the same gradient-
   descent mechanism as pretraining (Chapter 3), just starting from the
   pretrained checkpoint instead of random initialization — this is
   exactly what "continued pretraining"
   (../../from_scratch/tinystories-gpt-6m/docs/CONTINUING_TRAINING_ON_NEW_DATA.md)
   already covered, applied deliberately for behavior-shaping rather than
   just more of the same.
   Cost: the highest of any option here — needs to hold gradients and
   optimizer state for every parameter, the full memory/compute budget
   from Chapter 3, scaled to the whole model.
   Permanence: fully baked into the weights — no extra infrastructure
   needed at inference time, but a new full copy of the model exists.

4. PEFT — Parameter-Efficient Fine-Tuning (a small number of NEW weights
   trained, the rest FROZEN)
   Change: the base model's weights are frozen (no gradients computed for
   them at all); a small number of new, additional parameters are trained
   instead, inserted into the model in a way that shapes its behavior.
   LoRA (Chapter 17) is the dominant technique in this category.
   Cost: dramatically lower than full fine-tuning — gradients and
   optimizer state are only needed for the small new parameter set, not
   the whole model.
   Permanence: the new parameters (an "adapter") can be kept as a small,
   separate file, loaded alongside the frozen base model — or merged in,
   at your choice.
```

### Why this is a spectrum of "how much you touch the model," not a ranked list

These four aren't tiers where one is strictly "better" — they solve different problems:

- **Need the model to know today's news, or your company's private documents?** RAG —
  no amount of fine-tuning teaches a model facts efficiently the way giving it the actual
  document to read does, and fine-tuned-in facts go stale the moment the underlying
  information changes.
- **Need a specific output format or tone, applied consistently?** Prompting first
  (cheapest), PEFT if prompting isn't reliable enough.
- **Need the model to reliably follow instructions or converse at all** (the base→chat
  gap from [Chapter 7](07_history_how_we_got_here.md#generation-6-from-predicts-text-to-follows-instructions-2022-present))?
  This needs actual weight updates — PEFT or full fine-tuning, covered by
  [Chapter 18](18_instruction_tuning_and_sft.md)'s SFT objective.
- **Need to adapt an already-capable model to a narrower domain/style, on limited
  hardware?** PEFT (LoRA) — the sweet spot this curriculum's own fine-tuning projects
  both use.

## Grounded in This Repo's Code

Both fine-tuning projects in this repo are **PEFT**, specifically **LoRA**, not full
fine-tuning — and the code itself makes the "which weights are frozen" distinction
explicit and checkable:

```python
# fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py
base_model = AutoModelForCausalLM.from_pretrained(args.model_id, ...)
base_model.config.use_cache = False
base_model.gradient_checkpointing_enable()

lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, ...)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()   # <- prints exactly how few parameters actually train
```

`get_peft_model` is the exact mechanical step that implements the PEFT category above:
it freezes every one of `base_model`'s original weights and inserts new, small,
trainable LoRA matrices alongside them — `model.print_trainable_parameters()` reports
something like "trainable params: 4.5M || all params: 1.1B || trainable%: 0.4%" when run,
a direct, numeric confirmation of exactly how much of the "full fine-tuning" cost this
approach avoids. The full mechanism of what those inserted matrices actually are and how
they work is [Chapter 17](17_lora_and_qlora.md).

## Deep-Dive: Why PEFT Specifically Matters for This Curriculum's "No GPU" Constraint

Recall from [Chapter 3](03_how_neural_networks_learn.md#step-4-gradient-descent--actually-using-the-gradient-to-improve):
training needs to store, per trainable parameter, not just the parameter itself but its
gradient and (for AdamW) two additional optimizer state values — roughly 4x a parameter's
own memory footprint, minimum. For a 1.1B-parameter model like `TinyLlama` (used in this
repo's `fine_tuning/tinyllama-1.1b-lora/`), full fine-tuning would need memory for
**every one** of those 1.1B parameters' gradients and optimizer state — well beyond what
a MacBook's unified memory comfortably handles alongside everything else running. PEFT/
LoRA needs that same 4x overhead only for the small inserted adapter matrices (often
under 1% of the base model's parameter count, per the real `print_trainable_parameters()`
output above) — this is *the* mechanism-level reason PEFT is what makes fine-tuning a
billion-parameter-class model on a laptop feasible at all, not just a nice-to-have
efficiency gain.

## Trade-offs

| Lever | Upside | Cost |
|---|---|---|
| Prompting | Instant, no training, fully reversible | Unreliable for complex/consistent behavior changes; costs tokens every request |
| RAG | Model gains access to information beyond its training data, updatable instantly | Real infrastructure to build/operate (retrieval index); doesn't change the model's underlying behavior/style |
| Full fine-tuning | Maximum capacity to change behavior; no extra inference-time infrastructure | Highest compute/memory cost; produces a full new model copy per fine-tune |
| PEFT (LoRA) | Dramatically lower compute/memory; small, swappable adapter files | Slightly less expressive than full fine-tuning for very large behavior changes (a real, usually acceptable trade at typical adaptation scale) |

## Try It Yourself

- Run `model.print_trainable_parameters()` (already called in
  [`../../fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py`](../../fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py))
  during an actual training run and note the exact trainable-parameter percentage — this
  turns "PEFT trains far fewer parameters" from an abstract claim into a real, observed
  number for this specific model and LoRA configuration.

## Common Misconceptions

- **"Fine-tuning is always the first tool to reach for when a model isn't doing what you
  want."** Often the wrong first move — if the actual problem is missing knowledge (not
  missing behavior), RAG solves it faster and more reliably than any amount of
  fine-tuning; if it's a formatting/tone issue, prompting alone frequently suffices.
- **"PEFT is a worse, cut-down version of full fine-tuning."** It's a different
  trade-off, not a strictly worse one — for adapting an already-capable pretrained model
  (the common case), PEFT's dramatically lower cost usually isn't paired with a
  meaningful quality loss for the task.
- **"RAG and fine-tuning are competing techniques — you pick one."** They're frequently
  used together: fine-tuning shapes *how* a model behaves/responds, RAG supplies *what*
  specific information it has access to — genuinely different, complementary jobs.

## Practice Questions

1. A team wants their model to always cite internal company documentation accurately,
   including documents added last week. Which lever is the right primary tool, and why
   would fine-tuning alone be a poor fit here?
2. Why does PEFT's memory advantage over full fine-tuning come specifically from
   optimizer state and gradients, not just from "training fewer numbers"?
3. `model.print_trainable_parameters()` reports 0.4% of parameters are trainable in a
   LoRA setup. Explain, using Chapter 3's 4x-overhead figure, roughly how much smaller
   the training memory footprint is compared to full fine-tuning the same base model.

## Key Terms

- **Full fine-tuning**: updating every parameter of a pretrained model via continued
  gradient descent.
- **PEFT (Parameter-Efficient Fine-Tuning)**: freezing the base model and training a
  small number of new parameters instead — LoRA is the dominant current technique.
- **RAG (Retrieval-Augmented Generation)**: inserting retrieved documents into the prompt
  at inference time, rather than training anything into the model.
- **Adapter**: the small set of new, trainable parameters a PEFT method inserts —
  storable and shareable separately from the frozen base model.
