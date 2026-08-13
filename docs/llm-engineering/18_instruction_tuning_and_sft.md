# Instruction Tuning & Supervised Fine-Tuning (SFT)

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 3 — Fine-Tuning. Builds on
[Chapter 16](16_fine_tuning_landscape.md) (PEFT/LoRA as the *how*) and
[Chapter 17](17_lora_and_qlora.md) (the mechanism) — this chapter is the *what you
actually train on*, a separate, orthogonal question from how weights get updated.

## In Plain English

Instruction tuning teaches a model to behave like an assistant — follow an instruction,
answer a question, hold a conversation — rather than just continuing whatever text it's
given. The mechanism doing the teaching is called **SFT (Supervised Fine-Tuning)**, and
the genuinely important thing to understand is: **it's the exact same next-token-
prediction objective from pretraining** ([Chapter 8](08_what_is_a_language_model.md)) —
nothing new mathematically. What changes is the *data*: instead of raw web text, SFT
trains on curated instruction/response pairs, formatted consistently.

## The First-Principles Explanation

### SFT's training data: structured conversation pairs, not raw text

A pretraining example is just a chunk of arbitrary text. An SFT example is a structured
exchange:

```
{"role": "user", "content": "Explain LoRA in simple terms."}
{"role": "assistant", "content": "LoRA freezes the original model and trains two small
matrices that get added on top..."}
```

Real, large SFT datasets — like `HuggingFaceH4/ultrachat_200k`, used in
[`fine_tuning/tinyllama-1.1b-lora/`](../../fine_tuning/tinyllama-1.1b-lora/) — contain
tens or hundreds of thousands of these exchanges, covering a wide range of tasks and
styles, so the model learns the general *pattern* of being a helpful assistant, not just
memorized answers to specific questions.

### The chat template: turning structured data into the raw text a model actually trains on

The model still only ever sees plain text (per [Chapter 9](09_tokenization.md)) — a "chat
template" is the exact string format that structured conversation gets converted into,
consistently, so the model can learn to recognize where a turn starts/ends and who's
speaking:

```python
# fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py
return tokenizer.apply_chat_template(
    example["messages"], tokenize=False, add_generation_prompt=False,
)
```

`apply_chat_template` isn't a generic function — it applies the *specific* format the
target model (`TinyLlama-1.1B-Chat`, here) was itself originally trained with, stored as
part of the tokenizer. This detail matters: using the wrong chat template (or none at
all) during fine-tuning teaches the model an inconsistent format from what it already
learned during its own original instruction tuning, actively working against the
fine-tune rather than building on it.

### The loss objective: still cross-entropy, still next-token prediction

Nothing about the loss function changes from pretraining
([Chapter 3](03_how_neural_networks_learn.md#step-2-the-loss-function)) — SFT still
minimizes cross-entropy loss on next-token prediction. The only genuine design choice is
**which tokens' loss actually counts** — and this repo's two fine-tuning projects make
*different* choices here, both real and defensible, worth understanding precisely.

## Grounded in This Repo's Code: Two Different Real Approaches to Loss Masking

### `custom-gpt-153m`: explicit assistant-only masking

```python
# from_scratch/custom-gpt-153m/tiny_llm.py
def encode_with_assistant_mask(tokenizer, text):
    ...
    is_assistant = current_role == "assistant"
    mask.extend([1 if is_assistant else 0] * len(line_ids))

def masked_next_token_loss(logits, targets, target_mask, vocab_size):
    token_losses = F.cross_entropy(..., reduction="none")
    return (token_losses * target_mask).sum() / mask_sum
```

Loss is computed for **every** token, but then multiplied by a mask that zeroes out
everything except the assistant's own response tokens before averaging — the model still
*sees* the system/user turns as input context, but is never penalized (or rewarded) for
how well it could predict them, only for its actual responses.

### `tinyllama-1.1b-lora`: loss on the entire sequence

```python
# fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

Hugging Face's standard causal-LM collator (`mlm=False` means "causal, not masked-language-
modeling" — see [Chapter 6](06_nlp_architecture_landscape.md#the-transformers-three-architectural-sub-families)'s
encoder/decoder distinction) computes loss across the **entire** tokenized sequence by
default here — including the user's instruction tokens, not just the assistant's
response. The model is being trained, in part, to predict the *user's* turn as well as
its own.

### Why this difference is real and worth understanding, not a bug

Both are legitimate, commonly-seen choices in real SFT implementations:

- **Assistant-only masking** (`custom-gpt-153m`'s approach) is more precisely targeted —
  every bit of gradient signal is spent specifically on improving response quality,
  arguably a "purer" instruction-tuning objective.
- **Whole-sequence loss** (`tinyllama-1.1b-lora`'s approach, and the more common default
  in many popular SFT libraries/scripts) is simpler to implement correctly, and there's a
  reasonable argument that learning to model the *user's* side of a conversation well
  also helps the model understand context and instructions better — a genuinely debated
  point in the field, not a settled one.

**The concrete, checkable difference this produces**: with whole-sequence loss, if a
training example's instruction is unusually long relative to its response, a large
fraction of that example's gradient signal goes toward predicting the instruction text
rather than the response — a real, measurable dilution of the "teach the model to
respond well" signal that assistant-only masking avoids entirely.

## Deep-Dive: Why SFT Alone Often Isn't the Final Step for Production Chat Models

SFT teaches a model to produce *plausible* assistant-style responses, matching the
training data's patterns. It does **not**, by itself, teach the model which of several
plausible responses is actually *better* — more helpful, more accurate, safer. That's a
genuinely different training signal, requiring comparisons between outputs rather than a
single "correct" target sequence to predict — the subject of
[Chapter 19](19_rlhf_and_dpo.md) (RLHF and DPO). Production chat models (including the
`-Chat` suffix models like `TinyLlama-1.1B-Chat` that this repo's LoRA project fine-tunes
further) have typically already been through both stages before you ever download them —
worth knowing so "SFT" and "the finished chat model's full training pipeline" aren't
conflated as the same thing.

## Trade-offs

| Choice | Upside | Cost |
|---|---|---|
| Assistant-only loss masking | More precisely targeted training signal | Requires explicit role-tracking code (`encode_with_assistant_mask`) rather than a standard library default |
| Whole-sequence loss | Simpler, uses standard library collators directly | Dilutes gradient signal on examples with long instructions relative to responses |
| Fine-tuning an already-instruction-tuned model (`-Chat` variant) | Builds on existing instruction-following competence | Risk of the chat-template mismatch failure mode below if not handled carefully |
| Fine-tuning a base (non-chat) model with SFT | Full control over the instruction-following behavior taught | Needs a genuinely representative, sufficiently large SFT dataset to teach the behavior from nothing |

## Failure Modes to Raise Proactively

- **Using the wrong (or no) chat template when fine-tuning an already-chat-tuned model**
  — actively conflicts with formatting the model already learned, degrading rather than
  improving its instruction-following behavior.
- **Assuming whole-sequence loss and assistant-only loss produce equivalent results** —
  as shown above, they don't when instruction length varies significantly across
  training examples; worth checking which a given script/library actually does before
  assuming.

## Try It Yourself

- Compare [`custom-gpt-153m/tiny_llm.py`'s `encode_with_assistant_mask`](../../from_scratch/custom-gpt-153m/tiny_llm.py)
  against [`tinyllama-1.1b-lora/train_tinyllama_lora.py`'s `DataCollatorForLanguageModeling`
  usage](../../fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py) side by side, and
  identify exactly which lines are responsible for the loss-masking difference described
  in this chapter.

## Practice Questions

1. Why does SFT use the exact same loss function as pretraining, and what's the only
   thing that genuinely changes between the two stages?
2. A training example has a 200-token instruction and a 20-token response. Explain,
   concretely, how whole-sequence loss and assistant-only-masked loss would treat this
   example differently.
3. Why would using a different chat template than the one a model was originally
   instruction-tuned with actively hurt fine-tuning results, rather than just being a
   neutral formatting choice?

## Key Terms

- **SFT (Supervised Fine-Tuning)**: fine-tuning on curated instruction/response pairs
  using the standard next-token-prediction loss.
- **Chat template**: the exact text format (role markers, separators) a specific model
  expects conversational data to be encoded as.
- **Loss masking**: selectively excluding certain tokens (e.g., the instruction/prompt)
  from contributing to the training loss.
- **Base model vs. chat/instruct model**: a model that has only seen pretraining, versus
  one that has already been through SFT (and often further post-training) —
  [Chapter 7](07_history_how_we_got_here.md#generation-6-from-predicts-text-to-follows-instructions-2022-present).
