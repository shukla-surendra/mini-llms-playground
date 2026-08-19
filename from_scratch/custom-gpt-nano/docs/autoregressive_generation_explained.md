# Autoregressive Text Generation — From Prompt to Next Token

## Overview

This document explains how the `generate()` function in the nanoGPT example generates text, from the initial prompt all the way through repeated next-token prediction.

The key idea is:

> The model does not generate a complete sentence in one operation. It repeatedly predicts **one next token**, appends that token to the sequence, and uses the updated sequence to predict the next token.

In this particular nanoGPT implementation, the tokenizer is **character-level**, so one token is effectively one character. To make the mechanics easier to understand, the examples below use a **word-level tokenizer**. The generation logic is the same.

---

# 1. What Does Autoregressive Mean?

"Autoregressive" means:

> The model's previous output becomes part of the input for the next prediction.

Imagine the prompt:

```text
What is the capital of France?
```

A simplified word-level model might generate:

```text
What is the capital of France?
                         ↓
                      predict
                         ↓
                       Paris
```

Then:

```text
What is the capital of France? Paris
                              ↓
                           predict
                              ↓
                              .
```

Then:

```text
What is the capital of France? Paris.
                                  ↓
                               predict
                                  ↓
                                The
```

So generation is:

```text
Prompt
  ↓
Predict one token
  ↓
Append token
  ↓
Predict next token
  ↓
Append token
  ↓
Repeat
```

---

# 2. What Is a Token?

A token is the unit the model actually processes.

Your current nanoGPT uses a **character tokenizer**.

For example:

```text
What
```

might become:

```text
W → token
h → token
a → token
t → token
```

So the model effectively generates:

```text
W → h → a → t → ...
```

Modern LLMs normally use subword tokenizers such as BPE-style tokenization. They may represent:

```text
What is the capital of France?
```

as something more like:

```text
What
 is
 the
 capital
 of
 France
?
```

or a different set of subword pieces.

The important concept is:

> **The model generates one tokenizer token at a time, not necessarily one word at a time.**

---

# 3. Start With the Prompt

Suppose we use a simplified word-level tokenizer.

Prompt:

```text
What is the capital of France?
```

Imagine the tokenizer converts it to:

```text
What      → 10
is        → 11
the       → 12
capital   → 13
of        → 14
France?   → 15
```

Therefore:

```python
idx = [10, 11, 12, 13, 14, 15]
```

In PyTorch, because we have one input sequence:

```text
idx.shape = (1, 6)
```

where:

```text
1 = batch size
6 = number of tokens
```

Your code creates this with:

```python
idx = torch.tensor(
    [encode_prompt(prompt, stoi)],
    dtype=torch.long,
    device=device,
)
```

---

# 4. The Generation Loop

The main generation mechanism is:

```python
for _ in range(max_new_tokens):
```

If:

```python
max_new_tokens = 5
```

the model will perform the next-token generation process up to five times.

Each iteration does:

```text
1. Select the available context
2. Run the model
3. Take the prediction at the last position
4. Apply temperature
5. Convert logits to probabilities
6. Sample one token
7. Append that token to the sequence
8. Repeat
```

---

# 5. Context Window

The first important line inside the loop is:

```python
idx_cond = idx[:, -model.cfg.block_size:]
```

This limits the context given to the model.

## What does `-block_size` mean?

Python slicing:

```python
idx[:, -8:]
```

means:

> Take the last 8 tokens.

For example:

```text
idx =
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

If:

```python
block_size = 8
```

then:

```python
idx_cond = idx[:, -8:]
```

becomes:

```text
[5, 6, 7, 8, 9, 10, 11, 12]
```

The first four tokens are not passed to the model.

## Why from the right?

Because the right side contains the **most recent tokens**.

The model uses the most recent available context to predict what comes next.

This creates a sliding context window:

```text
Full sequence:

[1][2][3][4][5][6][7][8][9][10][11][12]

                    ┌─────────────────────┐
                    │     model context   │
                    │ [5][6][7][8][9][10][11][12]
                    └─────────────────────┘
```

After another token is generated:

```text
[1][2][3][4][5][6][7][8][9][10][11][12][13]

                      ┌─────────────────────┐
                      │     model context   │
                      │ [6][7][8][9][10][11][12][13]
                      └─────────────────────┘
```

The window moves to the right.

---

# 6. Important: `idx` Grows, `idx_cond` Does Not

This is an important distinction.

When we do:

```python
idx = torch.cat([idx, next_id], dim=1)
```

the full `idx` sequence grows.

For example:

```text
Step 0:
idx = [1, 2, 3, 4, 5, 6]
shape = (1, 6)

Step 1:
idx = [1, 2, 3, 4, 5, 6, 7]
shape = (1, 7)

Step 2:
idx = [1, 2, 3, 4, 5, 6, 7, 8]
shape = (1, 8)
```

However, the model only receives:

```python
idx_cond = idx[:, -model.cfg.block_size:]
```

So if:

```python
block_size = 8
```

and the full sequence has 12 tokens:

```text
idx:
[1,2,3,4,5,6,7,8,9,10,11,12]
```

the model receives:

```text
idx_cond:
[5,6,7,8,9,10,11,12]
```

Therefore:

```text
idx
 ↓
Full generated sequence
 ↓
Can keep growing


idx_cond
 ↓
Recent context only
 ↓
Maximum size = block_size
```

---

# 7. Run the Model

Next:

```python
logits, _ = model(idx_cond)
```

Suppose:

```text
idx_cond.shape = (1, 6)
```

and the vocabulary contains 50,000 possible tokens.

The model might produce:

```text
logits.shape = (1, 6, 50000)
```

The dimensions mean:

```text
( batch, sequence_length, vocabulary_size )
```

So:

```text
1       = one sequence
6       = six input positions
50000   = 50,000 possible next tokens
```

The model produces predictions at **every position**.

Conceptually:

```text
What       → prediction
is         → prediction
the        → prediction
capital    → prediction
of         → prediction
France?    → prediction
```

But during generation we only need the prediction after the **last token**.

---

# 8. Take Only the Last Prediction

Your code:

```python
logits = logits[:, -1, :] / temperature
```

This line has two operations.

First:

```python
logits[:, -1, :]
```

Second:

```python
/ temperature
```

## Understanding `[:, -1, :]`

The model output has:

```text
(batch, sequence_position, vocabulary)
```

So:

```python
:
```

means:

> Take everything from the batch dimension.

```python
-1
```

means:

> Take the last sequence position.

```python
:
```

means:

> Take every vocabulary score.

Therefore:

```python
logits[:, -1, :]
```

means:

> **Give me the scores for every possible next token at the last position of the current context.**

For example:

```text
Input:

What is the capital of France?
                              ↑
                         current end
```

The model may produce scores like:

```text
Paris     → 8.2
London    → 4.1
Berlin    → 3.8
Madrid    → 3.2
Tokyo     → 2.1
...
```

These raw scores are called **logits**.

We don't use the predictions for earlier positions because they are not what we need right now.

---

# 9. Why Do We Divide by Temperature?

The code is:

```python
logits = logits[:, -1, :] / temperature
```

Temperature controls how sharp or random the eventual probability distribution will be.

## Temperature = 1.0

```text
8 / 1.0 = 8
4 / 1.0 = 4
2 / 1.0 = 2
```

The model's original distribution is unchanged.

## Temperature < 1

For example:

```text
temperature = 0.5
```

Then:

```text
8 / 0.5 = 16
4 / 0.5 = 8
2 / 0.5 = 4
```

The differences become larger.

After softmax, the highest-probability tokens become more dominant.

Therefore:

> Lower temperature → more predictable and less random.

## Temperature > 1

For example:

```text
temperature = 2
```

Then:

```text
8 / 2 = 4
4 / 2 = 2
2 / 2 = 1
```

The differences become smaller.

After softmax, probabilities become more spread out.

Therefore:

> Higher temperature → more random.

---

# 10. Convert Logits to Probabilities

Next:

```python
probs = F.softmax(logits, dim=-1)
```

Before softmax we have arbitrary scores:

```text
Paris     8.2
London    4.1
Berlin    3.8
Madrid    3.2
```

Softmax converts them into a probability distribution:

```text
Paris     70%
London     5%
Berlin     4%
Madrid     3%
...
```

The probabilities sum to approximately:

```text
1.0
```

or:

```text
100%
```

Now we can choose the next token based on these probabilities.

---

# 11. Select the Next Token

Your code:

```python
next_id = torch.multinomial(probs, num_samples=1)
```

This samples one token from the probability distribution.

For example:

```text
Paris     70%
London     5%
Berlin     4%
Madrid     3%
...
```

The selected token might be:

```text
Paris
```

Suppose its token ID is:

```text
20
```

Then:

```python
next_id = [20]
```

Notice that this is **not necessarily `argmax`**.

`argmax` would always choose:

```text
Paris
```

because it has the highest probability.

`multinomial` samples according to the distribution, so a lower-probability token can occasionally be selected.

This is one source of variation between generations.

---

# 12. Append the New Token

Now we reach:

```python
idx = torch.cat([idx, next_id], dim=1)
```

This is the key autoregressive step.

Suppose:

```text
idx =
[10, 11, 12, 13, 14, 15]
```

and:

```text
next_id =
[20]
```

Then:

```python
torch.cat([idx, next_id], dim=1)
```

produces:

```text
[10, 11, 12, 13, 14, 15, 20]
```

So:

```python
idx = ...
```

updates the full sequence.

Conceptually:

```text
Before:

What is the capital of France?

Model predicts:

Paris

After append:

What is the capital of France? Paris
```

---

# 13. Why `dim=1`?

The tensor is shaped approximately like:

```text
(batch, sequence)
```

For example:

```text
idx.shape = (1, 6)
next_id.shape = (1, 1)
```

We want:

```text
(1, 6)
+
(1, 1)
=
(1, 7)
```

So we concatenate along dimension 1, the sequence dimension:

```python
torch.cat([idx, next_id], dim=1)
```

The batch dimension remains unchanged.

---

# 14. The Next Iteration

Now the loop starts again.

The updated sequence is:

```text
What is the capital of France? Paris
```

The model receives this new context:

```python
logits, _ = model(idx_cond)
```

and predicts the next token.

Suppose it predicts:

```text
.
```

Then we append it:

```text
What is the capital of France? Paris.
```

Then the next iteration uses:

```text
What is the capital of France? Paris.
```

and predicts another token.

---

# 15. Full Example

Using a simplified word-level tokenizer, imagine:

```text
Initial prompt:

What is the capital of France?
```

### Iteration 1

```text
Input:
What is the capital of France?

Model prediction:
Paris

Append:
What is the capital of France? Paris
```

### Iteration 2

```text
Input:
What is the capital of France? Paris

Model prediction:
.

Append:
What is the capital of France? Paris.
```

### Iteration 3

```text
Input:
What is the capital of France? Paris.

Model prediction:
The

Append:
What is the capital of France? Paris. The
```

### Iteration 4

```text
Input:
What is the capital of France? Paris. The

Model prediction:
capital

Append:
What is the capital of France? Paris. The capital
```

### Iteration 5

```text
Input:
What is the capital of France? Paris. The capital

Model prediction:
of

Append:
What is the capital of France? Paris. The capital of
```

The model is not generating the whole answer at once.

It is repeatedly doing:

```text
Current context
      ↓
     GPT
      ↓
Next-token probability distribution
      ↓
   sample one
      ↓
Append token
      ↓
Updated context
      ↓
     GPT again
```

---

# 16. Complete Code Flow

Here is the important part of your original function:

```python
idx = torch.tensor(
    [encode_prompt(prompt, stoi)],
    dtype=torch.long,
    device=device,
)

for _ in range(max_new_tokens):

    idx_cond = idx[:, -model.cfg.block_size:]

    logits, _ = model(idx_cond)

    logits = logits[:, -1, :] / temperature

    probs = F.softmax(logits, dim=-1)

    next_id = torch.multinomial(
        probs,
        num_samples=1
    )

    idx = torch.cat(
        [idx, next_id],
        dim=1
    )
```

Read it as English:

```text
Create tokenized prompt.

REPEAT:

    Keep only the latest allowed context.

    Send that context through GPT.

    From all model predictions,
    take only the prediction at the last position.

    Adjust it using temperature.

    Convert scores into probabilities.

    Sample one token.

    Append that token to the full sequence.

UNTIL max_new_tokens is reached.
```

---

# 17. The Most Important Mental Model

Keep these three variables separate in your mind:

## `idx`

```text
FULL SEQUENCE
```

It grows:

```text
[What is the capital of France?]
[What is the capital of France? Paris]
[What is the capital of France? Paris .]
[What is the capital of France? Paris . The]
...
```

## `idx_cond`

```text
WHAT THE MODEL CURRENTLY SEES
```

It is limited by:

```python
model.cfg.block_size
```

Example:

```text
idx:
[1 2 3 4 5 6 7 8 9 10]

block_size = 5

idx_cond:
[6 7 8 9 10]
```

## `next_id`

```text
THE ONE NEW TOKEN THE MODEL JUST PREDICTED
```

Example:

```text
next_id = [20]
```

Then:

```text
idx + next_id
      ↓
new idx
```

---

# 18. One Diagram to Remember

```text
                    FULL SEQUENCE (`idx`)
                           │
                           │ grows every step
                           ▼
        [What][is][the][capital][of][France?][Paris][.]
                           │
                           │
                           ▼
             Take last `block_size` tokens
                           │
                           ▼
                    `idx_cond`
                           │
                           ▼
                         GPT
                           │
                           ▼
                logits for every position
                           │
                           ▼
                  take `[:, -1, :]`
                           │
                           ▼
             next-token logits only
                           │
                           ▼
                divide by temperature
                           │
                           ▼
                       softmax
                           │
                           ▼
                  probability distribution
                           │
                           ▼
                  `torch.multinomial`
                           │
                           ▼
                       `next_id`
                           │
                           ▼
                 append to `idx`
                           │
                           └───────────────┐
                                           │
                                           ▼
                                      Repeat
```

---

# 19. What the Model Actually Learned

This is perhaps the most important conceptual point.

The model was trained to perform:

```text
Given previous tokens → predict the next token
```

It was not separately trained with a magical:

```text
generate_complete_answer()
```

operation.

During training, it learns many examples such as:

```text
The capital of France → is
The capital of France is → Paris
The capital of France is Paris → .
```

At inference time, we repeatedly use that ability:

```text
Prompt
  ↓
Predict token
  ↓
Add token
  ↓
Predict token
  ↓
Add token
  ↓
Predict token
  ↓
...
```

Therefore:

> **A long generated response is the result of repeatedly solving the next-token prediction problem.**

---

# 20. One Important Difference: Training vs Generation

During training, the model can receive a complete sequence at once.

For example:

```text
What is the capital of France? Paris.
```

It can calculate predictions for multiple positions in parallel.

Conceptually:

```text
What → predict "is"
is → predict "the"
the → predict "capital"
capital → predict "of"
of → predict "France?"
France? → predict "Paris"
Paris → predict "."
```

During generation, the future tokens are unknown.

So generation must proceed sequentially:

```text
Prompt
  ↓
predict token 1
  ↓
append token 1
  ↓
predict token 2
  ↓
append token 2
  ↓
predict token 3
  ↓
...
```

This sequential dependency is why autoregressive generation can be computationally expensive.

---

# 21. Why Modern LLMs Use KV Cache

Your simple code does this on every iteration:

```python
logits, _ = model(idx_cond)
```

That means the model repeatedly processes the context.

For example:

```text
Step 1:
A B C D

Step 2:
A B C D E

Step 3:
A B C D E F

Step 4:
A B C D E F G
```

A production inference engine can cache intermediate attention information from previous tokens using a **KV cache**.

Conceptually:

```text
Without KV cache:

Process A B C D
Process A B C D E
Process A B C D E F
Process A B C D E F G


With KV cache:

Process A B C D
Cache results

Then process only E
Reuse cache

Then process only F
Reuse cache

Then process only G
Reuse cache
```

This is one of the major optimizations used by real LLM serving systems.

Your simple nanoGPT implementation intentionally keeps the generation loop easy to understand rather than implementing production KV caching.

---

# 22. Final Summary

The complete generation process is:

```text
1. Tokenize the prompt
        ↓
2. Store tokens in `idx`
        ↓
3. Keep the latest `block_size` tokens as `idx_cond`
        ↓
4. Run GPT
        ↓
5. Get logits for every position
        ↓
6. Select only the last position:
       logits[:, -1, :]
        ↓
7. Divide logits by temperature
        ↓
8. Apply softmax
        ↓
9. Get probabilities
        ↓
10. Sample one `next_id`
        ↓
11. Append it:
       idx = torch.cat([idx, next_id], dim=1)
        ↓
12. Repeat
```

The three lines that are most important to remember are:

```python
idx_cond = idx[:, -model.cfg.block_size:]
```

**What context does the model see?**

```python
logits = logits[:, -1, :] / temperature
```

**What should the next token be, and how random should the choice be?**

```python
idx = torch.cat([idx, next_id], dim=1)
```

**Add the model's prediction to the sequence so it can influence the next prediction.**

That three-step mental model is enough to understand the core of autoregressive generation:

```text
SEE CONTEXT
    ↓
PREDICT NEXT TOKEN
    ↓
APPEND TOKEN
    ↓
REPEAT
```
