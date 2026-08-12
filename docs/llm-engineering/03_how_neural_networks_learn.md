# How Neural Networks Learn: Loss, Backpropagation, Gradient Descent

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 0 — Deep Learning
Foundations. Builds on [Chapter 1](01_neurons_layers_and_networks.md) (what a parameter
is) and [Chapter 2](02_parameters_vs_hyperparameters.md) (that parameters are what get
learned) — this chapter is the actual mechanism of *how* they get learned. Every training
run in this repo, and every LLM ever trained, runs this exact loop, over and over.

## In Plain English

Training a neural network is a repeated, four-step cycle: make a guess, measure how wrong
the guess was, figure out which direction each individual weight should nudge to make the
guess slightly less wrong next time, then actually nudge every weight a tiny bit in that
direction. Repeat this cycle millions of times, on millions of examples, and the weights
gradually settle into values that make good guesses. Nothing about this is more
mysterious than that — the apparent complexity of a trained LLM is the *result* of this
simple loop run an enormous number of times, not evidence of a fundamentally different
mechanism.

## The First-Principles Explanation

### The four-step loop

```
1. FORWARD PASS   — feed input through the network, get an output (a prediction)
2. LOSS           — compare the prediction to the correct answer with a single number:
                     how wrong was this guess?
3. BACKWARD PASS  — compute, for EVERY parameter in the network, how much that specific
                     parameter contributed to the loss, and in which direction changing
                     it would reduce the loss (this is BACKPROPAGATION)
4. OPTIMIZER STEP — actually nudge every parameter a small step in the loss-reducing
                     direction found in step 3 (this is GRADIENT DESCENT)

Repeat, using a new batch of training examples each time.
```

### Step 2: the loss function

A **loss function** converts "how wrong was the prediction" into one single number that
can be compared and minimized. For a language model predicting the next token
([Chapter 8](08_what_is_a_language_model.md)), the standard choice is **cross-entropy
loss** — a function from information theory that penalizes a model heavily for assigning
low probability to the *correct* next token, and rewards it for assigning that token high
probability. Lower loss = better predictions; the entire training process is, mechanically,
nothing more than an automated search for parameter values that make this one number as
small as possible, averaged across a huge number of examples.

### Step 3: backpropagation — the chain rule, applied systematically

**Backpropagation** (often shortened to "backprop") computes the **gradient** — for every
single parameter in the network, a number telling you: if I nudge this specific parameter
up slightly, does the loss go up or down, and by how much? It does this efficiently by
working backward from the loss, through the network, layer by layer, using the calculus
chain rule to combine each layer's local effect into the total effect on the final loss.

The crucial practical fact: this is a **fully automatic, mechanical process** in modern
deep learning frameworks — you never hand-write the calculus. You just call one function
(`.backward()` in PyTorch), and every parameter in the network gets its gradient computed
automatically, no matter how many layers deep the network is.

### Step 4: gradient descent — actually using the gradient to improve

Once every parameter has a gradient (a direction and magnitude for how it should change),
**gradient descent** is the rule for actually updating it:

```
new_weight = old_weight - (learning_rate × gradient)
```

The gradient points in the direction that would *increase* the loss (that's what its
sign means), so subtracting a fraction of it moves the weight in the loss-*decreasing*
direction — hence "descent." The **learning rate** (a hyperparameter, per
[Chapter 2](02_parameters_vs_hyperparameters.md)) controls how big a step to take: too
large and training can overshoot and become unstable; too small and training crawls.

## Grounded in This Repo's Code

Every one of these four steps appears explicitly,in order, in
[`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py)'s main training loop
(around line 549):

```python
# STEP 1: FORWARD PASS
xb, yb, mb = get_batch(train_tokens, train_target_mask, effective_context_length)
logits = model(xb)                                       # the model's guess

# STEP 2: LOSS
loss = masked_next_token_loss(logits, yb, mb, vocab_size) # cross-entropy, under the hood
                                                            # (masked_next_token_loss calls
                                                            # F.cross_entropy internally,
                                                            # line 137)

# STEP 3: BACKWARD PASS (backpropagation)
loss_to_backprop = loss / grad_accum_steps
loss_to_backprop.backward()                                # computes gradients for
                                                            # EVERY parameter, automatically

# STEP 4: OPTIMIZER STEP (gradient descent)
if ((step - start_step + 1) % grad_accum_steps == 0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # see below
    optimizer.step()                                        # actually update every weight
    optimizer.zero_grad(set_to_none=True)                   # reset gradients for next round
```

### Why `zero_grad()` is necessary: PyTorch accumulates gradients by default

A detail that surprises people new to this: calling `.backward()` doesn't *set* a
parameter's gradient, it *adds* to whatever gradient value was already there. If you
don't explicitly reset gradients to zero between steps, they'd keep accumulating across
every batch forever — `optimizer.zero_grad()` is the explicit reset, called once per
optimizer step. This is also *why* gradient accumulation (`grad_accum_steps = 32`) works
at all: by deliberately calling `.backward()` multiple times *without* zeroing in
between, the gradients from 32 small batches naturally sum together into the equivalent
of one large batch's gradient, before a single `optimizer.step()` applies them — a direct
consequence of this accumulation behavior, turned into a deliberate feature rather than a
bug to avoid.

### Why `clip_grad_norm_` exists: preventing exploding gradients

Occasionally, a batch produces an unusually large gradient — if applied directly, it
could push a parameter to a wildly wrong value in one step, destabilizing training.
`clip_grad_norm_(model.parameters(), max_norm=1.0)` rescales the *entire* set of
gradients (as one combined vector) if their overall magnitude exceeds `1.0`, without
changing their relative direction — a standard, cheap safeguard against this specific
instability.

### AdamW: not plain gradient descent, but a refined version of it

`optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)` (line 417)
uses **AdamW**, not the bare gradient-descent formula from the first-principles section
above. AdamW is the standard choice for Transformer training specifically because it
improves on plain gradient descent in two real ways:
- **Adaptive per-parameter learning rates** — it tracks a running estimate of each
  parameter's gradient history and adjusts that parameter's effective step size
  individually, rather than using one global learning rate for every parameter equally.
- **Decoupled weight decay** — `weight_decay=0.1` is a regularization term (pulling
  weights gently toward zero to reduce overfitting) applied in a way that's mathematically
  separated from the gradient-based update, a refinement over the older, coupled approach
  ("Adam" without the "W") that empirically trains better for Transformer-style models.

## Deep-Dive: What the Learning Rate Schedule Is Actually Doing

`lr_for_step` (line 148) implements **warmup followed by cosine decay** — worth
understanding as a real design decision, not an arbitrary curve shape:

```
Early steps (warmup):  learning rate ramps UP from near-zero to the full `lr`
                        Why: at the very start, the model's random initial weights
                        produce large, noisy gradients — taking full-size steps
                        immediately can destabilize training before it gets going.

Later steps (decay):   learning rate smoothly decreases from `lr` toward `min_lr`,
                        following a cosine curve
                        Why: as the model gets closer to a good solution, smaller
                        steps let it settle precisely rather than oscillating around
                        the target.
```

This warmup-then-decay shape (a hyperparameter *schedule*, chosen entirely in advance,
per [Chapter 2](02_parameters_vs_hyperparameters.md)) is close to universal across modern
Transformer training, not specific to this repo.

## Try It Yourself

- Add a `print(loss.item())` right after the loss is computed in `tiny_llm.py`'s training
  loop and watch it over the first few hundred steps — you should see it start high
  (the model's random initial weights predict badly) and generally decrease, directly
  observing gradient descent doing its job.
- Temporarily remove `optimizer.zero_grad(set_to_none=True)` from the loop (on a throwaway
  branch, not for real training) and observe training instability or divergence — a
  direct, hands-on demonstration of why that line is necessary.

## Common Misconceptions

- **"Backpropagation computes the loss."** No — the *forward pass* computes the loss;
  backpropagation computes the *gradient of the loss with respect to every parameter*, a
  completely different quantity used for the next step.
- **"Gradient descent finds the single best possible set of weights."** It doesn't
  guarantee global optimality — it's a local, iterative improvement process that
  typically converges to a *good* solution, not provably *the* best one; this is a real,
  accepted limitation of the entire field, not a flaw specific to any one implementation.
- **"A lower loss on the training data always means a better model."** Not necessarily —
  a model can achieve very low loss on training data while generalizing poorly to new
  data (overfitting) — this is exactly why `tiny_llm.py` tracks *both* `train_loss` and
  `test_loss` separately, a topic covered fully in
  [Chapter 15](00_roadmap.md#part-2--pretraining-building-a-model-from-zero).

## Practice Questions

1. Walk through the four-step loop for one training batch, naming which line of
   `tiny_llm.py`'s training loop corresponds to each step.
2. Why does PyTorch require an explicit `zero_grad()` call rather than automatically
   resetting gradients after every `optimizer.step()`? What deliberate feature does this
   default behavior make possible?
3. Explain, in your own words, what `clip_grad_norm_` protects against, and why it
   rescales the gradient vector as a whole rather than clipping each parameter's gradient
   independently.

## Key Terms

- **Forward pass**: running input through the network to produce a prediction.
- **Loss function**: a single number quantifying how wrong a prediction was; cross-entropy
  is the standard choice for next-token prediction.
- **Backpropagation**: the automatic, chain-rule-based algorithm computing the gradient of
  the loss with respect to every parameter.
- **Gradient**: for one parameter, the direction and magnitude that would most increase
  the loss if that parameter were nudged — gradient descent moves in the *opposite*
  direction.
- **Gradient descent**: the update rule that adjusts each parameter by a small step
  opposite its gradient, scaled by the learning rate.
- **Learning rate**: the hyperparameter controlling how large each gradient-descent step
  is.
- **AdamW**: an optimizer refining plain gradient descent with per-parameter adaptive
  learning rates and decoupled weight decay.
- **Gradient accumulation**: summing gradients across several forward/backward passes
  before one optimizer step, simulating a larger batch size.
- **Gradient clipping**: rescaling gradients to prevent unstably large parameter updates.
- **Warmup / cosine decay**: a common learning-rate schedule shape — ramping up early,
  smoothly decreasing later.
