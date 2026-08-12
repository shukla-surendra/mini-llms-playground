# Parameters vs. Hyperparameters

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 0 — Deep Learning
Foundations. Builds on [Chapter 1](01_neurons_layers_and_networks.md)'s introduction of
weights/biases as parameters — this chapter draws the precise line between what a model
*learns* and what a human *decides*, since the two categories get conflated constantly in
casual conversation and behave completely differently.

## In Plain English

**Parameters** are the numbers the model figures out for itself during training — the
weights and biases from [Chapter 1](01_neurons_layers_and_networks.md). **Hyperparameters**
are the settings *you* choose before training even starts, and that training itself never
changes — things like how many layers to stack, how big each layer is, how fast the model
should learn. A useful shorthand: parameters are the answer; hyperparameters are the
question you set up before asking for an answer.

## The First-Principles Explanation

### The precise distinction

| | Parameters | Hyperparameters |
|---|---|---|
| Who sets it | Learned automatically via training ([Chapter 3](03_how_neural_networks_learn.md)) | Chosen by a human, ahead of training |
| Does it change during training? | Yes — that's the entire point of training | No — fixed for the duration of one training run |
| Example | The specific numeric value of one weight in a `Linear` layer | How many layers the network has |
| How many are there | Millions to billions, for any real model | A handful to a few dozen |

**The test that resolves any ambiguous case**: does gradient descent
([Chapter 3](03_how_neural_networks_learn.md)) directly compute and update this value
during training? If yes, it's a parameter. If a human chose it before training started,
and training doesn't touch it, it's a hyperparameter — even if, confusingly, its *value*
changes *during* a run according to a schedule (like a learning rate that decreases over
time) — the schedule itself, and the fact that it changes, was still a human decision made
in advance, not something gradient descent discovered.

## Grounded in This Repo's Code

[`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py)'s `# -------- CONFIG
--------` section (starting around line 45) is, in its entirety, a list of
hyperparameters — every single value there is chosen by whoever runs the script, before
training starts, and none of them are updated by the training loop itself:

```python
context_length = 1024      # how many tokens of context the model can see at once
embed_size = 768           # the width of every layer (Chapter 1's "how many neurons")
num_heads = 12              # how many attention heads (Chapter 10)
num_layers = 16             # how many GPTBlocks are stacked
dropout = 0.1                # a regularization strength (see below)
batch_size = 1                # how many sequences processed per forward pass
grad_accum_steps = 32        # how many micro-batches before one optimizer update
lr = 2e-4                     # the starting learning rate
min_lr = 2e-5                 # the learning rate floor after decay
steps = 1000000               # how many training steps to run
```

Contrast this with the model's actual **parameters**, which are never listed by hand
anywhere in the code — they're created automatically, with random initial values, the
moment `TinyGPT(...)` is instantiated (`model = TinyGPT(...)`, near line 409), and are
*only* ever changed by `optimizer.step()` (line 560) — the literal moment gradient
descent updates them, covered fully in [Chapter 3](03_how_neural_networks_learn.md).

This is the exact line that draws the boundary: everything in the CONFIG section above
`model = TinyGPT(...)` is a hyperparameter, decided before that line runs; everything
inside `model.state_dict()` after that line is a parameter, and stays a parameter no
matter how many training steps modify its value.

### `embed_size`, `num_heads`, `num_layers` are hyperparameters that determine *how many*
### parameters exist

A subtlety worth being precise about: hyperparameters like `embed_size` and `num_layers`
don't directly become parameters — but they determine the **shape** (and therefore the
count) of the parameters that *do* get created. Change `num_layers` from 16 to 32, and
you haven't changed any weight's value — you've changed how many `GPTBlock`s worth of
weights exist at all. This is exactly the relationship the parameter-count breakdown in
[`from_scratch/custom-gpt-153m/README.md`](../../from_scratch/custom-gpt-153m/README.md#parameter-count-current-config)
walks through: `embed_size`, `num_layers`, and `vocab_size` (from
[Chapter 9](09_tokenization.md)) are the hyperparameters that arithmetic combines to
produce the 152,791,296 parameter count.

## Deep-Dive: Why This Distinction Actually Matters, Practically

This isn't a pedantic vocabulary distinction — it has real, practical consequences:

- **You can't "fix" a bad hyperparameter by training longer.** If `embed_size` is set too
  small for the task, no amount of additional training steps will let the model represent
  what a larger `embed_size` could — the model's *capacity* is a hyperparameter decision,
  made before training, that training cannot undo.
- **Hyperparameters require their own search process** ([Chapter 4](04_hyperparameter_tuning.md)),
  because there's no gradient telling you "increase `num_layers`" the way there's a
  gradient telling every weight which direction to move.
- **Dropout is a hyperparameter that intentionally interferes with parameters during
  training.** `dropout = 0.1` in this repo means: during training, randomly zero out 10%
  of activations at each forward pass, forcing the network to not over-rely on any single
  neuron. This is a hyperparameter (`0.1`, chosen ahead of time, never changed by
  training) that shapes *how* the parameters end up learning — a good example of a
  hyperparameter whose whole purpose is to influence the parameter-learning process
  itself, without being a parameter.

## Try It Yourself

- In `tiny_llm.py`, change `num_layers` from 16 to 8, and re-run
  `sum(p.numel() for p in model.parameters())` — confirm the total parameter count drops
  roughly in proportion (per-block parameter count × the new layer count), directly
  observing a hyperparameter change reshaping how many parameters exist.
- Look at `make_checkpoint_payload` (line 360) — notice it saves both the model's
  `state_dict` (parameters) *and* `embed_size`, `num_heads`, `num_layers`, `context_length`
  (hyperparameters) into the same checkpoint file. This is precisely *why* — the resume
  logic (line 423 onward) explicitly checks that a checkpoint's saved hyperparameters
  match the current run's config before allowing a resume, since loading parameters
  trained under a *different* set of hyperparameters would silently produce a broken
  model (shape mismatches at best, garbage at worst).

## Common Misconceptions

- **"A model with more hyperparameters is more powerful."** Hyperparameter *count* isn't
  the lever — their *values* (particularly ones like `embed_size`/`num_layers` that
  determine parameter count) are what actually affects capacity, and there are always
  only a handful of hyperparameters regardless of model scale.
- **"The learning rate schedule (`lr_for_step`, changing value over training) means the
  learning rate is a parameter, not a hyperparameter."** The *schedule itself* — warmup
  length, decay shape, min/max values — is entirely decided in advance by the
  hyperparameters `lr`, `min_lr`, and the formula in `lr_for_step` (line 148); no gradient
  ever adjusts it. It changing value during training doesn't make it learned.
- **"Once training starts, hyperparameters can't matter anymore."** They matter for the
  entire run — `dropout`, `grad_accum_steps`, and the learning rate schedule are all
  hyperparameters actively shaping every single training step, not just the initial setup.

## Practice Questions

1. `grad_accum_steps = 32` never appears inside `model.state_dict()`. Explain, precisely,
   why it's a hyperparameter and not a parameter, using this chapter's "who computes it"
   test.
2. Why does the resume-checkpoint logic in `tiny_llm.py` need to verify that saved
   hyperparameters match the current config before loading saved parameters?
3. Give an example of a hyperparameter that determines *how many* parameters exist,
   versus one that doesn't change parameter count but still affects training. Use two
   specific values from `tiny_llm.py`'s CONFIG section.

## Key Terms

- **Parameter**: a weight or bias, learned automatically by gradient descent during
  training.
- **Hyperparameter**: a setting chosen by a human before training, left unchanged by the
  training process itself (even if its *value* follows a predetermined schedule).
- **Model capacity**: roughly, how much a model's architecture (determined by
  hyperparameters like `embed_size`/`num_layers`) is theoretically capable of
  representing — a ceiling training cannot raise on its own.
- **Dropout**: a regularization hyperparameter that randomly zeros activations during
  training to reduce over-reliance on specific neurons.
