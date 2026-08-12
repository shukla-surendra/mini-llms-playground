# Hyperparameter Tuning: What to Tune and How

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 0 — Deep Learning
Foundations. Builds directly on [Chapter 2](02_parameters_vs_hyperparameters.md)'s core
fact: hyperparameters aren't learned by gradient descent, so *something else* has to
choose good values for them. This chapter is that process.

## In Plain English

Since gradient descent ([Chapter 3](03_how_neural_networks_learn.md)) only improves
parameters, hyperparameters have to be chosen a different way — by trying values,
watching what happens, and adjusting. This can range from "an experienced practitioner's
educated guess based on similar past projects" to "an automated search trying hundreds of
combinations." Both are legitimate; which one makes sense depends on how expensive each
training run is and how much time is available to search.

## The First-Principles Explanation

### Why hyperparameter tuning is a genuinely different problem than training

Training (Part 0's [Chapter 3](03_how_neural_networks_learn.md)) has a gradient telling
you exactly which direction to move every parameter. Hyperparameter tuning has no such
signal — there's no gradient telling you "increase `num_layers` by 2." The only way to
know if a hyperparameter choice was good is to actually run (or partially run) training
with it and observe the result — making hyperparameter tuning fundamentally more
expensive per "step" than parameter learning, since each data point requires a real
training run, not one backward pass.

### The main tuning strategies, from cheapest to most systematic

- **Manual / intuition-based** — an experienced practitioner picks values based on
  similar past projects, published papers' reported settings, or rules of thumb. Cheapest,
  fastest to start, and often good enough — this is what this repo's `tiny_llm.py` does
  (its config comment literally says "~150M-parameter configuration with conservative
  micro-batching for MPS laptops," a human judgment call, not a search result).
- **Grid search** — pick a small set of candidate values for each hyperparameter, try
  every combination. Simple and exhaustive within the grid, but the number of combinations
  grows multiplicatively with each additional hyperparameter (a real, practical limit).
- **Random search** — sample random combinations from a defined range, rather than every
  grid point. Often more efficient than grid search in practice, since it doesn't waste
  budget exploring many values of a hyperparameter that turns out not to matter much,
  while under-exploring one that does.
- **Bayesian optimization** (tools like Optuna) — uses the results of previous trials to
  intelligently choose which combination to try next, rather than randomly or
  exhaustively — the most sample-efficient approach, worth reaching for when each
  training run is expensive enough that manual/grid/random search would cost too much
  compute.

### Which hyperparameters actually matter most for a Transformer

Not all hyperparameters are equally sensitive — worth knowing where to focus:

| Hyperparameter | Sensitivity | Why |
|---|---|---|
| Learning rate | **Very high** | Too high: training diverges/becomes unstable. Too low: painfully slow convergence. Usually the single most impactful knob. |
| Batch size (effective, via `batch_size × grad_accum_steps`) | Medium-high | Affects gradient noise and stability; interacts directly with learning rate (larger effective batch often tolerates a higher learning rate) |
| Model width/depth (`embed_size`, `num_layers`) | High, but mostly a *capacity* ceiling | Determines what the model *could* learn, not how efficiently it learns it |
| Dropout | Medium | Too high: underfits (the model can't use its own capacity effectively). Too low, on a small dataset: overfits |
| Warmup steps | Low-medium | Mainly affects early-training stability, less consequential once training is underway |

## Grounded in This Repo's Code

`tiny_llm.py` bakes real hyperparameter-tuning reasoning directly into comments — worth
reading as an example of applied judgment, not just abstract theory:

```python
# TODO: If training on AWS g5.2xlarge (A10G GPU), set batch_size=16 and grad_accum_steps=2
# to utilize the 24GB VRAM and speed up training significantly.
batch_size = 1  # Set to 1 for MPS/Laptop to minimize VRAM usage
grad_accum_steps = 32  # Accumulate gradients to simulate batch_size=32
```

This comment is a real, concrete instance of hyperparameter tuning reasoning: `batch_size`
and `grad_accum_steps` together determine the **effective batch size**
(`batch_size × grad_accum_steps = 32`, held constant in both scenarios), but the *split*
between them is chosen based on available GPU/unified memory — a Mac laptop's limited
memory forces `batch_size=1` with heavy accumulation; a `g5.2xlarge`'s 24GB A10G (per
the sibling `platform-lab` repo's `gpu_infrastructure/phase6_production_operations/
25_single_gpu_instance_selection_g5_g6.md` chapter on single-GPU AWS instances)
can hold a real batch of 16 sequences at once, needing far less accumulation to reach the
same effective batch size, and running faster as a direct result (real batching is more
efficient than simulating it via many small accumulated steps).

This is the exact hardware-aware hyperparameter reasoning [Chapter 2](02_parameters_vs_hyperparameters.md)'s
distinction sets up: **the correct value of a hyperparameter is not a universal constant —
it depends on the model, the data, and the hardware it's running on**, and choosing well
requires understanding all three, not just the model architecture in isolation.

### Using `train_loss` vs. `test_loss` as your tuning feedback signal

`tiny_llm.py`'s `estimate_loss` function (line 320) computes both `train_loss` and
`test_loss` every `eval_interval` steps, logged to `logs/train_eval_history.csv`. This
pair of numbers is the actual feedback signal a hyperparameter-tuning process watches:

```
train_loss decreasing, test_loss decreasing together  -> healthy, keep going
train_loss decreasing, test_loss flat or increasing    -> overfitting; consider
                                                            increasing dropout, reducing
                                                            model size, or getting more
                                                            training data
train_loss and test_loss both flat, high                -> underfitting; consider a
                                                            higher learning rate, more
                                                            training steps, or more
                                                            model capacity
```

This is a real, actionable diagnostic loop — not a one-time setup decision, but something
you'd watch throughout a training run and adjust hyperparameters in response to, exactly
as [`from_scratch/custom-gpt-153m/docs/LLM_DEV_GUIDE.md`'s quality-tracking section](../../from_scratch/custom-gpt-153m/docs/LLM_DEV_GUIDE.md#9-evaluation-during-training)
already describes operationally.

## Deep-Dive: The Learning-Rate/Batch-Size Interaction, Precisely

A specific, commonly-encountered interaction worth understanding: a larger effective
batch size produces a less noisy (more averaged) gradient estimate at each step, which
often means training can tolerate — and benefit from — a *higher* learning rate without
becoming unstable. This is why hyperparameters shouldn't be tuned in complete isolation
from each other; changing `grad_accum_steps` (and therefore effective batch size) without
reconsidering `lr` risks leaving a real gain on the table, or destabilizing a
previously-fine learning rate. A full grid or random search naturally captures this
interaction (by trying combinations, not each hyperparameter independently); manual
tuning has to hold this relationship in mind deliberately.

## Try It Yourself

- Pick one hyperparameter in `tiny_llm.py` (e.g. `dropout`), change it, and run a short
  training session (a few hundred steps is enough to see a trend). Compare the
  `train_loss`/`test_loss` gap in `logs/train_eval_history.csv` against a baseline run —
  this is hyperparameter tuning's actual feedback loop, done by hand.
- If you have access to both a memory-constrained machine (like a laptop) and a
  GPU-equipped one, try the two `batch_size`/`grad_accum_steps` configurations the
  code comment suggests, and compare wall-clock training speed for the same effective
  batch size — a direct, observed instance of hardware-aware hyperparameter choice.

## Common Misconceptions

- **"There's one universally correct hyperparameter setting for a given model size."**
  As the batch-size example shows, the *right* value depends on the hardware and dataset
  too, not just the architecture — published "recommended settings" are starting points,
  not guarantees.
  applicable everywhere.
- **"A smaller learning rate is always the 'safer' choice."** Too small a learning rate
  isn't safe — it's just a different failure mode (extremely slow convergence, or getting
  effectively stuck long before reaching a good solution within the training budget).
- **"Hyperparameter tuning happens once, before training starts, and then you're done."**
  As the `train_loss`/`test_loss` diagnostic loop shows, it's often an iterative process
  — watching a run's early behavior and adjusting, not a one-shot decision made in
  isolation from any actual training feedback.

## Practice Questions

1. Why does increasing effective batch size often allow a higher learning rate without
   destabilizing training — what's the mechanism connecting the two?
2. A training run shows `train_loss` steadily decreasing but `test_loss` flat after a
   certain point. Name two specific hyperparameter changes that could address this, and
   explain the mechanism behind each.
3. Why is grid search's cost described as growing "multiplicatively" with each additional
   hyperparameter — work through a concrete example with 3 hyperparameters, each with 4
   candidate values.

## Key Terms

- **Grid search / random search / Bayesian optimization**: three strategies for
  systematically searching a hyperparameter space, in increasing order of sample
  efficiency and implementation complexity.
- **Effective batch size**: `batch_size × grad_accum_steps` — the true number of examples
  averaged into one gradient-descent update.
- **Overfitting**: a model fitting training data very well while generalizing poorly to
  new data — diagnosed by a growing gap between `train_loss` and `test_loss`.
- **Underfitting**: a model failing to fit even the training data well — diagnosed by
  both `train_loss` and `test_loss` remaining high.
