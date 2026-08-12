# Neurons, Layers, and Neural Networks

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 0 — Deep Learning
Foundations. This is the very first chapter for a reason: an LLM is not a separate kind
of thing from a neural network — it *is* a neural network, a very large and specifically
shaped one. Everything in this curriculum builds on the vocabulary introduced here.

## In Plain English

A neuron, in the deep-learning sense, is a tiny decision-making unit: it takes several
numbers in, multiplies each by an importance weight, adds them up (plus one extra
"bias" number), and passes the result through a simple function that decides how strongly
to "fire." A neural network is just many of these tiny units, arranged in layers, each
layer's output feeding the next layer's input. Nothing here is mysterious — the
"intelligence" of a trained network isn't in any single neuron, it's in the *pattern* of
millions of weights, tuned together, that the whole network settles into during training
([Chapter 3](03_how_neural_networks_learn.md)).

## The First-Principles Explanation

### One neuron, in full

```
output = activation_function( (input_1 × weight_1) + (input_2 × weight_2) + ... + bias )
```

- **Inputs** — numbers coming from either the raw data or a previous layer's outputs.
- **Weights** — one learned number per input, controlling how much that input matters.
- **Bias** — one more learned number, added regardless of the inputs — lets the neuron
  shift its output up or down independent of what it's fed (without it, a neuron with all
  zero inputs is stuck always outputting zero).
- **Activation function** — a nonlinear function applied to the weighted sum. This is the
  single most important detail in this whole chapter, covered in the deep-dive below.

### Layers: neurons arranged side by side, then stacked

A **layer** is a group of neurons that all take the same inputs but each has its own
independent weights and bias — so a layer with 768 neurons, given a 768-number input,
produces 768 different numbers out, each a different weighted combination. Stack several
layers — each one's output feeding the next one's input — and you get a **multi-layer
neural network**. "Deep learning" literally means "learning with networks that have many
stacked layers" — depth is the defining property the name refers to.

### Weights and biases, together, are called parameters

Every weight and every bias in the network is a **parameter** — a number the network
*learns* during training, as opposed to a setting a human chooses ahead of time (that
second category is covered fully in [Chapter 2](02_parameters_vs_hyperparameters.md)).
When you hear a model described as having "153 million parameters," this is exactly what
that's counting: every weight and bias, across every layer, added up.

## Grounded in This Repo's Code

`nn.Linear` in PyTorch is a direct implementation of "a full layer of neurons" — each
`nn.Linear(in_features, out_features)` internally holds a weight matrix and a bias
vector, and computes exactly the weighted-sum formula above for every one of its
`out_features` neurons simultaneously. This repo's
[`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py) uses it directly, most
readably in the `MLP` class:

```python
class MLP(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),   # a layer of 3072 neurons
            nn.GELU(),                                 # the activation function
            nn.Linear(4 * embed_size, embed_size),    # a layer of 768 neurons
            nn.Dropout(dropout),
        )
```

This is a complete, working 2-layer neural network — `embed_size` (768) numbers in, an
intermediate layer of `4 * embed_size` (3072) neurons, a nonlinearity (`GELU`), then back
down to 768. The entire `TinyGPT` model (covered fully in
[Chapter 10](10_transformer_architecture.md)) is, underneath the Transformer-specific
mechanisms, sixteen of these MLP blocks plus attention layers, stacked — the same
building block from this chapter, repeated and combined at scale.

## Deep-Dive: Why the Activation Function Is Not Optional

This is worth understanding precisely, because it's the single fact that explains why
"deep" networks work at all. **Without a nonlinear activation function, stacking any
number of linear layers is mathematically equivalent to just one linear layer.** A linear
transformation followed by another linear transformation is still just a linear
transformation — the composition of two matrix multiplications is another matrix
multiplication. If `MLP` in the code above didn't have `nn.GELU()` between its two
`nn.Linear` layers, stacking 16 `GPTBlock`s worth of these would collapse, mathematically,
into something no more expressive than a single layer — all that depth would buy nothing.

The activation function's nonlinearity is what actually lets a deep network represent
complex, non-straight-line relationships between input and output — it's not a minor
implementation detail, it's the property that makes "deep learning" meaningfully
different from linear regression stacked on itself.

Different activation functions are real, specific choices, not interchangeable defaults:
- **GELU** (used in this repo's MLP) — a smooth, differentiable approximation that tends
  to outperform older choices in Transformer-style models, part of why it's the common
  default in this architecture family.
- **ReLU** — simpler, just `max(0, x)`, historically very common, computationally cheap,
  but has a known "dying neuron" failure mode (a neuron stuck outputting 0 for every
  input can stop learning entirely, since its gradient there is also 0).
- **Sigmoid/Tanh** — older choices, largely superseded in hidden layers of deep networks
  because their gradients shrink toward the extremes of their range, contributing to a
  historically real training problem ("vanishing gradients") in deep stacks.

## Try It Yourself

- In a Python shell, with PyTorch installed (already in
  [`requirements.txt`](../../from_scratch/custom-gpt-153m/requirements.txt)):
  ```python
  import torch.nn as nn
  layer = nn.Linear(4, 3)
  print(layer.weight.shape)   # torch.Size([3, 4]) — one row per output neuron
  print(layer.bias.shape)     # torch.Size([3])    — one bias per output neuron
  ```
  Confirm for yourself: a `Linear(4, 3)` layer has exactly `3 × 4 + 3 = 15` parameters —
  the same weight-and-bias counting this chapter describes, verified directly.
- Sum up every `nn.Linear` and `nn.Embedding` layer's parameter count in `tiny_llm.py`'s
  `TinyGPT` class by hand, or with `sum(p.numel() for p in model.parameters())`, and
  compare against the ~152.8M figure documented in
  [`from_scratch/custom-gpt-153m/README.md`](../../from_scratch/custom-gpt-153m/README.md#parameter-count-current-config).

## Common Misconceptions

- **"A neural-network neuron works like a brain neuron."** The name is a loose historical
  analogy, not a claim of biological accuracy — an artificial neuron is a simple weighted
  sum plus a nonlinearity, vastly simpler than an actual biological neuron's behavior.
- **"More neurons/layers automatically makes a model smarter."** Not automatically — a
  larger network has more capacity, but whether that capacity is used well depends on
  training data, training time, and hyperparameters ([Chapter 4](04_hyperparameter_tuning.md));
  a large, undertrained network can perform worse than a smaller, well-trained one.
- **"The activation function is a minor tuning detail."** As the deep-dive shows, it's
  the specific property that makes depth mathematically meaningful at all — removing it
  isn't a small quality regression, it collapses the model's representational power
  entirely.

## Practice Questions

1. Why does stacking ten linear layers with no activation function in between produce a
   model no more expressive than a single linear layer?
2. In `MLP`'s two `nn.Linear` layers, why does the first one expand from 768 to 3072
   dimensions before the second one projects back down to 768, rather than staying at
   768 throughout?
3. What's the exact parameter count of an `nn.Linear(768, 768)` layer, and what are the
   two components (weight, bias) that make it up?

## Key Terms

- **Neuron**: a unit computing a weighted sum of its inputs plus a bias, passed through
  an activation function.
- **Weight**: a learned number controlling how much one input contributes to a neuron's
  output.
- **Bias**: a learned, input-independent offset added to a neuron's weighted sum.
- **Activation function**: a nonlinear function (GELU, ReLU, etc.) applied after the
  weighted sum — the property that makes deep stacks of layers meaningfully more
  expressive than one layer.
- **Layer**: a group of neurons sharing the same inputs, each with independent
  weights/bias.
- **Parameter**: any single learned weight or bias in the network — the unit "152.8M
  parameters" is counted in.
- **Deep learning / deep neural network**: a network with multiple stacked layers.
