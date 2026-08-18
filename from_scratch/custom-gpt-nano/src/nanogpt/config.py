"""
The model's "shape" and the training run's knobs, in one place.

WHAT is a "hyperparameter"? Any number you (the human) choose *before* training starts
and that training itself never changes — as opposed to a "parameter" (a weight inside
the model, like one number in an attention matrix), which training *does* change, via
gradient descent, on every step. `n_embd`, `n_layer`, `learning_rate` below are all
hyperparameters. Every number inside the model's weight tensors is a parameter.
Deep dive: docs/llm-engineering/02_parameters_vs_hyperparameters.md (repo root).

This file has two dataclasses instead of one, because the two groups of numbers answer
genuinely different questions and get used at different times:
  - GPTConfig  -> "what does the network look like?" (needed every time you rebuild the
                  model, including at generation time, long after training is done)
  - TrainConfig -> "how do we run the training loop?" (only needed while training)
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    """How many distinct tokens (here: distinct characters) the model can ever see or
    produce. Set from the tokenizer's alphabet, not hand-picked (see tokenizer.py) —
    it has to match exactly, or the embedding table below is the wrong shape."""

    block_size: int = 64
    """AKA "context length": the maximum number of tokens the model looks at when
    predicting the next one. A window, not a memory — anything before the start of the
    window is invisible to the model, full stop. 64 is small on purpose: on this
    project's ~5KB toy corpus, 64 characters is already 1-2 full sentences, plenty of
    context for the patterns this corpus actually has."""

    n_embd: int = 128
    """AKA "d_model" / "hidden size": the length of the vector each token is turned
    into, and the width every internal computation keeps thereafter. Bigger = the model
    can represent more nuance per token, at the cost of ~quadratically more compute in
    the attention step below. 128 is deliberately tiny — see the README's parameter
    table for exactly where these ~0.8M parameters go."""

    n_head: int = 4
    """How many parallel "attention heads" split `n_embd` between them. Must divide
    `n_embd` evenly — each head gets `n_embd // n_head` dimensions to work with (128/4 =
    32 here). Why more than one head at all: one head learns one *kind* of relationship
    between tokens (e.g. "the previous word"); multiple heads let the model track
    several different relationships in parallel, at the same layer, for free."""

    n_layer: int = 4
    """How many Transformer blocks are stacked. Each block gets one more chance to
    refine every token's representation using what it learned from the tokens before
    it. More layers = deeper reasoning chains, more parameters, more compute."""


@dataclass
class TrainConfig:
    batch_size: int = 32
    """How many independent (x, y) training windows go through the model together in
    one forward/backward pass. Purely a throughput/stability knob — averaging the
    gradient over more examples per step gives a less noisy update direction, at the
    cost of more memory per step. Does not change what the model *can* learn."""

    learning_rate: float = 3e-3
    """The size of each gradient-descent weight update — see train.py's docstring for
    the mechanism. Higher than you'd use on a 50M+ model on purpose: this model and
    this dataset are both tiny, so bigger, faster steps converge in seconds instead of
    minutes without destabilizing training. Deep dive:
    docs/llm-engineering/03_how_neural_networks_learn.md."""

    max_steps: int = 2000
    """How many optimizer updates to run in total. Not epochs — see data.py's
    docstring for why this codebase samples random windows instead of walking the
    corpus in order."""

    eval_interval: int = 200
    """How often (in steps) to pause training and measure loss on the held-out
    validation split — see train.py for why this has to be a *separate* pass from the
    training loss you see every step."""

    eval_iters: int = 50
    """How many validation batches to average over each time we evaluate. One batch
    alone is a noisy, small sample; averaging several gives a steadier number."""

    device: str = "auto"
    """"auto" picks the fastest available backend at runtime: Apple Silicon "mps",
    then CUDA, falling back to plain "cpu" — see train.py's `pick_device()`."""
