# Causal Q/K/V attention, from intuition to code

This guide explains the most important mechanism in a GPT: **causal self-attention**.
Read the sections in order. The first assumes no machine-learning background; the last
connects each idea to the implementation in [`../src/wordgpt/model.py`](../src/wordgpt/model.py).

## Level 1: the plain-language idea

Imagine reading this sentence one word at a time:

```text
the cat sat on the mat .
```

When you reach `sat`, you naturally use earlier words to understand it. You connect
`sat` to `cat`: *who sat?* You might mostly ignore `the`, because it is less useful for
that question.

Attention gives the model that same ability: for every current word, it can look back
at earlier words and decide which ones matter most.

It is called **self-attention** because all of those words come from the same sentence.
It is called **causal** because a word may look only **backward**, never forward.

Why never forward? GPT is trained to predict the next token. If it could look at the
future, the answer would already be visible:

```text
input:   the  cat  sat
answer:  cat  sat  on
```

To predict the token after `cat`, the model cannot be allowed to see `sat` or `on`.
That would be like answering a fill-in-the-blank question after reading the answer key.

## Level 2: the three labels—Query, Key, and Value

For each token, the model creates three small lists of numbers. We call them vectors.
The names are borrowed from looking something up in a library:

| Name | Everyday question | Role in attention |
|---|---|---|
| **Query (Q)** | “What am I looking for?” | Describes what the current token needs from context. |
| **Key (K)** | “What topics can I help with?” | Describes what information a token offers. |
| **Value (V)** | “What should I hand over?” | Carries the information that is actually shared. |

The model learns the numerical contents of all three during training. We do **not**
program a rule such as “verbs should look for nouns.” Initially the vectors are random;
after many examples, gradient descent changes them so useful relationships get higher
attention.

### A library analogy

Suppose `sat` wants to understand its subject.

- `sat` makes a **Query** resembling “I need the thing that performed this action.”
- Earlier token `cat` has a **Key** resembling “I could be an acting thing.”
- The Query and Key match well, so `cat` gets a high attention score.
- `cat`'s **Value**—its useful contextual information—is mixed into `sat`'s new
  representation.

The real model does not use English labels such as “subject.” It only uses learned
numbers. The labels above are a helpful mental model for the job those numbers perform.

## Level 3: one worked attention example

Consider the third token in this short sequence:

```text
position:  0    1    2
token:    the  cat  sat
```

When processing `sat`, the causal rule lets it consider `the`, `cat`, and itself. It
calculates a match score between `sat`'s Query and each available Key. Pretend it has
learned these scores:

| `sat` compares its Query with the Key of… | Raw match score |
|---|---:|
| `the` | 0.2 |
| `cat` | 3.0 |
| `sat` | 0.8 |

Raw scores are not probabilities yet. **Softmax** converts them into positive fractions
that add up to one:

| Earlier token | Attention weight | Interpretation |
|---|---:|---|
| `the` | 0.05 | Barely use this information. |
| `cat` | 0.80 | This is strongly relevant. |
| `sat` | 0.15 | Keep some information about the current word. |

The model multiplies each token's Value vector by its weight and adds them together:

```text
new representation for "sat"
  = 0.05 × Value(the)
  + 0.80 × Value(cat)
  + 0.15 × Value(sat)
```

This new vector is not a word replacement. It is a richer internal description of
`sat`, containing information gathered from the relevant earlier context. Later layers
can use that description when predicting the next token, perhaps `on`.

Different attention heads can learn different useful patterns. One may often connect a
verb to its subject; another may track repetition, nearby punctuation, or the beginning
of a phrase. Those are patterns we can inspect after training, not rules guaranteed by
the architecture.

## Level 4: the causal mask

For the sentence below, the row says which positions a token is allowed to read.

```text
             visible tokens
current       the   cat   sat   on
the            yes    no    no   no
cat            yes   yes    no   no
sat            yes   yes   yes   no
on             yes   yes   yes  yes
```

This is a lower-triangular pattern: allowed cells run diagonally down and left. In the
source code it is made once when the attention layer is created:

```python
self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
```

Before softmax, blocked future positions are replaced by negative infinity:

```python
scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
```

Softmax turns a negative-infinity score into exactly zero probability. This is not a
polite request to the model; it is a structural restriction. No weight update can make
the model attend to a masked future position.

## Level 5: how the code turns this into tensors

In this project, the input `x` has shape `(B, T, C)`:

| Symbol | Meaning | Example during `make dry-run` |
|---|---|---:|
| `B` | number of independent examples in a batch | 16 |
| `T` | tokens per example/context window | 12 |
| `C` | embedding width: numbers representing each token | 96 |

First, one linear layer produces three versions of each token representation:

```python
q, k, v = self.qkv(x).split(C, dim=-1)
```

Each has shape `(B, T, C)`. The next lines split `C` across four attention heads, giving
each head 24 numbers per token. Then this line compares every Query against every Key:

```python
scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
```

The result has shape `(B, H, T, T)`. The two `T` dimensions form the table of “current
token versus token it might read.” The division by `sqrt(head_size)` keeps values in a
range where softmax can still learn effectively.

Finally, the essential four lines are:

```python
scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)
attended = weights @ v
return self.proj(attended)
```

Read them as:

```text
hide the future -> convert relevance to percentages -> collect useful Values -> mix heads
```

## Try it yourself

Run the safe inspection path first:

```bash
cd from_scratch/custom-gpt-word
make dry-run
```

It prints a real input window and its shifted target. Pick one position in the input and
ask: “If the model is predicting the target at this position, which earlier words could
help?” Then compare your answer with the causal-mask rule: all earlier words and the
current word are available; future words are not.

After `make train`, try generation with a familiar corpus prefix:

```bash
make generate PROMPT="the cat"
```

The generated token is selected from the final position's output distribution. Its
attention layers have only used the prompt so far (and previously generated tokens),
which is why this process can keep producing one token at a time.
