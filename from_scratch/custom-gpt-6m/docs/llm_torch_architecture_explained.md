# Understanding the PyTorch LLM / GPT Architecture

This note summarizes the discussion about the two small GPT-style Transformer implementations and clarifies **what each component is, why it exists, how the pieces are organized, and how data moves from input to output**.

The goal is to understand the architecture conceptually before getting lost in PyTorch implementation details.

---

## 1. The big picture

An LLM can be viewed as a machine that takes token IDs and predicts the next token.

```text
Token IDs
   ↓
Embedding
   ↓
Transformer Block 1
   ↓
Transformer Block 2
   ↓
...
   ↓
Transformer Block N
   ↓
Final Normalization
   ↓
LM Head
   ↓
Logits
   ↓
Next-token probabilities / prediction
```

The important idea is:

> **The whole LLM is a stack of Transformer blocks followed by an output projection.**

---

# 2. PyTorch vs Transformer vs GPT vs Block

These terms refer to different levels.

```text
PyTorch
  ↓
framework used to implement neural networks

Transformer
  ↓
neural-network architecture

GPT
  ↓
decoder-only / causal Transformer style

TinyGPT / TinyStoriesGPT
  ↓
a particular implementation of that architecture
```

A Python class such as:

```python
class GPTBlock(nn.Module):
```

is simply how the programmer groups a set of operations into a reusable component.

A **class is not necessarily one mathematical operation**.

---

# 3. What does `num_layers = N` mean?

In these models, `num_layers` means the number of Transformer blocks that are stacked sequentially.

For example:

```text
num_layers = 4
```

means:

```text
Input
  ↓
GPT Block 1
  ↓
GPT Block 2
  ↓
GPT Block 3
  ↓
GPT Block 4
  ↓
Output
```

The blocks have the **same architecture**, but they normally have **different learned weights**.

It does NOT mean:

```text
Run one block 4 times using exactly the same weights
```

Instead it means:

```text
Create four separate GPTBlock objects

Block 1 → its own weights
Block 2 → its own weights
Block 3 → its own weights
Block 4 → its own weights
```

The original code creates them using:

```python
self.blocks = nn.ModuleList(
    [GPTBlock(...) for _ in range(num_layers)]
)
```

So the blocks are sequential in the forward pass.

---

# 4. Whole model vs one block

There are two levels that are easy to confuse.

## Level 1 — Whole model

```text
Input
 ↓
Embedding
 ↓
Block × N
 ↓
Final Norm
 ↓
LM Head
 ↓
Output
```

This describes the entire LLM.

## Level 2 — One Transformer block

```text
Input
 ↓
Norm
 ↓
Attention
 ↓
Residual
 ↓
Norm
 ↓
MLP
 ↓
Residual
 ↓
Output
```

The second diagram is **inside each item represented by `Block × N` in the first diagram**.

So:

```text
Whole Model
│
├── Embedding
│
├── Block 1
│    ├── Norm
│    ├── Attention
│    ├── Residual
│    ├── Norm
│    ├── MLP
│    └── Residual
│
├── Block 2
│    └── same structure, different weights
│
├── ...
│
├── Block N
│
├── Final Norm
│
└── LM Head
```

This nesting is one of the most important things to understand.

---

# 5. Why do we need Embedding?

The input to the model is token IDs.

For example:

```text
"The dog runs"
```

might become something like:

```text
[15, 892, 431]
```

These are just IDs.

The embedding converts each ID into a vector:

```text
Token ID 892
    ↓
[0.2, -0.7, 0.4, 0.1, ...]
```

So:

> **Embedding = convert token IDs into numerical representations that the neural network can process.**

With an embedding size of 256:

```text
token IDs
   ↓
(batch, sequence)
   ↓
Embedding
   ↓
(batch, sequence, 256)
```

---

# 6. Position information

A Transformer needs to know where tokens occur.

For example:

```text
dog loves cat
```

is different from:

```text
cat loves dog
```

There are two different approaches in the two models discussed.

### Model 1

Uses **RoPE (Rotary Positional Embedding)**.

Position is introduced by rotating Q and K inside attention.

```text
Q ──→ RoPE
K ──→ RoPE
V ──→ unchanged
```

### Model 2 — TinyStoriesGPT

Uses a learned position embedding:

```python
self.pos_emb = nn.Embedding(context_length, embed_size)
```

and combines it with token embeddings:

```python
h = self.token_emb(x) + self.pos_emb(pos)
```

So its input stage is:

```text
Token IDs
   │
   ├──→ Token Embedding ──┐
   │                      │
   └──→ Position Embedding┤
                          ↓
                         ADD
                          ↓
                       Dropout
                          ↓
                     Transformer
                     Blocks
```

---

# 7. What is Attention?

Attention allows token representations to exchange information.

Consider:

```text
"The animal didn't cross the road because it was tired."
```

When processing `it`, the model can use information from earlier tokens to determine what `it` may refer to.

Conceptually:

```text
Token
  ↓
Attention
  ↓
Look at relevant other tokens
  ↓
Gather useful information
```

A useful mental model is:

> **Attention = communication between tokens.**

It answers something like:

> "Which other tokens are relevant to me, and how much should I use their information?"

---

# 8. Q, K, V

Inside attention, the input is projected into three representations:

```text
x
│
├──→ Q (Query)
├──→ K (Key)
└──→ V (Value)
```

A useful intuition:

```text
Query = what information am I looking for?
Key   = what information do I advertise?
Value = what information do I provide?
```

Attention uses Q and K to determine relevance, then uses those relevance weights to combine V.

Conceptually:

```text
Q + K
  ↓
attention scores
  ↓
softmax
  ↓
attention weights
  ↓
weights × V
  ↓
attention output
```

The exact mathematical details can be learned separately.

---

# 9. What does "causal" mean?

GPT-style language modeling uses causal attention.

For a sequence:

```text
The cat sat on the mat
```

the token at position 3 can attend to positions 1, 2, and 3, but not future positions.

Conceptually:

```text
          Keys
          1  2  3  4

Query 1   ✓  ✗  ✗  ✗
Query 2   ✓  ✓  ✗  ✗
Query 3   ✓  ✓  ✓  ✗
Query 4   ✓  ✓  ✓  ✓
```

This prevents the model from cheating during next-token prediction.

---

# 10. What is Norm?

This is a **numerical stability / representation scaling operation**.

Neural networks pass around vectors containing many numbers.

After many calculations, their scale can become inconvenient for training.

Normalization helps keep the representations well-behaved.

A useful beginner mental model:

```text
Representation
      ↓
    Norm
      ↓
More controlled representation
```

It is not primarily a "thinking" operation.

It is more like maintaining the numerical conditions under which the network operates effectively.

The two models use different normalization choices:

```text
Model 1 → RMSNorm
Model 2 → LayerNorm
```

---

# 11. Why does Norm appear before Attention?

The discussed models use **Pre-Norm** blocks.

The flow is:

```text
x
 ↓
Norm
 ↓
Attention
```

The idea is:

> Normalize the current representation before giving it to the next major transformation.

In the second model:

```python
x = x + self.attn(self.ln_1(x))
```

This can be mentally expanded as:

```text
x
 ↓
LayerNorm
 ↓
Attention
 ↓
attention output
```

---

# 12. What is a Residual connection?

Suppose:

```text
x
 ↓
Attention
 ↓
new information
```

Instead of throwing away the original `x`, the model adds the new information to it:

```text
original x ──────────────┐
                         │
x → Attention → new info ─┤
                         ↓
                         +
                         ↓
                    new representation
```

In code:

```python
x = x + attention_output
```

So:

> **Residual = keep what we already have and add the new information.**

The same idea is used after the MLP.

Residual connections help information and gradients flow through deep networks.

---

# 13. What is MLP?

This was one of the main points clarified in the discussion.

**MLP is a component, not necessarily one single operation.**

In the second model it is implemented as:

```python
class MLP(nn.Module):
    ...
```

Inside it:

```python
self.net = nn.Sequential(
    nn.Linear(embed_size, 4 * embed_size),
    nn.GELU(),
    nn.Linear(4 * embed_size, embed_size),
    nn.Dropout(dropout),
)
```

Therefore:

```text
MLP
│
├── Linear
├── GELU
├── Linear
└── Dropout
```

So:

> **MLP = a small neural network made from several operations.**

The same idea applies to Attention: the `Attention` class is a component containing multiple underlying operations.

---

# 14. What does MLP actually do?

The most useful distinction is:

```text
Attention
    ↓
communication between tokens

MLP
    ↓
processing/transformation of each token's representation
```

Attention can gather information from other tokens.

MLP then transforms the resulting representation.

A simplified mental model:

```text
Attention:
"What information should I gather from the context?"

MLP:
"Now that I have this information, how should I transform/process it?"
```

This is simplified intuition, but very useful.

---

# 15. Why does the MLP expand and then shrink?

The second model uses:

```text
embed_size
    ↓
4 × embed_size
    ↓
GELU
    ↓
embed_size
```

For example, if:

```text
embed_size = 256
```

then:

```text
256
 ↓
1024
 ↓
GELU
 ↓
256
```

The larger intermediate representation provides a wider space in which to perform nonlinear computation.

Think of it as:

```text
small representation
       ↓
larger computational workspace
       ↓
nonlinear processing
       ↓
back to the original representation size
```

---

# 16. What is GELU?

GELU is an activation function.

It provides a nonlinear transformation.

Without nonlinear operations, stacking linear transformations would be much less expressive.

So:

```text
Linear
 ↓
GELU
 ↓
Linear
```

forms a small nonlinear neural network.

You do not need to memorize the GELU formula to understand the architecture.

---

# 17. Attention vs MLP — the key distinction

This is probably the most important conceptual distinction in the block.

```text
                 Transformer Block
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
         Attention                MLP
             │                     │
             ▼                     ▼
   "Get information             "Process/
    from other tokens"           transform it"
```

A useful analogy:

```text
Attention = communication
MLP       = computation
```

Again, this is an intuition rather than a literal description of every learned behavior.

---

# 18. One complete Transformer block

Now we can understand this:

```text
x
 ↓
Norm
 ↓
Attention
 ↓
Residual
 ↓
Norm
 ↓
MLP
 ↓
Residual
 ↓
Output
```

in plain language:

```text
                         START
                           │
                           ▼
                  Normalize representation
                           │
                           ▼
                  Let tokens communicate
                       (Attention)
                           │
                           ▼
              Keep old information + new info
                      (Residual)
                           │
                           ▼
                  Normalize again
                           │
                           ▼
                  Process/transform
                    representation
                        (MLP)
                           │
                           ▼
              Keep old information + new info
                      (Residual)
                           │
                           ▼
                         OUTPUT
```

That output becomes the input to the next Transformer block.

---

# 19. Multiple blocks

If:

```text
num_layers = 6
```

then:

```text
Embedding
   ↓
Block 1
   ↓
Block 2
   ↓
Block 3
   ↓
Block 4
   ↓
Block 5
   ↓
Block 6
   ↓
Final Norm
   ↓
LM Head
   ↓
Logits
```

Each block has the same general architecture but its own learned parameters.

---

# 20. Why have multiple blocks?

Each block gets the representation produced by the previous block and transforms/refines it further.

A useful intuition is:

```text
Embedding
   ↓
initial token representations

Block 1
   ↓
contextualized representations

Block 2
   ↓
further transformed representations

Block 3
   ↓
further refinement

...
   ↓
Block N
   ↓
rich contextual representation
```

Do not interpret this as a strict rule that Block 1 only learns one specific type of information and Block 2 another. All blocks are learned jointly.

---

# 21. Final Norm

After the last Transformer block, the model applies one final normalization:

```text
Block N
   ↓
Final Norm
```

This is separate from the normalization operations inside every block.

For the second model:

```python
self.ln_f = nn.LayerNorm(embed_size)
```

---

# 22. What is the LM Head?

The final hidden representation is not yet a prediction.

The LM Head converts the hidden representation into one score for every token in the vocabulary.

For example:

```text
Final hidden representation
          ↓
       LM Head
          ↓
┌──────────────────────┐
│ dog       0.2        │
│ cat       0.1        │
│ running   4.7        │
│ eating    1.2        │
│ ...                  │
└──────────────────────┘
```

These scores are called **logits**.

So:

> **LM Head = convert the final hidden representation into next-token scores.**

---

# 23. Weight tying

The models use:

```python
self.lm_head.weight = self.token_emb.weight
```

This means the embedding matrix and output projection share the same weights.

Conceptually:

```text
Token ID
   ↓
Embedding matrix
   ↓
hidden representation

and

hidden representation
   ↓
same matrix
   ↓
token logits
```

This saves parameters and is a common language-model technique.

---

# 24. `encode()` vs `forward()`

The second model separates the Transformer computation from the vocabulary projection.

`encode()` does approximately:

```text
Input
 ↓
Token + Position Embedding
 ↓
Dropout
 ↓
Block 1
 ↓
...
 ↓
Block N
 ↓
Final LayerNorm
 ↓
Hidden states
```

Then `forward()` does:

```python
return self.lm_head(self.encode(x))
```

So:

```text
encode()
   ↓
final hidden representation
   ↓
LM Head
   ↓
logits
```

This allows the hidden representation to be reused for another task.

---

# 25. Forward pass

The **forward pass** is the normal direction of computation.

It starts with the input and ends with the loss during training.

```text
INPUT
  ↓
Token IDs
  ↓
Embedding
  ↓
Position information
  ↓
Block 1
  ↓
Block 2
  ↓
...
  ↓
Block N
  ↓
Final Norm
  ↓
LM Head
  ↓
Logits
  ↓
Loss
```

Everything from input toward the loss is the forward computation.

---

# 26. Backward pass

After calculating the loss, PyTorch performs backpropagation.

The important point:

> **Backward pass does not mean running the model backwards to generate text.**

It means calculating gradients showing how the loss changes with respect to the model's parameters.

Conceptually:

```text
Forward:

Input
 ↓
Block 1
 ↓
Block 2
 ↓
...
 ↓
LM Head
 ↓
Loss


Backward:

Loss
 ↑
LM Head
 ↑
Block N
 ↑
...
 ↑
Block 2
 ↑
Block 1
 ↑
Embedding
```

The backward pass calculates gradients through the computation graph.

Then the optimizer uses those gradients to update the weights.

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Gradients
   ↓
Optimizer
   ↓
Updated weights
   ↓
Next training step
```

---

# 27. What happens to Attention during backward?

Attention participates in both directions.

Forward:

```text
x
 ↓
Q K V
 ↓
attention calculation
 ↓
attention output
```

Backward:

```text
Loss
 ↓
gradient through attention output
 ↓
gradients through attention calculations
 ↓
gradients for Q/K/V projections
 ↓
gradients for attention parameters
```

So Attention is not just an inference mechanism. Its parameters are learned during training through backpropagation.

---

# 28. The complete mental model

Keep these three levels in mind.

## Level 1 — Whole model

```text
Input
 ↓
Embedding
 ↓
Block × N
 ↓
Final Norm
 ↓
LM Head
 ↓
Output
```

## Level 2 — One block

```text
Input
 ↓
Norm
 ↓
Attention
 ↓
Residual
 ↓
Norm
 ↓
MLP
 ↓
Residual
 ↓
Output
```

## Level 3 — Attention

```text
Input
 ↓
Q K V
 ↓
Attention scores
 ↓
Softmax
 ↓
Weighted V
 ↓
Attention output
```

These are not three competing diagrams.

They are **three zoom levels of the same architecture**.

```text
WHOLE MODEL
│
├── Embedding
│
├── BLOCK 1
│    ├── Norm
│    ├── Attention
│    │    ├── Q/K/V
│    │    ├── Scores
│    │    └── Weighted V
│    ├── Residual
│    ├── Norm
│    ├── MLP
│    │    ├── Linear
│    │    ├── GELU
│    │    └── Linear
│    └── Residual
│
├── BLOCK 2
│    └── same structure
│
├── ...
│
├── BLOCK N
│
├── Final Norm
│
└── LM Head
```

---

# 29. The two implementations you showed

## Model A — newer/custom model

```text
Token Embedding
      ↓
Transformer Blocks × N
      │
      ├── RMSNorm
      ├── Attention + RoPE
      ├── Residual
      ├── RMSNorm
      ├── SwiGLU
      └── Residual
      ↓
Final RMSNorm
      ↓
LM Head
```

It uses:

- RoPE for position
- RMSNorm
- SwiGLU
- SDPA attention
- optional KV caching for generation

## Model B — TinyStoriesGPT

```text
Token Embedding
      +
Position Embedding
      ↓
Transformer Blocks × N
      │
      ├── LayerNorm
      ├── Attention
      ├── Residual
      ├── LayerNorm
      ├── GELU MLP
      └── Residual
      ↓
Final LayerNorm
      ↓
LM Head
```

It uses:

- learned absolute position embeddings
- LayerNorm
- GELU MLP
- either PyTorch MultiheadAttention or SDPA
- optional gradient checkpointing

The **overall architecture is still the same basic GPT/decoder-only Transformer pattern**.

---

# 30. The simplest vocabulary to remember

| Term | Simple meaning |
|---|---|
| Token | Piece of text represented by an ID |
| Embedding | Converts token ID into a vector |
| Position Embedding / RoPE | Gives the model position information |
| Attention | Lets tokens exchange/retrieve information from other tokens |
| Q/K/V | Internal representations used by attention |
| Norm | Keeps representations numerically well-behaved |
| Residual | Keeps old representation and adds new information |
| MLP | Small neural network that transforms each token representation |
| GELU | Nonlinear activation used inside an MLP |
| SwiGLU | Alternative gated MLP design |
| Block | Attention + MLP + Norms + Residuals |
| Layer | Often means one Transformer block in LLM discussions |
| `num_layers` | Number of sequential Transformer blocks |
| LM Head | Converts hidden representation to vocabulary logits |
| Logits | Scores for possible next tokens |
| Forward | Input → output/loss computation |
| Backward | Loss → gradients through the computation graph |
| Optimizer | Uses gradients to update weights |
| KV Cache | Reuses previous K/V during generation |
| Gradient Checkpointing | Saves memory by recomputing activations during backward |

---

# 31. The one picture to remember

If all the details become confusing, return to this:

```text
                         LLM
                          │
                          ▼
                     Token IDs
                          │
                          ▼
                      Embedding
                          │
                          ▼
                 ┌─────────────────┐
                 │ Transformer     │
                 │ Block            │
                 │                 │
                 │ Norm            │
                 │  ↓              │
                 │ Attention       │ ← tokens communicate
                 │  ↓              │
                 │ Residual        │ ← keep old + new
                 │  ↓              │
                 │ Norm            │
                 │  ↓              │
                 │ MLP             │ ← process information
                 │  ↓              │
                 │ Residual        │ ← keep old + new
                 └─────────────────┘
                          │
                          ▼
                    repeat N times
                          │
                          ▼
                      Final Norm
                          │
                          ▼
                       LM Head
                          │
                          ▼
                        Logits
                          │
                          ▼
                   Next-token prediction
```

The most important conceptual distinction is:

> **Attention gathers information from other tokens. MLP processes the representation. Norm keeps the numerical computation stable. Residual connections preserve the existing representation while adding the newly computed information. A Transformer block packages these operations together, and many such blocks are stacked to form the LLM.**

---

# Part 2 — From concept to real PyTorch, class by class

Everything above is the *what* and *why*, deliberately independent of PyTorch mechanics ("the exact mathematical details can be learned separately"). This part is the *how*: the actual PyTorch machinery that turns those diagrams into a runnable, trainable model, walked through against this repo's own real, complete implementation —
[`custom-gpt-6m/src/gpt/model.py`](../src/gpt/model.py) (Model B / `TinyStoriesGPT`, 197 lines, fully self-contained) — with real code, real tensor shapes, and a worked example of modifying it into a genuinely different architecture. Read Part 1 first if you haven't; this part assumes you already have the conceptual model in your head and is about grounding it in code.

---

# 32. What `nn.Module` actually does (the thing that makes everything "just work")

Every learnable piece in PyTorch — a whole model, one block, one linear layer — subclasses `nn.Module`. Two lines make this work:

```python
class GPTBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, attn_impl="naive", causal=True):
        super().__init__()          # <-- sets up nn.Module's internal bookkeeping
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = CausalSelfAttention(embed_size, num_heads, dropout, attn_impl=attn_impl, causal=causal)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, dropout)
```

`super().__init__()` creates a few internal dictionaries (`_parameters`, `_modules`, `_buffers`) that you never touch directly. The trick: **any time you write `self.something = X` inside `__init__`, `nn.Module.__setattr__` intercepts that assignment** and, if `X` is itself an `nn.Module` (like `nn.LayerNorm(...)` or `CausalSelfAttention(...)`) or an `nn.Parameter`, it registers `X` into one of those internal dictionaries instead of just setting a plain Python attribute. This is why a single call like `model.parameters()` on `TinyStoriesGPT` — the top-level model — returns every weight in every nested submodule, all the way down through `blocks[3].mlp.net[0].weight`, with **zero manual bookkeeping code written anywhere in this file**. The nesting diagram from Part 1 (whole model → block → attention) is not just a conceptual picture — it is *literally* how `nn.Module` organizes its internal registry, one level per `self.x = SomeModule(...)` assignment.

Concretely, from this file:

| Attribute | What gets registered | Where |
|---|---|---|
| `self.token_emb`, `self.pos_emb` | Two `nn.Embedding` modules, each holding one weight matrix | `TinyStoriesGPT.__init__` |
| `self.blocks` | An `nn.ModuleList` of `num_layers` separate `GPTBlock` instances | `TinyStoriesGPT.__init__` |
| `self.ln_f` | One `nn.LayerNorm` | `TinyStoriesGPT.__init__` |
| `self.lm_head` | One `nn.Linear` (see weight tying, below) | `TinyStoriesGPT.__init__` |

`nn.ModuleList` matters specifically because a plain Python `list` of modules would **not** be registered — `nn.Module.__setattr__`'s interception only triggers on assignment, and a bare list assigned to `self.blocks` looks like an opaque object, not something to recurse into. This is a real, common beginner bug: write `self.blocks = [GPTBlock(...) for _ in range(num_layers)]` (a plain list) instead of `nn.ModuleList([...])`, and the model will still *run* (Python doesn't stop you), but `.parameters()` silently won't find any of those blocks' weights — the optimizer would update nothing inside them, and training would look like it's working (loss prints, no error) while doing nothing useful for most of the network's parameters.

## `module(x)` vs `module.forward(x)` — never call `forward` directly

Every class here defines `forward()`, but the code always calls the module itself — `self.attn(x)`, `self.mlp(x)`, `block(h)` — never `self.attn.forward(x)`. The reason: `nn.Module` defines `__call__`, and `__call__` is what actually runs when you write `module(x)`. It calls your `forward()` internally, but *around* that call it also: runs any registered forward hooks, and — critically for training — is the point at which the autograd engine records this call as a node in the computation graph it will later walk backward through. Calling `.forward()` directly skips that bookkeeping. It will often still *produce the same output number*, but gradients may not flow through it correctly, or hooks (used for things like activation logging or LoRA-style adapters) silently never fire. This is why "always call the module, never call `.forward` directly" is a real, load-bearing PyTorch convention, not just a style preference.

---

# 33. `nn.Parameter`, plain tensors, and buffers — three different things stored inside a module

- **`nn.Parameter`**: a tensor with `requires_grad=True` by default, and — the part that matters — one that `nn.Module.__setattr__` specifically recognizes and puts in `_parameters`, not `_buffers` or a plain attribute. This is what makes something "learnable": the optimizer only ever updates tensors it finds by walking `.parameters()`. You almost never construct `nn.Parameter` directly in this codebase, because `nn.Linear`, `nn.Embedding`, and `nn.LayerNorm` already wrap their own internal weight (and, for `Linear`/`LayerNorm`, bias) as `nn.Parameter` for you — that's *why* using these built-in layers instead of hand-rolling matrix multiplies gives you working `.parameters()`/`.state_dict()`/optimizer integration for free.
- **A plain tensor**: has no special registration at all. If you compute `pos = torch.arange(seq_len, device=x.device)` inside `forward()` (as `encode()` does), that's just an ordinary tensor — it's not a parameter, isn't saved in `state_dict()`, and the model doesn't "remember" it between calls. Correct here, since position indices are recomputed fresh every forward pass, not learned.
- **A buffer** (`self.register_buffer(name, tensor)`): the middle case — tracked by the module (moves with `.to(device)`, appears in `state_dict()` by default), but **not** a parameter, so the optimizer never touches it. The sibling `custom-gpt-350m-ddp` project uses this for its precomputed RoPE cos/sin tables:

  ```python
  # custom-gpt-350m-ddp/src/gpt/model.py — TinyGPT.__init__
  cos, sin = build_rope_cache(head_dim, context_length, rope_theta)
  self.register_buffer("rope_cos", cos, persistent=False)
  self.register_buffer("rope_sin", sin, persistent=False)
  ```

  This is the correct home for anything that's *derived, fixed math* (RoPE's rotation angles depend only on position and a formula, never learned from data) but still needs to physically live on the same device as the rest of the model and be reconstructible from a checkpoint. `persistent=False` additionally means: don't even bother saving it in `state_dict()` — recompute it fresh from `context_length`/`rope_theta` every time the model is built, since it's cheap and 100% deterministic from those two numbers anyway.

**Rule of thumb for your own new component**: if a tensor should change during training via gradient descent → make it (or let a built-in layer make it) an `nn.Parameter`. If it's fixed, derived, non-learned, but needs `.to(device)` to follow the model → `register_buffer`. If it's just scratch computation inside `forward()` → a plain tensor, no registration needed at all.

---

# 34. `TinyStoriesGPT`, line by line, with real tensor shapes

Shapes below use `B` = batch size, `T` = sequence length (≤ `context_length`), `E` = `embed_size`, `H` = `num_heads`, `D` = `head_dim = E / H`, `V` = `vocab_size`.

### `CausalSelfAttention.__init__` — two attention implementations, one interface

```python
if attn_impl == "naive":
    self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads,
                                       dropout=dropout, batch_first=True)
else:
    self.in_proj = nn.Linear(embed_size, 3 * embed_size, bias=True)   # fused Q,K,V projection
    self.out_proj = nn.Linear(embed_size, embed_size, bias=True)
    self.attn_dropout_p = dropout
```

`nn.MultiheadAttention` is a single built-in PyTorch layer that already contains its own Q/K/V projections, the score/softmax/weighted-sum math, and an output projection — you hand it raw embeddings and it does everything. The `sdpa` branch instead builds the pieces by hand, because `F.scaled_dot_product_attention` (used below) is a *kernel*, not a layer: it expects tensors that are **already** split into `(B, H, T, D)` per-head shape — it has no projection weights of its own. `self.in_proj` is one `nn.Linear(E, 3E)` rather than three separate `nn.Linear(E, E)` calls purely as an efficiency trick (one larger matmul instead of three smaller ones, same total math) — this is "the same trick `nn.MultiheadAttention` uses internally," per the code's own comment.

### `CausalSelfAttention.forward` — the sdpa path, shape by shape

```python
def forward(self, x):
    batch, seq_len, _ = x.shape                      # x: (B, T, E)
    ...
    qkv = self.in_proj(x)                             # (B, T, 3E)
    q, k, v = qkv.chunk(3, dim=-1)                     # each: (B, T, E)
    q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
    k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
    v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

    out = F.scaled_dot_product_attention(q, k, v,
              dropout_p=self.attn_dropout_p if self.training else 0.0,
              is_causal=self.causal)                   # (B, H, T, D)
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_size)  # (B, T, E)
    return self.dropout(self.out_proj(out))            # (B, T, E)
```

Reading this shape by shape is the single most useful exercise for actually understanding attention as code, not just diagram:
1. `chunk(3, dim=-1)` splits the fused `(B, T, 3E)` projection back into three `(B, T, E)` tensors — Q, K, V — purely by slicing, no computation.
2. `.view(B, T, H, D).transpose(1, 2)` is the "split into heads" step from Part 1's Q/K/V diagram, made concrete: each of the `E` embedding dimensions is reinterpreted as `H` independent heads of `D` dimensions each, then the head dimension is moved next to the batch dimension (`(B, H, T, D)`) because that's the shape `F.scaled_dot_product_attention` requires — it treats every `(B, H)` pair as its own independent attention computation over `T` positions.
3. `F.scaled_dot_product_attention(q, k, v, is_causal=True)` is doing, in one fused/optimized kernel call, exactly the "scores → softmax → weighted V" sequence from Part 1's Q/K/V diagram — `is_causal=True` is what enforces the "can't see the future" triangular mask from Part 1's section 9, without ever materializing that mask as an actual `(T, T)` tensor (that's the memory-efficiency reason `sdpa` exists as an alternative to the `naive` path at all).
4. `.transpose(1, 2).contiguous().view(...)` undoes step 2 — merge the `H` heads of `D` dimensions each back into one `E`-dimensional vector per token, so the rest of the network (which only knows about `(B, T, E)` shaped tensors) can keep working on it unchanged. `.contiguous()` is required here because `.transpose()` alone doesn't move any actual memory — it just changes how the tensor's existing memory is *read* — and `.view()` (unlike `.reshape()`) requires the underlying memory to already be laid out contiguously; skipping `.contiguous()` here is a real, common shape-juggling bug that throws a runtime error the first time you hit this exact pattern.

### The naive path's mask, precisely

```python
attn_mask = torch.triu(
    torch.full((seq_len, seq_len), float("-inf"), device=x.device),
    diagonal=1,
)
```

`torch.full((T, T), -inf)` starts with a `T×T` grid entirely filled with `-inf`. `torch.triu(..., diagonal=1)` keeps only the **upper triangle strictly above the main diagonal** (the "diagonal=1" offset) and zeroes out everything else — so the result is `0` everywhere a token is allowed to look (itself and the past) and `-inf` everywhere it isn't (the future), matching Part 1 section 9's ✓/✗ table exactly. Adding `-inf` to an attention score before softmax forces that score's post-softmax weight to exactly `0` — softmax of `-inf` is `0` regardless of what else is in the row, which is the actual mechanism "can't cheat by looking at the future" boils down to in real numbers, not just diagram arrows.

### `MLP` — `nn.Sequential`, the simplest container

```python
self.net = nn.Sequential(
    nn.Linear(embed_size, 4 * embed_size),
    nn.GELU(),
    nn.Linear(4 * embed_size, embed_size),
    nn.Dropout(dropout),
)
def forward(self, x):
    return self.net(x)
```

`nn.Sequential` is `nn.Module`'s simplest container: it just calls each child in order, feeding each one's output as the next one's input — no branching, no residuals, nothing clever. It's the right tool exactly when a component really is "do these operations one after another with nothing else going on," which is true for this MLP but is **not** true for `GPTBlock` (which needs the residual `+` and two different inputs going to `ln_1`/`ln_2`) — that's why `GPTBlock.forward` is written out explicitly instead of also being an `nn.Sequential`.

### `TinyStoriesGPT.encode` — the whole model, shape by shape

```python
def encode(self, x):                                  # x: (B, T) token IDs, dtype long
    _, seq_len = x.shape
    pos = torch.arange(seq_len, device=x.device)       # (T,)
    h = self.token_emb(x) + self.pos_emb(pos)           # (B,T,E) + (T,E) -> broadcasts to (B,T,E)
    h = self.drop(h)
    for block in self.blocks:
        h = block(h)                                    # (B,T,E) -> (B,T,E), shape never changes
    return self.ln_f(h)                                 # (B,T,E)

def forward(self, x):
    return self.lm_head(self.encode(x))                 # (B,T,E) -> (B,T,V)
```

Two shape facts worth internalizing: **every block preserves the exact shape `(B, T, E)`** — this is precisely what makes stacking `num_layers` of them trivial (`for block in self.blocks: h = block(h)`, no shape bookkeeping needed between blocks, ever) — and **only the very last step, `lm_head`, changes the last dimension**, from `E` (an internal representation size nobody outside the model needs to know) to `V` (one score per vocabulary word, the thing `torch.nn.functional.cross_entropy` or top-k sampling actually needs). `self.token_emb(x) + self.pos_emb(pos)` relies on ordinary broadcasting: `pos_emb(pos)` is shape `(T, E)` (no batch dimension — position indices are the same for every sequence in the batch), and PyTorch automatically broadcasts it against `token_emb(x)`'s `(B, T, E)` by treating the missing batch dimension as "repeat for every item in the batch."

---

# 35. Weight init and `.apply()` — why every fresh model starts from *controlled* randomness, not the same randomness every layer

```python
self.apply(self._init_weights)

def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

`nn.Module.apply(fn)` recursively calls `fn` on **every submodule in the tree** — every `nn.Linear` inside every `GPTBlock`'s attention and MLP, every `nn.Embedding`, all the way down — exactly the same recursive-registry mechanism from section 32, now used to *initialize* rather than just *discover* parameters. Without calling this, every `nn.Linear`/`nn.Embedding` would keep PyTorch's own default initialization instead (which is a reasonable general-purpose default, but not tuned to this specific model family) — GPT-style models conventionally initialize weights from a narrow `N(0, 0.02²)` normal distribution specifically because it keeps the *variance* of activations flowing through many stacked layers from exploding or vanishing before training has had a chance to learn anything better — this is a real, measured, widely-adopted convention (originating from the GPT-2 paper), not an arbitrary choice of number.

---

# 36. Autograd, traced through this model's own weight tying

```python
self.lm_head.weight = self.token_emb.weight
```

This single line means `lm_head` and `token_emb` are not two separate parameter tensors that happen to start equal — they are **the literal same tensor object**, referenced from two places. Autograd's consequence: during `loss.backward()`, gradients computed via the `lm_head` path (from the final logits, going backward) and gradients that would flow to `token_emb` via the normal embedding-lookup path **both accumulate into that one shared `.grad` tensor** — PyTorch doesn't know or care that a tensor is reachable from two different modules; it just sums every gradient contribution that flows into a given leaf tensor during the backward walk. So weight tying isn't just "share memory to save parameters" (though it does that too — one `V×E` matrix instead of two) — it also means the embedding table is trained by **two** loss signals every step (how well it embeds an input token, and how well its rows work as an output-vocabulary scoring matrix) instead of one, which is part of why it's a real quality technique, not just a memory optimization.

---

# 37. Worked example: modifying this model into a different real architecture

This is the practical payoff — an actual, runnable modification, not a hypothetical. The sibling `custom-gpt-350m-ddp` project already made exactly this set of changes for real (see its own [`src/gpt/model.py`](../../custom-gpt-350m-ddp/src/gpt/model.py)); reproduced and explained here as a guided exercise rather than just linked, so you can make the same edits yourself and see why each one is safe.

## Modification 1 — swap the GELU MLP for a gated SwiGLU MLP

Current (`MLP`, this file):
```python
class MLP(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
```

New — three matrices instead of two, a *gating* multiplication instead of a plain activation:

```python
class SwiGLU(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout):
        super().__init__()
        self.gate = nn.Linear(embed_size, hidden_size, bias=False)
        self.up   = nn.Linear(embed_size, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
```

Why this is a safe, self-contained swap: `SwiGLU.forward` still takes `(B, T, E)` in and returns `(B, T, E)` out — the exact same shape contract `MLP.forward` had — so `GPTBlock.forward`'s `x = x + self.mlp(self.ln_2(x))` line needs **zero changes**; only the `self.mlp = MLP(embed_size, dropout)` line in `GPTBlock.__init__` needs to become `self.mlp = SwiGLU(embed_size, hidden_size, dropout)`. This is the generalizable lesson: **as long as a replacement component preserves its predecessor's input/output shape contract, everything around it in the surrounding class is untouched.**

Why `hidden_size` is usually **not** still `4 * embed_size`: a GELU MLP is 2 matrices of `E × 4E` each = `8E²` parameters total. SwiGLU is 3 matrices; to spend the *same* total parameter budget, solve `3 × E × hidden_size = 8E²` → `hidden_size = 8E/3 ≈ 2.67E`, not `4E` — using `4E` with 3 matrices instead of 2 would silently make the model ~50% bigger than intended for its "size class," changing training cost and memory without changing the size number anyone thinks they configured. This exact `8/3` reasoning is in the sibling project's own model.py docstring — a real, load-bearing detail if you make this change, not a rounding choice.

## Modification 2 — swap `nn.LayerNorm` for `RMSNorm`

```python
class RMSNorm(nn.Module):
    """LayerNorm centres (subtracts the mean) and shifts (adds a learned bias);
    RMSNorm does neither — it only rescales by the root-mean-square, with one
    learned per-channel scale and no bias at all."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # <-- a real, direct nn.Parameter use

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
```

This is the one place in this whole worked example where you *do* reach for `nn.Parameter` directly (section 33's "almost never" case) — there's no built-in `RMSNorm` layer to delegate to as of older PyTorch versions (newer PyTorch releases do ship `nn.RMSNorm` — check `torch.__version__` and `dir(nn)` before hand-rolling this if you're on a recent enough version), so the one learnable tensor (`weight`, one scale value per channel) is created and registered by hand. `self.weight = nn.Parameter(...)` inside `__init__` is exactly the `nn.Module.__setattr__` interception from section 32 firing on a raw `nn.Parameter` instead of a nested `nn.Module` this time. Swap-in is identical in spirit to Modification 1: replace `self.ln_1 = nn.LayerNorm(embed_size)` / `self.ln_2 = nn.LayerNorm(embed_size)` (and the model-level `self.ln_f`) with `RMSNorm(embed_size)` — same `(B,T,E) -> (B,T,E)` contract, same call sites, no other code changes needed.

## Modification 3 — swap learned position embeddings for RoPE (the bigger one)

This one is *not* a drop-in shape-preserving swap like the first two — it changes *where* position information enters the model (Part 1, section 6): learned `pos_emb` is added once, at the input, before any blocks run; RoPE instead rotates Q and K **inside every block's attention call**, which means `CausalSelfAttention.forward` itself needs a new argument (`cos`, `sin`), not just a component swapped out underneath it. This is the right complexity tier to attempt only after Modifications 1 and 2 feel comfortable — see the sibling project's real, complete implementation (`build_rope_cache`, `apply_rope`, and `CausalSelfAttention.forward`'s `cos`/`sin` parameters, all in `custom-gpt-350m-ddp/src/gpt/model.py`) as the worked reference rather than reproducing it in full here; section 33 above already showed the buffer-registration half of it (`self.register_buffer("rope_cos", ...)`).

## The generalizable checklist for building any new modified model

1. **Identify the shape contract** of the piece you're replacing — what shape does it take in, what shape must it return? (Modifications 1 and 2 above both preserve `(B,T,E) -> (B,T,E)`; that's *why* they're simple.)
2. **Test the new component standalone first**, with a tiny dummy tensor, before wiring it into the full model:
   ```python
   m = SwiGLU(embed_size=64, hidden_size=170, dropout=0.0)
   x = torch.randn(2, 8, 64)     # (B=2, T=8, E=64) — cheap, no real data needed
   out = m(x)
   assert out.shape == x.shape
   ```
   This is the same "verify cheaply before spending real compute" principle behind this whole family of projects' smoke tests (`ddp_smoke_test.py`, `gpt-benchmark`) — a shape bug caught in one line above costs nothing; the same bug caught only after launching a real multi-hour training run costs real time and (on rented hardware) real money.
3. **Wire it into the surrounding class**, changing only the one `self.x = OldComponent(...)` line if the shape contract matches; expect to also change the *caller's* signature (like RoPE's `cos`/`sin`) if it doesn't.
4. **Update any hand-derived parameter-count formula** if one exists in the project (`ModelConfig.param_count()` in the larger sibling projects mirrors the architecture's real matmul shapes by hand specifically so parameter counts are known *before* training, not discovered after — changing the architecture without updating that formula makes the two silently drift apart).
5. **Run one real forward + backward pass on dummy data** through the *whole* model (not just the new component in isolation) before touching a real corpus — `loss = criterion(model(dummy_x), dummy_y); loss.backward()` — confirming gradients actually reach every new parameter (`p.grad is not None for p in new_component.parameters()`) is the cheapest possible check that the new piece is actually wired into the trainable graph, not silently disconnected the way an unregistered plain Python list (section 32) would be.
6. Only then point it at real data and a real training loop.
