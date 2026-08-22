# `model.py`, top to bottom, for someone new to Transformers

This doc walks through every class in
[`../src/wordgpt/model.py`](../src/wordgpt/model.py) in the order it's defined,
in plain language. It assumes no ML background. For the deepest mechanism inside
this file — the causal Q/K/V attention math — see the dedicated
[`CAUSAL_QKV_ATTENTION.md`](CAUSAL_QKV_ATTENTION.md); this doc recaps that part
briefly and spends most of its time on everything *around* attention: the
residual connections, the embeddings, weight tying, and how a raw integer id
becomes a next-word prediction.

## The one-sentence job of this file

Given a sequence of token ids (numbers standing in for words/punctuation), produce
one score per vocabulary word for **every position**, saying "how likely is each
possible word to come next, right here." Training compares that guess to the real
next word and nudges the weights; generation just reads off the guess at the last
position.

```text
idx: (B, T) integers  --model.py-->  logits: (B, T, vocab_size) scores
```

`B` = batch size (how many independent examples at once), `T` = tokens in the
current window (up to `block_size = 12` in this project's config), `vocab_size`
= how many distinct words/punctuation marks the tokenizer knows (`V` below —
its exact value depends on the corpus; run `make config` to see it).

The file has three classes, and they nest inside each other like this:

```text
GPT
 └── holds a list of n_layer=3 Block
      └── each Block holds one CausalSelfAttention + one MLP
```

## `CausalSelfAttention` — the "which earlier words matter" step

Full deep dive: [`CAUSAL_QKV_ATTENTION.md`](CAUSAL_QKV_ATTENTION.md). Short
version for this walkthrough:

- Every token turns into three vectors — **Query** ("what am I looking for"),
  **Key** ("what do I offer"), **Value** ("what do I actually hand over") — all
  produced by one shared `nn.Linear` (`self.qkv`).
- Comparing every Query against every Key gives a `(T, T)` table of "how relevant
  is token *j* to token *i*."
- The triangular `mask` zeroes out any cell where *j* is in the future relative
  to *i* — a token can never attend to something that hasn't happened yet. This
  is what makes it *causal* self-attention, and it's the reason training on a
  whole sentence at once doesn't let the model cheat by peeking at the answer.
- `softmax` turns those relevance numbers into weights that sum to 1 per row,
  then those weights blend the Value vectors together.

Why split into `n_head = 4` heads of `head_size = 24` each, instead of one
96-wide attention computation? Each head can specialize — one might learn to
track "the word right before me," another "the subject of this sentence" — and
splitting is free: it's the same total numbers, just computed as four smaller,
independent comparisons instead of one big one, then concatenated back together.

## `Block` — one round of "look around, then think"

```python
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # Residual path keeps earlier information available.
        return x + self.mlp(self.ln2(x))
```

A Block does two things to every token's representation, in sequence:

1. **Gather context** — attention mixes in information from earlier tokens.
2. **Think about it privately** — the MLP (`Linear -> GELU -> Linear`) processes
   each token's (now context-aware) vector independently, no cross-token mixing.
   It expands to `4 * n_embd = 384` numbers in the middle, then compresses back
   to 96. The extra width in the middle is where most of a Transformer's actual
   "reasoning capacity" lives — it's a much bigger layer than attention's
   projections, even though attention gets most of the mental picture.

**Why `x = x + attn(...)` instead of `x = attn(...)`?** This is a **residual
(skip) connection**. Instead of replacing a token's representation with
whatever attention/the MLP computed, the sublayer's output is *added* to the
original. Two consequences worth internalizing:

- Each sublayer only has to learn a *correction* to add on top of what's already
  there, not reconstruct the whole representation from scratch — an easier
  learning problem, especially early in training when weights are near-random.
- It gives gradients a direct path backward through every block during
  training (`+` distributes a gradient unchanged to both branches), which is
  why deep stacks of blocks remain trainable at all instead of the signal
  fading out after a few layers.

**Why `LayerNorm` right before each sublayer (`ln1`, `ln2`), not after?** This
is called *pre-norm*. It rescales a token's numbers to a consistent range right
before attention/the MLP look at them, which keeps training numerically stable
as more blocks are stacked. The residual add afterward uses the un-normalized
`x`, so normalization never throws away information — it only affects what the
sublayer *sees*, not what gets carried forward.

Three `Block`s are stacked in `GPT` (`n_layer = 3`), so this "gather context,
then think" cycle happens three times, each time refining the same 96-number
representation a little further.

## `GPT` — wiring the whole model together

```python
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
```

### Two embedding tables, added together

- `token_emb`: a lookup table with one learned 96-number row per vocabulary
  word. Row `stoi["cat"]` is "the model's current idea of what `cat` means,"
  and that row is adjusted by training.
- `pos_emb`: a *separate* lookup table with one learned 96-number row per
  **position** (0 through 11 — `block_size`). Row 0 means "I'm the first token
  in this window," row 3 means "I'm the fourth."

Attention itself has no built-in sense of order — it compares Queries and Keys
regardless of position, so without `pos_emb`, "cat sat" and "sat cat" would look
identical to the model. Adding a position vector to a token vector gives every
input a combined identity: *which word, at which slot*. `the cat` and a
hypothetical second `the` later in the same window get different vectors even
though they're the same token, because their position differs.

```python
positions = torch.arange(T, device=idx.device)
x = self.token_emb(idx) + self.pos_emb(positions)
```

### Weight tying: `lm_head.weight = self.token_emb.weight`

`token_emb` maps *id -> 96 numbers* (going in). `lm_head` maps *96 numbers ->
one score per id* (coming out) — structurally the transpose of the same job.
Both matrices are shaped so that reusing one for the other works, and it's the
same underlying learning signal either way: "make `cat`'s representation
distinctive" helps both recognize `cat` on the way in and predict `cat` on the
way out. Sharing the matrix means:

- Roughly `vocab_size * n_embd` fewer parameters to store and train (that's the
  single biggest weight matrix in a small model like this one).
- Word embeddings get twice the gradient updates per step — once from
  `token_emb`'s use, once from `lm_head`'s — often producing a better-trained
  embedding table for a fixed amount of data, which matters most exactly when
  the corpus is small, as it deliberately is in this project.

### `forward()` — the full pass, with shapes

```python
def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
    B, T = idx.shape
    assert T <= self.cfg.block_size
    positions = torch.arange(T, device=idx.device)
    x = self.token_emb(idx) + self.pos_emb(positions)
    for block in self.blocks:
        x = block(x)
    logits = self.lm_head(self.ln_f(x))
    loss = None if targets is None else F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
    return logits, loss
```

| Step | Shape after | What happened |
|---|---|---|
| `idx` in | `(B, T)` | integer ids, e.g. `(16, 12)` during `make dry-run` |
| `token_emb(idx) + pos_emb(positions)` | `(B, T, 96)` | each id becomes a 96-number vector, position-aware |
| after 3x `block` | `(B, T, 96)` | same shape — refined, not resized, at every block (see `Block` above) |
| `ln_f(x)` | `(B, T, 96)` | one last normalization before scoring |
| `lm_head(...)` | `(B, T, V)` | one score per vocabulary word, at every position |

The final `assert T <= self.cfg.block_size` exists because `pos_emb` only has
rows `0..block_size-1`; feeding a longer sequence would ask it for a position
it was never given a vector for. This is the model's **context window limit**
made concrete — it isn't a soft guideline, it's a lookup table with a fixed
number of rows.

**Why `.reshape(B * T, -1)` before `cross_entropy`?** PyTorch's `cross_entropy`
expects "one row of scores per example" — it doesn't natively know about a
separate `T` dimension. Flattening `(B, T, V)` into `(B*T, V)` treats every
position in every batch item as its own independent prediction (which,
correctly, it is: each position had a genuine next-token target), while
`targets.reshape(B * T)` lines up the matching correct-answer ids the same way.

`targets=None` (the default) skips the loss entirely — that's the path
`generate.py` uses, since there's no known "correct" next word during sampling,
only a prompt to extend.

## `num_parameters()`

```python
def num_parameters(self) -> int:
    return sum(parameter.numel() for parameter in self.parameters())
```

Just a total weight count across every layer — every `nn.Linear`,
`nn.Embedding`, and `nn.LayerNorm` in the model contributes its own weights (and
most, its own biases). With this project's config (`block_size=12, n_embd=96,
n_head=4, n_layer=3`) and `V` = vocab size from the corpus:

```text
token_emb  (V x 96)                     96 * V
pos_emb    (12 x 96)                     1,152
3 x Block:
  ln1                                       192
  attn.qkv    (96 -> 288, +bias)         27,936
  attn.proj   (96 -> 96,  +bias)          9,312
  ln2                                       192
  mlp in      (96 -> 384, +bias)         37,248
  mlp out     (384 -> 96, +bias)         36,960
  ---------------------------------------------
  per block                             111,840
final ln_f                                   192
lm_head                                        0   (reused token_emb.weight)
=================================================
total                              96*V + 336,864
```

Run `make config` to see the real `V` for the bundled corpus and the resulting
total — it prints `vocab_size`, tensor shapes, and the parameter count together
so you can check this table against the live numbers.

Notice the same pattern as most Transformers at any scale: the two MLP layers
inside each block (`37,248 + 36,960 = 74,208`) are the largest single piece of
a block — bigger than attention's own projections (`27,936 + 9,312 = 37,248`).
Stacking more `n_layer` mostly means "more MLP capacity," not "more attention."

## Where to go next

- Run `make dry-run` and match its printed shapes against the `forward()` table
  above — that's the fastest way to confirm this mental model against real
  tensors.
- [`WALKTHROUGH.md`](WALKTHROUGH.md) covers the pipeline this file sits inside:
  tokenizer -> data batching -> `GPT` -> training loop -> generation loop.
- [`CAUSAL_QKV_ATTENTION.md`](CAUSAL_QKV_ATTENTION.md) for the attention
  mechanism specifically, from intuition through the exact code.
