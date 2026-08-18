"""
The model itself: token -> embedding -> N Transformer blocks -> next-token prediction.

Written by hand (no `torch.nn.MultiheadAttention`, no `torch.nn.TransformerEncoder`, no
fused `F.scaled_dot_product_attention`) so every matrix multiply that makes up
"attention" is visible as an actual line of code with an actual tensor shape, not hidden
inside a single library call. `docs/llm-engineering/10_transformer_architecture.md` and
`docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md` (repo root) cover the
theory and the faster fused kernel this hand-written version is *equivalent to* — read
this file to see the mechanism, read those for the bigger picture and why production
code (the other five projects here) calls the fused version instead.

Shapes referenced everywhere below, so they're worth fixing once:
    B = batch_size    — how many independent training examples we process at once
    T = sequence length (<= block_size) — how many tokens are in each example
    C = n_embd         — the size of the vector representing each token
    H = n_head          — how many attention heads C gets split into
    hs = C // H         — how many dimensions each individual head gets
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """
    WHAT this layer does, in one sentence: for every token, decide how much to "borrow"
    from every *earlier* token's representation, and mix them together accordingly.

    WHY: a token's meaning depends on context. The word "bank" needs nearby words
    ("river bank" vs. "savings bank") to know which meaning is meant. Attention is the
    mechanism that lets every token look back at every earlier token and pull in
    whichever information is actually relevant to it — and, crucially, *how much*
    relevant (a learned weighting), not just a fixed nearby-words average.

    WHY "causal": this model predicts the *next* token from everything before it. If
    token 5 were allowed to attend to token 8, it would be cheating — using information
    from the future to predict the future. The mask below makes that structurally
    impossible: token 5 can only ever attend to tokens 0..5.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide evenly across n_head"
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head

        # One big Linear layer that produces Query, Key, and Value all at once (3x
        # n_embd out-features) rather than three separate Linear layers — mathematically
        # identical to three separate projections, just one matrix multiply instead of
        # three, which is friendlier to the GPU/accelerator.
        #   Query = "what am I looking for?"      (per token)
        #   Key   = "what do I contain?"           (per token, compared against queries)
        #   Value = "what do I actually hand over if picked?" (per token, the payload)
        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)

        # After attention mixes information across tokens, this layer gives the model
        # one more learned transformation to combine the heads' outputs before they're
        # added back into the residual stream (see Block below).
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        # A lower-triangular matrix of 1s: mask[i, j] == 1 means "token i is ALLOWED to
        # attend to token j". Row i has 1s in columns 0..i and 0s after — exactly the
        # "only look backward, never forward" rule causal attention needs.
        # `register_buffer` (not a plain attribute, not a Parameter): this tensor moves
        # with the model to whatever device you call `.to(device)` with, and gets saved
        # in checkpoints, but gradient descent never updates it — it's a fixed rule, not
        # something the model learns.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding size

        # One matmul, then split the result into three equal C-sized chunks along the
        # last dimension: this is exactly as if we'd run x through three separate
        # Linear(C, C) layers, just computed together.
        q, k, v = self.qkv_proj(x).split(C, dim=2)  # each: (B, T, C)

        # Split each of Q/K/V's C dimensions into `n_head` independent groups of
        # `head_size` dimensions, then move the head dimension before the sequence
        # dimension. Result shape (B, H, T, hs): PyTorch's batched matmul below then
        # treats every (batch, head) pair as its own independent 2D matrix problem —
        # this is *how* multiple heads run "in parallel" without a Python loop.
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, hs)

        # --- The actual "attention" computation ---

        # Step 1: how well does every token's Query match every token's Key?
        # (B,H,T,hs) @ (B,H,hs,T) -> (B,H,T,T): entry [b,h,i,j] is a single number
        # measuring how relevant token j's Key is to token i's Query, for head h.
        att = q @ k.transpose(-2, -1)

        # Step 2: scale down by sqrt(head_size). Without this, as head_size grows, the
        # dot products above grow with it (more terms summed), pushing softmax's input
        # to extreme values -> softmax saturates to near one-hot outputs -> gradients
        # through it vanish. Dividing by sqrt(head_size) keeps the scores in a range
        # where softmax (next step) stays sensitive to differences between them. This
        # is literally where the "Scaled" in "Scaled Dot-Product Attention" comes from.
        att = att / math.sqrt(self.head_size)

        # Step 3: apply the causal mask. Where causal_mask is 0 (token j is in the
        # future relative to token i), overwrite the score with -inf. softmax(-inf) = 0,
        # so those positions contribute exactly nothing to the weighted sum in step 5 —
        # this is what actually enforces "can't see the future", not just a convention.
        att = att.masked_fill(self.causal_mask[:T, :T] == 0, float("-inf"))

        # Step 4: turn the (now causally-masked) scores into a proper probability
        # distribution over "which earlier tokens to attend to", per query token. Each
        # row [b,h,i,:] now sums to 1.
        att = F.softmax(att, dim=-1)

        # Step 5: use those probabilities as weights over the *Value* vectors — this is
        # the actual "gather relevant information" step. (B,H,T,T) @ (B,H,T,hs) ->
        # (B,H,T,hs): token i's new representation is a weighted blend of every earlier
        # token's Value vector, weighted by how relevant step 1-4 decided each one was.
        out = att @ v

        # Undo the head split: move the head dimension back after T, then merge H and
        # hs back into a single C-sized dimension per token — (B,H,T,hs) -> (B,T,H,hs)
        # -> (B,T,C). `.contiguous()` is required because `.transpose` only changes how
        # the tensor is *viewed* in memory, not its actual layout, and `.view` (used by
        # reshape here) needs a contiguous (non-strided) layout to work.
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # One more learned linear transformation before this result gets added back
        # into the residual stream in Block.forward below.
        return self.out_proj(out)


class MLP(nn.Module):
    """
    The "feed-forward" half of a Transformer block. Where attention lets tokens share
    information *with each other*, the MLP processes each token's (now context-enriched)
    representation *independently* — the same small neural network applied to every
    token's vector, one at a time, with no cross-token interaction at all.

    Structurally: expand C -> 4*C, apply a nonlinearity, then project back down 4*C ->
    C. The 4x expansion is the same ratio the original GPT-2/GPT-3 papers used — a wider
    hidden layer gives the network more room to represent complex per-token
    transformations before compressing back to the model's working width `C`.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU: the nonlinearity. Without *some* nonlinearity here, stacking Linear
        # layers would collapse mathematically into one bigger Linear layer — no matter
        # how many you stack, a chain of purely linear transformations is still just
        # one linear transformation. The nonlinearity is what lets depth (more layers)
        # actually buy more representational power instead of being redundant.
        x = F.gelu(self.fc_in(x))
        return self.fc_out(x)


class Block(nn.Module):
    """
    One Transformer block: attention (tokens talk to each other), then an MLP (each
    token thinks alone) — each wrapped in its own LayerNorm-then-residual-add.

    WHY the residual connections (`x + sublayer(norm(x))`, not just `sublayer(norm(x))`):
    they give gradients a direct, unobstructed path back to every earlier layer during
    backpropagation, instead of having to flow through every intervening nonlinearity.
    Without them, stacking many layers (`n_layer` growing) makes training dramatically
    harder — an early insight from image-recognition networks (ResNets) that the
    Transformer architecture borrowed directly. Each block only has to learn a *change*
    to add to `x`, not a full replacement for it.

    WHY LayerNorm comes *before* the sublayer here ("pre-norm") rather than after
    ("post-norm", the original 2017 Transformer paper's choice): pre-norm keeps the
    residual path (the `x +` part) completely untouched by normalization, which in
    practice trains more stably at the model depths and learning rates this codebase
    (and virtually all modern LLMs) actually use. Deep dive on both LayerNorm and this
    specific choice: docs/llm-engineering/35_normalization_and_mlp_variants_rmsnorm_and_swiglu.md.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    The full model: turn token ids into vectors, run them through `n_layer` Transformer
    blocks, and produce a probability distribution over "what token comes next" at
    every position.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token embedding: a learned lookup table, one row (length n_embd) per possible
        # token id. Turns a discrete id (just an index, carrying no information about
        # meaning on its own) into a dense vector the rest of the network can do math
        # on and gradient descent can gradually shape to be meaningful. Deep dive:
        # docs/llm-engineering/05_embeddings_the_general_idea.md.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # Position embedding: attention (see CausalSelfAttention above) treats its
        # input as an unordered *set* of tokens — nothing in the Q/K/V matmuls above
        # depends on token order. Without this, "the cat sat" and "sat the cat" would
        # produce identical attention patterns. Adding a learned vector for "this is
        # position 0 / 1 / 2 / ..." to each token's embedding is what gives the model
        # any notion of order at all. Deep dive (and the RoPE alternative the 200m/350m
        # sibling projects use instead):
        # docs/llm-engineering/11_positional_encoding_variants_rope_and_beyond.md.
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # The output layer: projects each token's final n_embd-sized vector to
        # vocab_size logits (one raw, unnormalized score per possible next token).
        # `bias=False` because its weight is about to be *tied* to token_emb below.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: reuse the token-embedding matrix as the output layer's weight
        # matrix, instead of learning a second, separate vocab_size x n_embd matrix.
        # Intuition: "the vector that represents token X as an input" and "the vector
        # we compare against to decide how much to score token X as an output" are
        # doing a related job, and sharing them saves a large chunk of parameters (at
        # this model's size, this alone saves as many parameters as the token embedding
        # itself — see the README's parameter table) with no measurable quality cost, a
        # trick from the original GPT-2 paper this project reuses.
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        )

        tok_emb = self.token_emb(idx)  # (B, T, C) — "what token is this"
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)  # (T, C) — "what position is this"
        x = tok_emb + pos_emb  # (B, T, C), broadcast over the batch dimension

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy expects (N, num_classes) predictions and (N,) integer
            # targets, so flatten the batch and time dimensions together — every
            # position, across every example in the batch, is just one more
            # independent "predict the next token" classification problem.
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.view(B * T)
            )

        return logits, loss

    def num_parameters(self) -> int:
        """Total learnable scalars in the model. `lm_head.weight` is excluded from the
        count via the `set` below because it's the *same* tensor object as
        `token_emb.weight` (weight tying, see __init__) — counting both would
        double-count roughly a fifth of the model's parameters."""
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
