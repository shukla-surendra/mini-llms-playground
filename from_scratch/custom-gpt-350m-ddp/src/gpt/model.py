"""Decoder architecture — the single source of truth for the model.

This project departs from the GPT-2-style block used by the sibling
custom-gpt-{10m,50m,153m} projects. Four changes, each chosen for a reason that
matters at ~200M parameters aimed at reasoning rather than for novelty:

    GPT-2 style (siblings)          here                     why
    ────────────────────────────    ─────────────────────    ──────────────────────────
    learned absolute pos_emb        RoPE                     no hard context ceiling
    LayerNorm (scale + shift)       RMSNorm (scale only)     same quality, fewer ops
    GELU MLP, 4E hidden, 2 mats     SwiGLU, 8/3 E, 3 mats    better quality per param
    biases on every Linear          no biases                free parameters, no cost

    TinyGPT
     |- token_emb                  token identity (position comes from RoPE)
     |- blocks: GPTBlock x N
     |   |- CausalSelfAttention    RoPE-rotated Q/K, SDPA, causal
     |   |- SwiGLU                 gated MLP
     |- norm_f + lm_head           final RMSNorm, then next-token logits

Every dimension comes from config.ModelConfig — nothing here is hardcoded.

## Why RoPE, specifically

The siblings use `nn.Embedding(context_length, embed_size)`: a lookup table with one
row per position. Position 2049 in a model trained at 2048 simply has no row, so the
context window is a permanent, architectural ceiling — `MODEL_SIZING_GUIDE.md` calls
this out as a one-way door. RoPE instead *rotates* Q and K by an angle proportional to
absolute position, so attention scores depend only on the **relative** offset between
two tokens. Nothing is learned per position, so there is no table to run off the end
of, and the same weights can be run at a longer context later (with some degradation,
and better with interpolation) rather than not at all.

## Why SwiGLU at 8/3 E

A GELU MLP is two matrices: `E -> 4E -> E`, i.e. `8E^2` parameters. SwiGLU is three:
gate and up (`E -> f`) plus down (`f -> E`), i.e. `3Ef`. Setting `f = (8/3)E` makes
`3Ef = 8E^2` — identical parameter cost, and empirically better quality. That is why
`ffn_hidden` defaults to roughly `2.67 x embed_size` rather than `4 x`.

## KV caching (generation only)

`TinyGPT.forward(x, past_kv=None, use_cache=False, start_pos=0)` — with `use_cache=False`
(the default), every call is independent and the return value (`logits`) is unchanged,
so training's plain `model(window)` calls are unaffected. With `use_cache=True`, each
layer's already-rotated key/value tensors for `x`'s tokens come back alongside the
output and can be passed in as `past_kv` on the next call, so a 1-new-token decode step
only computes and rotates that one token's Q/K/V instead of redoing every earlier
token's from scratch. Unlike the sibling GPT-2-style projects, there is no attn_impl
branch to worry about here — SDPA is the only attention path, so every model built from
this file supports caching unconditionally (see inference/generate.py's generate_text(),
the only caller that uses this).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square norm: rescale by RMS, no mean subtraction, no bias.

    LayerNorm centres (subtract mean) and shifts (learned bias); RMSNorm does neither.
    The centring turns out not to matter for transformer quality, and dropping it plus
    the bias removes two elementwise passes per norm and one parameter vector per norm.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute in fp32 even under autocast: the mean-square of a bf16 vector loses
        # precision exactly where the norm is most sensitive.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(head_dim, max_seq_len, theta, device=None, dtype=torch.float32):
    """Precompute (cos, sin) of shape (max_seq_len, head_dim/2) for RoPE.

    `theta` sets how fast the rotation frequency decays across dimension pairs: low
    dimensions rotate fast (fine positional detail), high dimensions rotate slowly
    (long-range structure). Raising theta lengthens the slowest wavelength, which is
    the standard lever for extending context after training.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)          # (seq, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate the last dimension of x pairwise. x: (batch, heads, seq, head_dim)."""
    seq_len = x.shape[-2]
    cos = cos[:seq_len].view(1, 1, seq_len, -1)
    sin = sin[:seq_len].view(1, 1, seq_len, -1)
    x1, x2 = x.float().chunk(2, dim=-1)
    # Treat (x1, x2) as a complex number per dimension pair and multiply by e^{i*theta}.
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE, via `F.scaled_dot_product_attention`.

    Unlike the siblings there is no `attn_impl` switch: SDPA is the only path, since
    the naive `nn.MultiheadAttention` route cannot apply RoPE to Q/K without
    reimplementing the projection split anyway.
    """

    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.in_proj = nn.Linear(embed_size, 3 * embed_size, bias=False)
        self.out_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.attn_dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        """KV caching (generation only, see TinyGPT.forward and the module-level note
        in inference/generate.py). `cos`/`sin` are always the caller-sliced tables for
        x's actual absolute positions — see TinyGPT.forward — so apply_rope's own
        internal `[:seq_len]` slice is a no-op here, not a second, different slice.
        `past_kv`, when given, holds already-rotated key/value tensors from earlier
        calls; concatenating them with this call's freshly-rotated k/v is valid because
        RoPE rotates each token by its own absolute position exactly once, and that
        rotation never changes after the fact.
        """
        batch, seq_len, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        # (batch, seq, embed) -> (batch, heads, seq, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Position enters here and nowhere else — Q and K are rotated, V is not.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # is_causal only masks x's OWN tokens against each other (the multi-token
        # prefill case, past_kv=None) — once a cache exists, every key came from a
        # strictly earlier call than this one's queries, so nothing needs masking.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=(past_kv is None),
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_size)
        out = self.dropout(self.out_proj(out))
        return (out, (k, v)) if use_cache else out


class SwiGLU(nn.Module):
    """Gated MLP: `down(silu(gate(x)) * up(x))`.

    The gate branch decides, per hidden unit, how much of the up branch to let through
    — a multiplicative interaction a plain GELU MLP cannot express at the same
    parameter cost. See the module docstring for the 8/3 sizing.
    """

    def __init__(self, embed_size, hidden_size, dropout):
        super().__init__()
        self.gate = nn.Linear(embed_size, hidden_size, bias=False)
        self.up = nn.Linear(embed_size, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class GPTBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ffn_hidden, dropout):
        super().__init__()
        self.norm_1 = RMSNorm(embed_size)
        self.attn = CausalSelfAttention(embed_size, num_heads, dropout)
        self.norm_2 = RMSNorm(embed_size)
        self.mlp = SwiGLU(embed_size, ffn_hidden, dropout)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        attn_out = self.attn(self.norm_1(x), cos, sin, past_kv=past_kv, use_cache=use_cache)
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        x = x + attn_out
        x = x + self.mlp(self.norm_2(x))
        return (x, new_kv) if use_cache else x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context_length, embed_size, num_heads, num_layers,
                 ffn_hidden, dropout, rope_theta=10000.0):
        super().__init__()
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(embed_size, num_heads, ffn_hidden, dropout) for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        # Weight tying, as in the siblings: the same matrix maps ids -> vectors on the
        # way in and hidden states -> logits on the way out. At V=32,768 x E=896 that
        # is 29.4M parameters counted once instead of twice.
        self.lm_head.weight = self.token_emb.weight

        head_dim = embed_size // num_heads
        cos, sin = build_rope_cache(head_dim, context_length, rope_theta)
        # Buffers, not parameters: derived from position, never trained. persistent=False
        # keeps them out of the checkpoint — they are rebuilt from config on load, so a
        # context change does not need a migration.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @classmethod
    def from_config(cls, model_cfg, context_length=None, attn_impl=None):
        """Build from a config.ModelConfig.

        `attn_impl` is accepted and ignored — kept so callers shared with the sibling
        projects (trainer, benchmark) do not need a per-project branch. SDPA is the
        only attention path here.
        """
        return cls(
            vocab_size=model_cfg.vocab_size,
            context_length=context_length or model_cfg.context_length,
            embed_size=model_cfg.embed_size,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            ffn_hidden=model_cfg.ffn_hidden,
            dropout=model_cfg.dropout,
            rope_theta=model_cfg.rope_theta,
        )

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, past_kv=None, use_cache=False, start_pos=0):
        """past_kv/use_cache/start_pos are the incremental-decoding API (generation
        only — see inference/generate.py's generate_text()). `start_pos` is the
        absolute position of x's first token: 0 for a fresh/prefill call, or the
        running sequence length so far for a decode step continuing an existing cache
        — RoPE needs the true absolute position to rotate by, not just x's own local
        offset, exactly like the sibling projects' learned pos_emb does.

        With use_cache=False (the default), behavior and the return value (just
        `logits`) are exactly what they were before caching existed, so training's
        plain `model(window)` calls are entirely unaffected.
        """
        _, seq_len = x.shape
        if start_pos + seq_len > self.context_length:
            raise ValueError(
                f"sequence length {start_pos + seq_len} exceeds context_length "
                f"{self.context_length}; the RoPE cache is only built that far."
            )
        h = self.drop(self.token_emb(x))
        cos = self.rope_cos[start_pos:start_pos + seq_len]
        sin = self.rope_sin[start_pos:start_pos + seq_len]

        new_past_kv = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            out = block(h, cos, sin, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                h, kv = out
                new_past_kv.append(kv)
            else:
                h = out
        h = self.norm_f(h)
        logits = self.lm_head(h)
        return (logits, new_past_kv) if use_cache else logits
