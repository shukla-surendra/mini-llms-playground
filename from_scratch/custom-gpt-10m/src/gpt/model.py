"""GPT-style decoder architecture — the single source of truth for the model.

One model, built from its parts:

    TinyGPT                       the model you train and talk to
     |- token_emb / pos_emb       token identity + position
     |- blocks: GPTBlock x N      the stack
     |   |- CausalSelfAttention   tokens look at earlier tokens
     |   |- MLP                   each token processed independently
     |- ln_f + lm_head            final norm, then next-token logits

Every dimension comes from config.ModelConfig — nothing here is hardcoded, so
changing the preset changes the model with no edits to this file.

`attn_impl` is switchable between "naive" (explicit nn.MultiheadAttention + a
materialized causal mask, the original implementation) and "sdpa"
(F.scaled_dot_product_attention, fused/flash-attention-eligible kernels, never
materializes the full seq_len x seq_len mask) — same math, different memory-access
pattern. See docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md for the
full mechanism, and checkpoint.py's remap_attn_impl for how a checkpoint trained under
one implementation resumes correctly under the other (the two paths use different
parameter names for numerically-identical weights, so a plain load_state_dict across
implementations fails without that remap).

KV caching (generation only). `forward(x, past_kv=None, use_cache=False)` — with
`use_cache=False` (the default), every call is a full, independent forward pass and the
return value is unchanged (`logits`), so training's `model(window)` calls are completely
unaffected by any of this. With `use_cache=True`, each layer's already-computed key/value
projections for `x`'s tokens are returned alongside the output and can be passed back in
as `past_kv` on the next call — so a 1-new-token decode step only computes that one
token's Q/K/V and attends it against the cached K/V from every earlier step, instead of
recomputing every earlier token's K/V from scratch (see inference/generate.py's
generate_text(), the only caller that uses this). Only the "sdpa" path implements this:
nn.MultiheadAttention ("naive") has no public incremental-decoding API, so that path
always does a full recompute regardless of `use_cache` — generate_text() branches on
`model.attn_impl` accordingly, and checkpoint.load_model() defaults inference callers to
"sdpa" (remapping a "naive"-trained checkpoint's weights, since they're numerically
identical either way) specifically so this fast path is what generation actually uses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, attn_impl="naive"):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        if attn_impl not in ("naive", "sdpa"):
            raise ValueError(f"attn_impl must be 'naive' or 'sdpa', got {attn_impl!r}")
        self.attn_impl = attn_impl
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        if attn_impl == "naive":
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            # Manual Q/K/V projection (fused into one matmul, same trick
            # nn.MultiheadAttention uses internally) since F.scaled_dot_product_attention
            # is a pure attention kernel, not a layer — it expects already-projected
            # per-head tensors, not raw embeddings.
            self.in_proj = nn.Linear(embed_size, 3 * embed_size, bias=True)
            self.out_proj = nn.Linear(embed_size, embed_size, bias=True)
            self.attn_dropout_p = dropout

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        batch, seq_len, _ = x.shape

        if self.attn_impl == "naive":
            # No incremental-decoding API on nn.MultiheadAttention — always a full
            # recompute over exactly the tokens in `x`, past_kv is ignored. Correct only
            # when the caller always passes the full sequence-so-far (never a single new
            # token expecting cache reuse) — see generate_text()'s attn_impl branch.
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                diagonal=1,
            )
            out, _ = self.attn(
                query=x,
                key=x,
                value=x,
                attn_mask=causal_mask,
                need_weights=False,
            )
            out = self.dropout(out)
            return (out, None) if use_cache else out

        # sdpa path
        qkv = self.in_proj(x)  # (batch, seq_len, 3*embed_size)
        q, k, v = qkv.chunk(3, dim=-1)
        # (batch, seq_len, embed_size) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # is_causal is only needed to mask x's OWN tokens against each other (the
        # multi-token prefill case, past_kv=None). Once a cache exists, every key came
        # from a strictly earlier call than this one's queries, so nothing needs masking
        # — sdpa with is_causal=False here still expects the same-length q/k when
        # is_causal=True, which wouldn't even apply correctly once k is longer than q.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=(past_kv is None),
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_size)
        out = self.dropout(self.out_proj(out))
        return (out, (k, v)) if use_cache else out


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


class GPTBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, attn_impl="naive"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = CausalSelfAttention(embed_size, num_heads, dropout, attn_impl=attn_impl)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return (x, new_kv) if use_cache else x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context_length, embed_size, num_heads, num_layers, dropout,
                 attn_impl="naive"):
        super().__init__()
        self.context_length = context_length
        self.attn_impl = attn_impl
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(context_length, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(embed_size, num_heads, dropout, attn_impl=attn_impl) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

        # Common in GPT models: tie input embedding and output projection weights.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @classmethod
    def from_config(cls, model_cfg, context_length=None, attn_impl="naive"):
        """Build from a config.ModelConfig.

        `context_length` may be overridden when a tiny dataset can't fill the
        configured window (see data.dataset.effective_context_length).
        """
        return cls(
            vocab_size=model_cfg.vocab_size,
            context_length=context_length or model_cfg.context_length,
            embed_size=model_cfg.embed_size,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            attn_impl=attn_impl,
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
        """past_kv/use_cache/start_pos are the incremental-decoding API — see the class
        docstring's "KV caching" section. `start_pos` is the absolute position of x's
        first token in the sequence (0 for a fresh/prefill call, ids.size(1) - x.size(1)
        for a decode step continuing an existing cache) — position embeddings must
        reflect true position, not just x's own local offset, once x is a short
        continuation rather than the full sequence from the start.
        """
        _, seq_len = x.shape
        pos = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.drop(h)

        new_past_kv = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            out = block(h, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                h, kv = out
                new_past_kv.append(kv)
            else:
                h = out
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return (logits, new_past_kv) if use_cache else logits
