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

    def forward(self, x):
        batch, seq_len, _ = x.shape

        if self.attn_impl == "naive":
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
            return self.dropout(out)

        # sdpa path
        qkv = self.in_proj(x)  # (batch, seq_len, 3*embed_size)
        q, k, v = qkv.chunk(3, dim=-1)
        # (batch, seq_len, embed_size) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_size)
        return self.dropout(self.out_proj(out))


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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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

    def forward(self, x):
        _, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.lm_head(h)
