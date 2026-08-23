"""GPT-style decoder architecture — the single source of truth for the model.

    TinyGPT
     |- token_emb / pos_emb       token identity + position
     |- blocks: GPTBlock x N      the stack
     |   |- CausalSelfAttention   F.scaled_dot_product_attention (fused/flash-eligible)
     |   |- MLP                   each token processed independently
     |- ln_f + lm_head            final norm, then next-token logits, weight-tied to
     |                            token_emb

Every dimension comes from config.ModelConfig - nothing here is hardcoded, so changing
the config changes the model with no edits to this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.in_proj = nn.Linear(embed_size, 3 * embed_size, bias=True)
        self.out_proj = nn.Linear(embed_size, embed_size, bias=True)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_size)
        return self.resid_dropout(self.out_proj(out))


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
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = CausalSelfAttention(embed_size, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context_length, embed_size, num_heads, num_layers, dropout):
        super().__init__()
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(context_length, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(embed_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying
        self.apply(self._init_weights)

    @classmethod
    def from_config(cls, model_cfg):
        return cls(
            vocab_size=model_cfg.vocab_size,
            context_length=model_cfg.context_length,
            embed_size=model_cfg.embed_size,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
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

    def forward(self, x, targets=None):
        _, seq_len = x.shape
        assert seq_len <= self.context_length, (
            f"sequence length {seq_len} exceeds context_length {self.context_length}"
        )
        pos = torch.arange(seq_len, device=x.device)
        h = self.drop(self.token_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        logits = self.lm_head(self.ln_f(h))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
