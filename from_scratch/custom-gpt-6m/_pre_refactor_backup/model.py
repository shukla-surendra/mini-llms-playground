"""
GPT-style decoder-only Transformer, deliberately small.

Same architectural pattern as ../custom-gpt-153m/tiny_llm.py (causal self-attention,
pre-norm residual blocks, weight-tied output head) — see
docs/ARCHITECTURE.md for why every piece here is shaped the way it is, and
../../docs/llm-engineering/10_transformer_architecture.md for the full first-principles
explanation of the mechanism, class by class.

Two dimensions this module is switchable along, both explained in
docs/EFFICIENT_TRAINING.md and docs/MASKED_LM.md respectively:
  - `attn_impl`: "naive" (explicit `nn.MultiheadAttention` + materialized mask, the
    original implementation) vs "sdpa" (`F.scaled_dot_product_attention`, fused/
    flash-attention-eligible kernels, never materializes the full seq_len x seq_len
    mask).
  - `causal`: True (GPT-style, each position only attends to itself and the past) vs
    False (bidirectional encoder, used by the masked-LM variant in model_mlm.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, attn_impl="naive", causal=True):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        if attn_impl not in ("naive", "sdpa"):
            raise ValueError(f"attn_impl must be 'naive' or 'sdpa', got {attn_impl!r}")
        self.attn_impl = attn_impl
        self.causal = causal
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
            attn_mask = None
            if self.causal:
                attn_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                    diagonal=1,
                )
            out, _ = self.attn(
                query=x,
                key=x,
                value=x,
                attn_mask=attn_mask,
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
            is_causal=self.causal,
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
    def __init__(self, embed_size, num_heads, dropout, attn_impl="naive", causal=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = CausalSelfAttention(embed_size, num_heads, dropout, attn_impl=attn_impl, causal=causal)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyStoriesGPT(nn.Module):
    def __init__(self, vocab_size, context_length, embed_size, num_heads, num_layers, dropout,
                 attn_impl="naive", grad_checkpoint=False):
        super().__init__()
        self.context_length = context_length
        self.grad_checkpoint = grad_checkpoint
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(context_length, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(embed_size, num_heads, dropout, attn_impl=attn_impl, causal=True) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

        # Weight tying — see docs/ARCHITECTURE.md
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, x):
        """
        Hidden states after the final LayerNorm, before lm_head — i.e. everything
        forward() does except the vocabulary projection. Used directly for next-token
        logits (forward() below), and reused as-is by model_contrastive.py, which treats
        this causal model as a sequence encoder (last-token pooling + a projection head)
        rather than a next-token predictor. No separate architecture needed for that use
        case — this method is the only thing that changes between the two.
        """
        _, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                # Trades recompute for memory: activations for this block aren't kept
                # around for backward, they're recomputed from the block's input during
                # the backward pass instead. See docs/EFFICIENT_TRAINING.md.
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.ln_f(h)

    def forward(self, x):
        return self.lm_head(self.encode(x))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_model(vocab_size, context_length=256, embed_size=256, num_heads=8, num_layers=6, dropout=0.1,
                 attn_impl="naive", grad_checkpoint=False):
    return TinyStoriesGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        attn_impl=attn_impl,
        grad_checkpoint=grad_checkpoint,
    )


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
