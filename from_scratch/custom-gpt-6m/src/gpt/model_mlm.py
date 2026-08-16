"""
Bidirectional (BERT-style) masked-LM encoder — reuses model.py's GPTBlock with
causal=False rather than duplicating the Transformer block. See docs/MASKED_LM.md for the
masking policy (BERT's 80/10/10 rule) and why this repo's own BPE tokenizer (trained for
the causal-LM project, no [MASK] token) needs one synthetic reserved id.

The one architectural difference from TinyStoriesGPT (model.py) beyond bidirectionality:
no weight tying. Weight tying (lm_head.weight = token_emb.weight) is a clean fit for
causal LM because the model predicts "the next real vocabulary token," the same space the
input embeddings live in. Here, token_emb has one extra row for the reserved [MASK] id
(index `vocab_size`) that's never a valid *prediction* target — tying would mean either
predicting into a vocab_size+1-wide space (wasting a whole output class on a token that
can never legitimately be the answer) or slicing the tied weight matrix (works, but adds
complexity for no real benefit at this project's scale). Kept separate, deliberately.
"""
import torch
import torch.nn as nn

from .model import GPTBlock


class MaskedLMTinyStories(nn.Module):
    def __init__(self, vocab_size, context_length, embed_size, num_heads, num_layers, dropout,
                 attn_impl="naive"):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size  # reserved id, one past the real tokenizer vocab
        self.context_length = context_length

        # +1 embedding row for the reserved [MASK] id — see module docstring.
        self.token_emb = nn.Embedding(vocab_size + 1, embed_size)
        self.pos_emb = nn.Embedding(context_length, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(embed_size, num_heads, dropout, attn_impl=attn_impl, causal=False) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.mlm_head = nn.Linear(embed_size, vocab_size, bias=True)

        self.apply(self._init_weights)

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
        return self.mlm_head(h)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_mlm_model(vocab_size, context_length=256, embed_size=256, num_heads=8, num_layers=6, dropout=0.1,
                     attn_impl="naive"):
    return MaskedLMTinyStories(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        attn_impl=attn_impl,
    )


def apply_bert_masking(tokens, vocab_size, mask_token_id, mask_prob=0.15, generator=None):
    """
    BERT's 80/10/10 masking rule, applied to a batch of token id tensors.

    Returns (masked_input, labels): `labels` is -100 (PyTorch cross_entropy's default
    ignore_index) everywhere except the mask_prob fraction of *selected* positions, where
    it holds the true original token id — loss is computed only at selected positions,
    exactly like BERT's original pretraining objective.
    """
    labels = tokens.clone()
    select_mask = torch.rand(tokens.shape, device=tokens.device, generator=generator) < mask_prob
    labels[~select_mask] = -100

    masked_input = tokens.clone()
    rand = torch.rand(tokens.shape, device=tokens.device, generator=generator)

    replace_with_mask = select_mask & (rand < 0.8)
    masked_input[replace_with_mask] = mask_token_id

    replace_with_random = select_mask & (rand >= 0.8) & (rand < 0.9)
    random_tokens = torch.randint(0, vocab_size, tokens.shape, device=tokens.device, generator=generator)
    masked_input[replace_with_random] = random_tokens[replace_with_random]

    # remaining 10% of selected positions (rand >= 0.9): left unchanged on purpose.

    return masked_input, labels
