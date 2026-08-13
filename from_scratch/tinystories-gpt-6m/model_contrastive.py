"""
Contrastive self-supervised objective (SimCSE-style) built on top of the *unchanged*
causal TinyStoriesGPT from model.py — not a new backbone architecture. This is how real
causal-LM-based embedding models work (E5-mistral, LLM2Vec, GTR): take a decoder-only
model that was never trained as an encoder, pool a hidden state from it, and train a small
projection head on top with a contrastive objective. See docs/CONTRASTIVE_LEARNING.md for
the full mechanism and why this differs from a from-scratch bidirectional encoder (which is
what model_mlm.py builds for a different, unrelated reason).

The positive-pair trick (SimCSE, Gao et al. 2021): pass the *same* input through the model
twice in training mode. Dropout is stochastic per forward call, so the two passes produce
two different — but semantically identical, since it's the same input — representations.
Those two become a positive pair; every other sequence in the batch becomes a negative.
No data augmentation, no paired dataset needed — dropout noise alone is the augmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TinyStoriesGPT


class ContrastiveEncoder(nn.Module):
    def __init__(self, vocab_size, context_length, embed_size, num_heads, num_layers, dropout,
                 proj_dim=128, attn_impl="naive"):
        super().__init__()
        # The backbone is a completely ordinary TinyStoriesGPT — same class, same
        # causal-LM architecture used for next-token prediction elsewhere in this project.
        # It's dropout (inherited from this backbone) that makes the SimCSE trick work.
        self.backbone = TinyStoriesGPT(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_size=embed_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            attn_impl=attn_impl,
        )
        self.projection = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, proj_dim),
        )

    def forward(self, x):
        """
        Returns L2-normalized embeddings, one per sequence in the batch — last-token
        pooling (the last position's hidden state has, by construction of causal
        attention, attended to every earlier position, so it's a legitimate whole-sequence
        summary) followed by the projection head.
        """
        hidden = self.backbone.encode(x)       # (batch, seq_len, embed_size)
        pooled = hidden[:, -1, :]               # (batch, embed_size) — last-token pooling
        z = self.projection(pooled)             # (batch, proj_dim)
        return F.normalize(z, dim=-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_contrastive_model(vocab_size, context_length=256, embed_size=256, num_heads=8, num_layers=6,
                             dropout=0.1, proj_dim=128, attn_impl="naive"):
    return ContrastiveEncoder(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        proj_dim=proj_dim,
        attn_impl=attn_impl,
    )


def info_nce_loss(z1, z2, temperature=0.05):
    """
    Symmetric InfoNCE over in-batch negatives (the SimCLR/CLIP formulation). z1[i] and
    z2[i] are a positive pair (same underlying sequence, two dropout-noised passes);
    z1[i] and z2[j] for any j != i are negatives — every other sequence currently in the
    batch, for free, no separate negative-sampling step required.

    Returns (loss, accuracy) — accuracy is "for how many i does the positive pair have
    the single highest similarity of any pair," a directly interpretable retrieval-style
    metric alongside the raw contrastive loss.
    """
    batch_size = z1.size(0)
    logits = z1 @ z2.T / temperature  # (batch, batch); logits[i, j] = sim(z1_i, z2_j)
    labels = torch.arange(batch_size, device=z1.device)

    loss_fwd = F.cross_entropy(logits, labels)
    loss_bwd = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (loss_fwd + loss_bwd)

    with torch.no_grad():
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

    return loss, accuracy
