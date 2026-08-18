"""
`python -m nanogpt.show_config` (or `make config`) — print the model shape and exactly
where its parameters go, without training anything. Useful for building intuition about
which hyperparameter (config.py) controls which part of the parameter budget before
spending any time on a training run.
"""

from __future__ import annotations

from .config import GPTConfig
from .data import TextData
from .model import GPT


def main() -> None:
    data = TextData()
    cfg = GPTConfig(vocab_size=data.tokenizer.vocab_size)
    model = GPT(cfg)

    token_emb = cfg.vocab_size * cfg.n_embd
    pos_emb = cfg.block_size * cfg.n_embd
    total = model.num_parameters()
    per_block = (total - token_emb - pos_emb) // cfg.n_layer  # final LayerNorm's
    # handful of parameters get folded into this bucket's rounding — negligible.

    print(f"vocab_size  = {cfg.vocab_size}  (from data/corpus.txt's distinct characters)")
    print(f"block_size  = {cfg.block_size}")
    print(f"n_embd      = {cfg.n_embd}")
    print(f"n_head      = {cfg.n_head}  (head_size = {cfg.n_embd // cfg.n_head})")
    print(f"n_layer     = {cfg.n_layer}")
    print()
    print("parameter breakdown:")
    print(f"  token_embedding   {token_emb:>10,}  ({token_emb / total:5.1%})  "
          f"[reused as the output layer — see model.py's weight-tying note]")
    print(f"  position_embedding{pos_emb:>10,}  ({pos_emb / total:5.1%})")
    print(f"  {cfg.n_layer} x transformer_block ~{per_block:>7,} each"
          f"  ({(total - token_emb - pos_emb) / total:5.1%} combined)")
    print(f"  total             {total:>10,}")


if __name__ == "__main__":
    main()
