"""Print the vocabulary and model size without training."""

from .config import GPTConfig
from .data import TextData
from .model import GPT


def main() -> None:
    data = TextData()
    cfg = GPTConfig(vocab_size=data.tokenizer.vocab_size)
    model = GPT(cfg)
    print(f"vocab_size = {cfg.vocab_size} (words and punctuation from data/corpus.txt)")
    print(f"block_size = {cfg.block_size} tokens")
    print(f"n_embd = {cfg.n_embd}; n_head = {cfg.n_head}; head_size = {cfg.n_embd // cfg.n_head}; n_layer = {cfg.n_layer}")
    print(f"parameters = {model.num_parameters():,}")
    print("vocabulary:")
    print(" ".join(data.tokenizer.tokens))


if __name__ == "__main__":
    main()
