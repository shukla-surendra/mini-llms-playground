"""Turn the corpus into random next-token-prediction examples."""

from pathlib import Path

import tiktoken
import torch

from ..config import TOKENIZER_NAME


class TextData:
    def __init__(self, train_path: Path, test_path: Path):
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"{train_path} / {test_path} not found - run `make distill` first to "
                "generate a corpus from a local Ollama teacher."
            )
        self.tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
        train_text = train_path.read_text(encoding="utf-8")
        test_text = test_path.read_text(encoding="utf-8")
        if not train_text.strip() or not test_text.strip():
            raise ValueError(f"{train_path} / {test_path} is empty - run `make distill` first.")
        self.train_ids = torch.tensor(self.tokenizer.encode_ordinary(train_text), dtype=torch.long)
        self.test_ids = torch.tensor(self.tokenizer.encode_ordinary(test_text), dtype=torch.long)

    def get_batch(self, split: str, batch_size: int, context_length: int, device: str):
        data = self.train_ids if split == "train" else self.test_ids
        if len(data) <= context_length:
            raise ValueError(
                f"{split} split has only {len(data)} tokens, shorter than context_length "
                f"{context_length} - generate more distilled data (`make distill`) or "
                "lower context_length."
            )
        starts = torch.randint(len(data) - context_length, (batch_size,))
        x = torch.stack([data[i : i + context_length] for i in starts])
        y = torch.stack([data[i + 1 : i + context_length + 1] for i in starts])
        return x.to(device), y.to(device)
