"""`gpt-data` — download TinyStories, train the tokenizer, write tokenized train/val bins."""

from ..data.prepare import main as _main


def main():
    _main()


if __name__ == "__main__":
    main()
