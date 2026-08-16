"""`gpt-test` — run the curated story-prompt set, render an HTML report."""

from ..evaluation.story_report import main as _main


def main():
    _main()


if __name__ == "__main__":
    main()
