"""`gpt-audit` — corpus quality gate to run before a long training run."""

import argparse

from ..config import load_settings
from ..data.audit import audit, format_report


def main():
    parser = argparse.ArgumentParser(description="Audit the built corpus.")
    parser.parse_args()

    _, _, paths, _ = load_settings()
    print(format_report(audit(paths)))


if __name__ == "__main__":
    main()
