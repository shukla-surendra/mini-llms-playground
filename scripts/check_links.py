#!/usr/bin/env python3
"""Validate relative Markdown links across the repo's docs.

For every `[text](target)` link in a tracked `.md` file, check that a relative file
target actually exists ANYWHERE in the repo — not just within mkdocs' docs_dir. This
repo's curriculum (docs/llm-engineering/) deliberately cross-links to real source files
(model.py, checkpoint.py, ...) outside docs/, which `mkdocs build --strict`'s own
link-checker can't see (it only validates targets within docs_dir) and would otherwise
flag as false-positive "not found" warnings on every single one. External links
(http/https), in-page anchors (`#...`), and `mailto:` are skipped.

Usage:
    python3 scripts/check_links.py
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".venv", ".git", "site", "__pycache__", "target", "node_modules"}

# [text](target) - capture the target, non-greedy, no nested parens.
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def markdown_files():
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in EXCLUDE_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def check() -> int:
    broken: list[str] = []
    files = list(markdown_files())
    for md in files:
        text = md.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            # Strip an optional title: [x](path "title")
            target = target.split(" ", 1)[0]
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            base = target.split("#", 1)[0]  # drop any anchor
            if not base:
                continue
            resolved = (md.parent / base).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")

    if broken:
        print("Broken relative links found:")
        for b in broken:
            print(f"  MISS  {b}")
        return 1
    print(f"OK: all relative Markdown links resolve ({len(files)} files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(check())
