#!/usr/bin/env python3
"""
Handoff Link Pre-Commit Check
-----------------------------

Blocks commits that include PR creation links instead of real PR URLs.

What is blocked:
- Links containing '/pull/new/' (PR creation pages)
- Compare links used for creation (e.g., '/compare/main...<branch>?expand=1')

Allowed examples:
- https://github.com/<org>/<repo>/pull/123

Usage (pre-commit):
  Add a pre-commit hook that invokes:
    python -m scripts.tools.handoff_check

This script scans STAGED changes (git diff --cached) for added/modified
Markdown and text files under Documents/ and .github/ and blocks if creation
links are found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


CREATION_LINK_PATTERNS = [
    re.compile(r"https?://github\.com/[^\s]*/pull/new/", re.I),
    re.compile(r"https?://github\.com/[^\s]*/compare/[^\s]*\?expand=1", re.I),
]

SCAN_FILE_EXT = {".md", ".txt"}
SCAN_PATH_PREFIXES = ("Documents/", ".github/")


def get_staged_files() -> list[Path]:
    # Get added or modified files in index
    res = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
        capture_output=True,
        text=True,
        check=False,
    )
    files = []
    for line in (res.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        files.append(Path(line))
    return files


def read_staged_file(path: Path) -> str:
    # Read content from index (staged version)
    res = subprocess.run(
        ["git", "show", f":{path.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout or ""


def should_scan(path: Path) -> bool:
    return (
        any(str(path).startswith(pfx) for pfx in SCAN_PATH_PREFIXES)
        and path.suffix.lower() in SCAN_FILE_EXT
    )


def has_creation_link(text: str) -> bool:
    for pat in CREATION_LINK_PATTERNS:
        if pat.search(text):
            return True
    return False


def main() -> int:
    failures = []
    for path in get_staged_files():
        if not should_scan(path):
            continue
        content = read_staged_file(path)
        if has_creation_link(content):
            failures.append(path)

    if failures:
        print("\n❌ Handoff check failed: Found PR creation links in staged files:")
        for p in failures:
            print(f"  - {p}")
        print(
            "\nPlease replace creation links with actual PR URLs (e.g., https://github.com/<org>/<repo>/pull/123)."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


