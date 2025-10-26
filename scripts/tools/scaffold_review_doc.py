#!/usr/bin/env python3
"""
Scaffold Review Doc
-------------------

Creates a review doc in the current branch at:
  Documents/reviews/REVIEW_<topic>_<YYYY-MM-DD>.md

Usage:
  python -m scripts.tools.scaffold_review_doc --topic claude_queue_dashboard --pr 123 --branch claude/queue-dashboard-... --commit <sha>

The scaffold enforces location in the CURRENT BRANCH, ensuring reviews live
with the code being reviewed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


TEMPLATE = """## Review – {topic} ({date})

Branch: {branch}
PR: {pr_url}
Head: {short_sha} ({full_sha})

### Summary
- 

### Findings
1)  – 
   - File:line – 
   - Fix:

### Actionable Diffs (sketch)
```diff
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a review doc in the current branch")
    parser.add_argument("--topic", required=True, help="Short topic slug (e.g., claude_queue_dashboard)")
    parser.add_argument("--pr", required=False, help="PR number or full URL")
    parser.add_argument("--branch", required=True, help="Target branch being reviewed")
    parser.add_argument("--commit", required=False, default="", help="Head commit full SHA")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    today = datetime.utcnow().strftime("%Y-%m-%d")
    reviews_dir = repo_root / "Documents" / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)

    short_sha = args.commit[:9] if args.commit else ""
    full_sha = args.commit or ""
    pr_url = args.pr if (args.pr and args.pr.startswith("http")) else (f"https://github.com/<org>/<repo>/pull/{args.pr}" if args.pr else "")

    path = reviews_dir / f"REVIEW_{args.topic}_{today}.md"
    if path.exists():
        print(f"File already exists: {path}")
        return

    content = TEMPLATE.format(
        topic=args.topic,
        date=today,
        branch=args.branch,
        pr_url=pr_url,
        short_sha=short_sha,
        full_sha=full_sha,
    )
    path.write_text(content)
    print(f"Created: {path}")


if __name__ == "__main__":
    main()


