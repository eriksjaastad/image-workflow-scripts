#!/usr/bin/env python3
"""
Generate Archive Cleanup Candidate Report (no deletions)
=======================================================

Scans Documents/archives/* and lists files that exceed retention windows
and are not referenced by any non-archives document. Writes a markdown report.

Retention policy (from Documents/README.md):
  - archives/sessions/: 12 months
  - archives/ai/: 6 months
  - archives/dashboard/: 6 months
  - archives/implementations/: keep indefinitely (excluded)
  - archives/misc/: 3 months

Reference rule: If a file is linked from any non-archives doc, keep it.

Age is determined by the "Last Updated:" line in the document when present,
falling back to filesystem modification time.

Output: data/daily_summaries/doc_cleanup_report_YYYYMMDD.md

No file operations are performed. This is a read-only analyzer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set


RE_LAST_UPDATED = re.compile(r"^(?:\*\*Last Updated:\*\*|Last Updated:)\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate archive cleanup candidates report (no deletes)")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2] / "Documents"),
        help="Path to Documents directory (default: repo/Documents)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "daily_summaries"),
        help="Directory to write the report (default: data/daily_summaries)",
    )
    parser.add_argument(
        "--today",
        default=dt.date.today().isoformat(),
        help="Override today's date (YYYY-MM-DD)",
    )
    return parser.parse_args()


def read_last_updated(md_path: Path) -> dt.date | None:
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for line in text.splitlines()[:50]:
        m = RE_LAST_UPDATED.match(line.strip())
        if m:
            try:
                return dt.date.fromisoformat(m.group(1))
            except Exception:
                return None
    return None


def collect_references(docs_root: Path) -> Set[Path]:
    """Return a set of absolute Paths referenced by non-archive docs.
    Resolves relative links against the source document directory.
    Ignores external links (http:// or https://) and anchors (#...).
    """
    refs: Set[Path] = set()
    for md in docs_root.rglob("*.md"):
        # Skip archives when collecting inbound links
        try:
            rel = md.relative_to(docs_root).as_posix()
        except Exception:
            continue
        if rel.startswith("archives/"):
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in MD_LINK_RE.finditer(text):
            target = m.group(1)
            if "://" in target or target.startswith("#"):
                continue
            # Resolve relative to source doc directory
            candidate = (md.parent / target).resolve()
            if candidate.exists():
                refs.add(candidate)
    return refs


def retention_days(rel_path: str) -> int | None:
    if rel_path.startswith("archives/sessions/"):
        return 365
    if rel_path.startswith("archives/ai/"):
        return 182  # ~6 months
    if rel_path.startswith("archives/dashboard/"):
        return 182
    if rel_path.startswith("archives/implementations/"):
        return None  # keep indefinitely
    if rel_path.startswith("archives/misc/"):
        return 90
    # Default: do not mark as candidate outside known archives
    return None


def main() -> None:
    args = parse_args()
    docs_root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.fromisoformat(args.today)

    inbound_refs = collect_references(docs_root)

    candidates: List[Dict[str, str]] = []
    scanned = 0

    for md in docs_root.rglob("*.md"):
        rel = md.relative_to(docs_root).as_posix()
        if not rel.startswith("archives/"):
            continue
        scanned += 1
        keep_days = retention_days(rel)
        if keep_days is None:
            # Either indefinite retention or out-of-scope
            continue
        # Age determination
        last = read_last_updated(md)
        if last is None:
            # fallback to mtime
            last = dt.date.fromtimestamp(md.stat().st_mtime)
        age_days = (today - last).days
        referenced = md.resolve() in inbound_refs
        if age_days > keep_days and not referenced:
            candidates.append(
                {
                    "path": rel,
                    "last_updated": last.isoformat(),
                    "age_days": str(age_days),
                    "referenced": "no",
                    "threshold_days": str(keep_days),
                }
            )

    # Write markdown report
    report_name = f"doc_cleanup_report_{today.strftime('%Y%m%d')}.md"
    report_path = out_dir / report_name

    lines: List[str] = []
    lines.append("# Documentation Archive Cleanup Candidates\n")
    lines.append(f"Generated: {today.isoformat()}\n")
    lines.append(f"Documents root: {docs_root}\n")
    lines.append(f"Scanned archived docs: {scanned}\n")
    lines.append(f"Candidates: {len(candidates)}\n")
    lines.append("\n")
    if candidates:
        lines.append("| Path | Last Updated | Age (days) | Threshold | Referenced |\n")
        lines.append("|------|--------------|------------|----------|------------|\n")
        for c in sorted(candidates, key=lambda x: (x["path"])):
            lines.append(
                f"| {c['path']} | {c['last_updated']} | {c['age_days']} | {c['threshold_days']} | {c['referenced']} |\n"
            )
    else:
        lines.append("No candidates beyond retention thresholds.\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()


