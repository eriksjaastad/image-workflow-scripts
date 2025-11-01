#!/usr/bin/env python3
"""
Audit Files vs Decisions DB (by project)
=======================================

Purpose:
- Cross-check the decisions DB expected files against what's on disk.
- Quickly detect "expected but missing" and "unexpected extras" issues.

Safe by design:
- Read-only scan of production images and companions
- Writes NEW reports under data/daily_summaries/

Usage:
  python scripts/tools/audit_files_vs_db.py --project mojo3 --write-report

Options:
  --dirs: optional custom list of directories to scan
  --project: project id (e.g., mojo1, mojo2, mojo3)
  --write-report: write JSON + MD report into data/daily_summaries/
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class ExpectedFile:
    filename: str
    action: str  # approve | crop | reject | other


@dataclass
class FoundFile:
    filename: str
    paths: list[str]


def detect_artifact_candidates(images: list[str], paths: list[str]) -> list[str]:
    """Heuristics to flag artifact candidates for a decision row.
    - duplicates across directories
    - mismatched base stems
    Returns list of reason strings
    """
    reasons: list[str] = []
    dirs = {str(Path(p).parent) for p in paths}
    if len(dirs) > 1:
        reasons.append("multi_directory")

    def base_stem(name: str) -> str:
        parts = name.split("_stage")
        return parts[0] if parts else Path(name).stem

    stems = {base_stem(Path(n).name) for n in images if n}
    if len(stems) > 1:
        reasons.append("mismatched_stems")
    return reasons


@dataclass
class AuditSummary:
    project_id: str
    total_decisions: int
    by_action: dict[str, int]
    expected_kept_count: int
    kept_found: int
    kept_missing: int
    rejects_found_anywhere: int
    duplicates: int


def is_image_file(path: Path) -> bool:
    try:
        return path.suffix.lower() in IMAGE_EXTENSIONS
    except Exception:
        return False


def load_expected_from_db(
    project_id: str, db_root: Path
) -> tuple[list[ExpectedFile], dict[str, int]]:
    db_path = db_root / f"{project_id}.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Decisions DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # We need: images (JSON array of filenames), user_selected_index, user_action
    rows = cur.execute(
        """
        SELECT images, user_selected_index, user_action
        FROM ai_decisions
        """
    ).fetchall()
    conn.close()

    expected: list[ExpectedFile] = []
    by_action: dict[str, int] = {}

    for images_json, selected_idx, action in rows:
        try:
            imgs = json.loads(images_json) if images_json else []
        except Exception:
            imgs = []
        action = (action or "").lower().strip()
        by_action[action] = by_action.get(action, 0) + 1

        # Determine selected filename when available
        filename: str | None = None
        if isinstance(selected_idx, int) and 0 <= selected_idx < len(imgs):
            # imgs may be strings or objects; support both
            v = imgs[selected_idx]
            if isinstance(v, str):
                filename = Path(v).name
            elif isinstance(v, dict):
                fn = v.get("filename") or v.get("name")
                if fn:
                    filename = Path(fn).name

        # If no selected filename, skip expectation for this row
        if not filename:
            continue

        # Expectation policy (minimal, pragmatic):
        # - approve, crop → considered "kept"; should exist somewhere among key dirs
        # - reject → may be in delete staging or no longer present (we only count presence)
        # - other → unknown; include but do not enforce
        expected.append(ExpectedFile(filename=filename, action=action))

    return expected, by_action


def index_disk(paths: list[Path]) -> dict[str, list[str]]:
    """Build an index: filename -> list of absolute paths where it exists."""
    idx: dict[str, list[str]] = {}
    for root in paths:
        if not root.exists() or not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if not is_image_file(p):
                continue
            key = p.name
            idx.setdefault(key, []).append(str(p.resolve()))
    return idx


def compute_audit(
    project_id: str,
    expected: list[ExpectedFile],
    disk_index: dict[str, list[str]],
) -> tuple[AuditSummary, dict[str, FoundFile], list[str]]:
    expected_kept = [e for e in expected if e.action in ("approve", "crop")]
    expected_rejected = [e for e in expected if e.action == "reject"]

    kept_found = 0
    kept_missing = 0
    rejects_found_anywhere = 0
    duplicates = 0

    found_details: dict[str, FoundFile] = {}
    problems: list[str] = []

    # Evaluate kept expectations
    for e in expected_kept:
        paths = disk_index.get(e.filename, [])
        if paths:
            kept_found += 1
            if len(paths) > 1:
                duplicates += 1
            found_details[e.filename] = FoundFile(filename=e.filename, paths=paths)
            # Artifact heuristics at the filename-level are weak; defer to decision-level in future.
        else:
            kept_missing += 1
            problems.append(f"Missing kept file: {e.filename}")

    # Evaluate rejects (presence is not an error, but informative)
    for e in expected_rejected:
        paths = disk_index.get(e.filename, [])
        if paths:
            rejects_found_anywhere += 1
            found_details[e.filename] = FoundFile(filename=e.filename, paths=paths)

    total_decisions = len(expected)
    by_action: dict[str, int] = {}
    for e in expected:
        by_action[e.action] = by_action.get(e.action, 0) + 1

    summary = AuditSummary(
        project_id=project_id,
        total_decisions=total_decisions,
        by_action=by_action,
        expected_kept_count=len(expected_kept),
        kept_found=kept_found,
        kept_missing=kept_missing,
        rejects_found_anywhere=rejects_found_anywhere,
        duplicates=duplicates,
    )

    return summary, found_details, problems


def write_reports(
    summary: AuditSummary,
    found: dict[str, FoundFile],
    problems: list[str],
) -> tuple[Path, Path]:
    out_dir = Path("data") / "daily_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"audit_files_vs_db_{summary.project_id}_{ts}"

    # JSON
    json_path = out_dir / f"{base}.json"
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": asdict(summary),
        "found": {k: asdict(v) for k, v in found.items()},
        "problems": problems,
        "artifact_candidates": [],
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown (short)
    md_path = out_dir / f"{base}.md"
    lines: list[str] = []
    lines.append(f"# Audit Files vs DB — {summary.project_id}")
    lines.append("")
    s = summary
    lines.append("## Summary")
    lines.append(f"- decisions: {s.total_decisions}")
    lines.append(f"- by_action: {json.dumps(s.by_action)}")
    lines.append(f"- expected_kept: {s.expected_kept_count}")
    lines.append(f"- kept_found: {s.kept_found}")
    lines.append(f"- kept_missing: {s.kept_missing}")
    lines.append(f"- rejects_found_anywhere: {s.rejects_found_anywhere}")
    lines.append(f"- duplicates: {s.duplicates}")
    lines.append("")
    if problems:
        lines.append("## Problems (first 50)")
        for p in problems[:50]:
            lines.append(f"- {p}")
        if len(problems) > 50:
            lines.append(f"- ... and {len(problems)-50} more")
        lines.append("")
    # Placeholder section for artifact candidates (decision-level when DB flagging is available)
    lines.append("## Artifact Candidates (scaffolding)")
    lines.append(
        "- Detected via multi-directory and mismatched-stem heuristics (to be expanded)"
    )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return json_path, md_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit decisions DB vs on-disk files.")
    ap.add_argument("--project", required=True, help="Project id (e.g., mojo3)")
    ap.add_argument(
        "--dirs",
        nargs="*",
        help="Directories to scan; defaults to common workflow dirs",
    )
    ap.add_argument(
        "--write-report", action="store_true", help="Write JSON + MD report"
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_id: str = args.project

    # Default search dirs (common workflow locations)
    default_dirs = [
        "crop",
        "crop_auto",
        "selected",
        "__crop",
        "__crop_auto",
        "__selected",
        "__cropped",
        "__delete_staging",
        "mojo1",
        "mojo2",
        "mojo3",
    ]
    scan_dirs = [Path(d).resolve() for d in (args.dirs if args.dirs else default_dirs)]

    # Load expected from DB
    db_root = Path("data") / "training" / "ai_training_decisions"
    expected, by_action_db = load_expected_from_db(project_id, db_root)

    # Index disk once
    disk_index = index_disk(scan_dirs)

    # Compute audit
    summary, found, problems = compute_audit(project_id, expected, disk_index)

    # Print quick human summary
    print("\n" + "=" * 80)
    print(f"AUDIT FILES vs DB — project: {project_id}")
    print("=" * 80 + "\n")
    print(f"decisions:            {summary.total_decisions}")
    print(f"by_action:            {json.dumps(summary.by_action)}")
    print(f"expected_kept:        {summary.expected_kept_count}")
    print(f"kept_found:           {summary.kept_found}")
    print(f"kept_missing:         {summary.kept_missing}")
    print(f"rejects_found_anywhere: {summary.rejects_found_anywhere}")
    print(f"duplicates:           {summary.duplicates}")
    print()

    if args.write_report:
        jp, mp = write_reports(summary, found, problems)
        print(f"Reports written:\n- {jp}\n- {mp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
