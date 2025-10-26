#!/usr/bin/env python3
"""
Audit Orphan Decision Files
---------------------------

Scans a directory (default: mojo3/faces) for .decision files and checks whether
corresponding PNGs (same stem) exist in the same directory or elsewhere in the
repository. Writes a JSON and CSV report under data/daily_summaries/.

Usage:
  python -m scripts.tools.audit_orphan_decisions --dir mojo3/faces

This tool is READ-ONLY and does not move or modify any files.
"""

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict


@dataclass
class DecisionRecord:
    decision_path: str
    expected_png: str
    png_in_same_dir: bool
    found_png_paths: List[str]
    status: str  # present_same_dir | moved_elsewhere | missing


def find_matching_pngs(repo_root: Path, stem: str) -> List[Path]:
    # Search repo for any .png with the same stem
    return list(repo_root.rglob(f"**/{stem}.png"))


def classify_record(decision_path: Path, found_pngs: List[Path]) -> DecisionRecord:
    stem = decision_path.stem
    expected_png = decision_path.with_suffix('.png')
    png_in_same_dir = expected_png.exists()

    if png_in_same_dir:
        status = 'present_same_dir'
    elif found_pngs:
        status = 'moved_elsewhere'
    else:
        status = 'missing'

    return DecisionRecord(
        decision_path=str(decision_path.resolve()),
        expected_png=expected_png.name,
        png_in_same_dir=png_in_same_dir,
        found_png_paths=[str(p.resolve()) for p in found_pngs],
        status=status,
    )


def write_reports(repo_root: Path, records: List[DecisionRecord]) -> Dict[str, str]:
    out_dir = repo_root / 'data' / 'daily_summaries'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base = f"orphan_decisions_{ts}"

    # JSON report
    json_path = out_dir / f"{base}.json"
    with open(json_path, 'w') as jf:
        json.dump({
            'generated_at': ts,
            'total': len(records),
            'summary': {
                'present_same_dir': sum(1 for r in records if r.status == 'present_same_dir'),
                'moved_elsewhere': sum(1 for r in records if r.status == 'moved_elsewhere'),
                'missing': sum(1 for r in records if r.status == 'missing'),
            },
            'records': [asdict(r) for r in records],
        }, jf, indent=2)

    # CSV report
    csv_path = out_dir / f"{base}.csv"
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=[
            'decision_path', 'expected_png', 'png_in_same_dir', 'status', 'found_png_paths'
        ])
        writer.writeheader()
        for r in records:
            writer.writerow({
                'decision_path': r.decision_path,
                'expected_png': r.expected_png,
                'png_in_same_dir': r.png_in_same_dir,
                'status': r.status,
                'found_png_paths': json.dumps(r.found_png_paths),
            })

    return {'json': str(json_path), 'csv': str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit orphan .decision files and locate matching PNGs')
    parser.add_argument('--dir', dest='target_dir', default='mojo3/faces', help='Directory to scan for .decision files')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    target_dir = (repo_root / args.target_dir).resolve()

    if not target_dir.exists():
        raise SystemExit(f"Target directory not found: {target_dir}")

    decision_files = list(target_dir.glob('*.decision'))
    records: List[DecisionRecord] = []

    for d in sorted(decision_files):
        stem = d.stem
        matches = find_matching_pngs(repo_root, stem)
        records.append(classify_record(d, matches))

    outputs = write_reports(repo_root, records)

    # Minimal console summary
    present_same_dir = sum(1 for r in records if r.status == 'present_same_dir')
    moved_elsewhere = sum(1 for r in records if r.status == 'moved_elsewhere')
    missing = sum(1 for r in records if r.status == 'missing')

    print(f"Scanned {len(decision_files)} .decision files in {target_dir}")
    print(f"  present_same_dir: {present_same_dir}")
    print(f"  moved_elsewhere:  {moved_elsewhere}")
    print(f"  missing:          {missing}")
    print(f"Reports written:\n  JSON: {outputs['json']}\n  CSV:  {outputs['csv']}")


if __name__ == '__main__':
    main()
