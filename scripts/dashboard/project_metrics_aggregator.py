#!/usr/bin/env python3
"""
Project Metrics Aggregator
==========================
Aggregates per-project throughput metrics for the dashboard from local data sources.

Inputs (all optional, local only):
- data/projects/*.project.json
- data/timer_data/projects/*.json (future-proofed; not required)
- data/file_operations_logs/*.log and data/log_archives/*.gz

Outputs (dict keyed by projectId):
- projectId, title, status, startedAt, finishedAt
- totals: images_processed, operations_by_type
- throughput: images_per_hour (end-to-end, based on started/finished or started/now)
- timeseries: daily files processed for sparkline rendering
- baseline: placeholder structure for future comparisons

Caching:
- Lightweight mtime-based cache: recompute only if any relevant file mtime increases.

Notes:
- Timestamps are treated uniformly as naive ISO strings. If timezone info is present (e.g., Z or +02:00),
  it is normalized by removing tzinfo to avoid timezone math. This avoids DST complexity for now and is
  consistent with the dashboard assumption of local, naive ISO handling.
"""

from __future__ import annotations

import json
import gzip
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_iso_naive(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601-like strings into a naive datetime.

    - Supports 'Z' by converting to '+00:00' first.
    - If offset-aware, drops tzinfo to make it naive.
    - Returns None for falsy/invalid values.
    """
    if not value:
        return None
    try:
        v = value
        if v.endswith('Z'):
            v = v[:-1] + '+00:00'
        dt = datetime.fromisoformat(v)
        # Drop tzinfo if present (treat uniformly as naive)
        return dt.replace(tzinfo=None)
    except Exception:
        return None


@dataclass
class ProjectMetrics:
    project_id: str
    title: str
    status: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    images_processed: int
    operations_by_type: Dict[str, int]
    images_per_hour: float
    timeseries_daily: List[Tuple[str, int]]  # (YYYY-MM-DD, count)
    baseline: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'projectId': self.project_id,
            'title': self.title,
            'status': self.status,
            'startedAt': self.started_at,
            'finishedAt': self.finished_at,
            'totals': {
                'images_processed': self.images_processed,
                'operations_by_type': self.operations_by_type,
            },
            'throughput': {
                'images_per_hour': self.images_per_hour,
            },
            'timeseries': {
                'daily_files_processed': self.timeseries_daily,
            },
            'baseline': self.baseline,
        }


class ProjectMetricsAggregator:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.projects_dir = self.data_dir / 'data' / 'projects'
        self.timer_projects_dir = self.data_dir / 'data' / 'timer_data' / 'projects'
        self.file_ops_dir = self.data_dir / 'data' / 'file_operations_logs'
        self.log_archives_dir = self.data_dir / 'data' / 'log_archives'

        self._cache_key: Optional[Tuple[int, int, int, int]] = None
        self._cache_value: Dict[str, Dict[str, Any]] = {}

    # ---------------------------- Public API ----------------------------
    def aggregate(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate per-project metrics with mtime-based caching."""
        cache_key = self._compute_cache_key()
        if self._cache_key == cache_key and self._cache_value:
            return self._cache_value

        manifests = self._load_manifests()
        file_ops = list(self._iter_file_operations())

        results: Dict[str, Dict[str, Any]] = {}
        for mf in manifests:
            project_id = mf.get('projectId') or ''
            if not project_id:
                continue
            title = mf.get('title') or project_id
            status = mf.get('status')
            started_at = mf.get('startedAt')
            finished_at = mf.get('finishedAt')
            root_hint = (mf.get('paths') or {}).get('root') or ''

            proj_ops = self._filter_ops_for_project(file_ops, root_hint)
            images_processed, ops_by_type = self._sum_ops(proj_ops)
            timeseries_daily = self._daily_counts(proj_ops)
            images_per_hour = self._compute_images_per_hour(started_at, finished_at, images_processed)

            metrics = ProjectMetrics(
                project_id=project_id,
                title=title,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                images_processed=images_processed,
                operations_by_type=ops_by_type,
                images_per_hour=images_per_hour,
                timeseries_daily=timeseries_daily,
                baseline={'images_per_hour_baseline': None},
            )
            results[project_id] = metrics.to_dict()

        self._cache_key = cache_key
        self._cache_value = results
        return results

    # ---------------------------- Helpers ----------------------------
    def _compute_cache_key(self) -> Tuple[int, int, int, int]:
        def latest_mtime(path: Path) -> int:
            latest = 0
            if path.exists():
                for p in path.rglob('*'):
                    try:
                        latest = max(latest, int(p.stat().st_mtime))
                    except Exception:
                        pass
            return latest

        return (
            latest_mtime(self.projects_dir),
            latest_mtime(self.timer_projects_dir),
            latest_mtime(self.file_ops_dir),
            latest_mtime(self.log_archives_dir),
        )

    def _load_manifests(self) -> List[Dict[str, Any]]:
        manifests: List[Dict[str, Any]] = []
        if self.projects_dir.exists():
            for mf in sorted(self.projects_dir.glob('*.project.json')):
                try:
                    with open(mf, 'r') as f:
                        pj = json.load(f)
                    # Ensure required surface keys exist
                    pj.setdefault('paths', {})
                    manifests.append(pj)
                except Exception:
                    continue
        # Future hook: timer_data/projects/*.json if needed
        return manifests

    def _iter_file_operations(self):
        if self.file_ops_dir.exists():
            for log in self.file_ops_dir.glob('*.log'):
                yield from self._iter_file_ops_file(log)
        if self.log_archives_dir.exists():
            for gz in self.log_archives_dir.glob('*.gz'):
                yield from self._iter_file_ops_file(gz)

    def _iter_file_ops_file(self, path: Path):
        try:
            if path.suffix == '.gz':
                with gzip.open(path, 'rt') as f:
                    for line in f:
                        rec = self._safe_json(line)
                        if rec and rec.get('type') == 'file_operation':
                            yield rec
            else:
                with open(path, 'r') as f:
                    for line in f:
                        rec = self._safe_json(line)
                        if rec and rec.get('type') == 'file_operation':
                            yield rec
        except Exception:
            return

    @staticmethod
    def _safe_json(line: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(line)
        except Exception:
            return None

    @staticmethod
    def _filter_ops_for_project(file_ops: List[Dict[str, Any]], root_hint: str) -> List[Dict[str, Any]]:
        if not root_hint:
            return []
        root_hint = str(root_hint)
        out: List[Dict[str, Any]] = []
        for r in file_ops:
            src = r.get('source_dir') or ''
            dst = r.get('dest_dir') or ''
            wd = r.get('working_dir') or ''
            if root_hint in src or root_hint in dst or root_hint in wd:
                out.append(r)
        return out

    @staticmethod
    def _sum_ops(records: List[Dict[str, Any]]) -> Tuple[int, Dict[str, int]]:
        total = 0
        by_type: Dict[str, int] = {}
        for r in records:
            op = r.get('operation') or 'unknown'
            count = int(r.get('file_count') or 0)
            by_type[op] = by_type.get(op, 0) + count
            # Define "images processed" as crop operations primarily; fallback to all ops if crop absent
            total += count if op == 'crop' else 0
        if total == 0:
            total = sum(by_type.values())
        return total, by_type

    @staticmethod
    def _daily_counts(records: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        bucket: Dict[str, int] = {}
        for r in records:
            ts = r.get('timestamp') or r.get('timestamp_str') or ''
            day = 'unknown'
            try:
                v = ts
                if isinstance(v, str) and v.endswith('Z'):
                    v = v[:-1] + '+00:00'
                dt = datetime.fromisoformat(v)
                dt = dt.replace(tzinfo=None)
                day = dt.date().isoformat()
            except Exception:
                # leave as 'unknown' if parse fails
                pass
            bucket[day] = bucket.get(day, 0) + int(r.get('file_count') or 0)
        # Drop 'unknown' from sparkline unless it is the only bucket
        if 'unknown' in bucket and len(bucket) > 1:
            bucket.pop('unknown', None)
        return sorted(bucket.items(), key=lambda kv: kv[0])

    @staticmethod
    def _compute_images_per_hour(started_at: Optional[str], finished_at: Optional[str], images_processed: int) -> float:
        start_dt = _parse_iso_naive(started_at)
        end_dt = _parse_iso_naive(finished_at) or datetime.now()
        if not start_dt:
            return 0.0
        seconds = max((end_dt - start_dt).total_seconds(), 1.0)
        hours = seconds / 3600.0
        return round(images_processed / hours, 2)


def main():
    agg = ProjectMetricsAggregator(Path(__file__).resolve().parents[2])
    data = agg.aggregate()
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()


