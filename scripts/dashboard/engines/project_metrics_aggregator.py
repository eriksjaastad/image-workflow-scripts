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

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from scripts.utils.companion_file_utils import get_file_operation_metrics
from scripts.utils.datetime_utils import normalize_to_naive_utc


def _parse_iso_naive(value: str | None) -> datetime | None:
    """Parse ISO-8601-like strings into a naive UTC datetime.

    - Supports 'Z' by converting to '+00:00' first.
    - If offset-aware, converts to UTC then drops tzinfo.
    - Returns None for falsy/invalid values.
    """
    if not value:
        return None
    try:
        v = value
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        dt = datetime.fromisoformat(v)
        # Convert to UTC, then drop tzinfo (proper timezone handling)
        return normalize_to_naive_utc(dt)
    except Exception:
        return None


@dataclass
class ProjectMetrics:
    project_id: str
    title: str
    status: str | None
    started_at: str | None
    finished_at: str | None
    images_processed: int
    operations_by_type: dict[str, int]
    operations_by_dest: dict[str, dict[str, int]]
    images_per_hour: float
    work_minutes: float  # NEW: Store work time for billed vs actual comparison
    timeseries_daily: list[tuple[str, int]]  # (YYYY-MM-DD, count)
    baseline: dict[str, Any]
    tools: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "projectId": self.project_id,
            "title": self.title,
            "status": self.status,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "totals": {
                "images_processed": self.images_processed,
                "operations_by_type": self.operations_by_type,
                "operations_by_dest": self.operations_by_dest,
                "work_hours": round((self.work_minutes / 60.0) / 0.25)
                * 0.25,  # 15-minute precision (0.25 hour increments)
            },
            "throughput": {
                "images_per_hour": self.images_per_hour,
            },
            "timeseries": {
                "daily_files_processed": self.timeseries_daily,
            },
            "baseline": self.baseline,
            "tools": self.tools,
        }


class ProjectMetricsAggregator:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.projects_dir = self.data_dir / "data" / "projects"
        self.timer_projects_dir = self.data_dir / "data" / "timer_data" / "projects"
        self.file_ops_dir = self.data_dir / "data" / "file_operations_logs"
        self.log_archives_dir = self.data_dir / "data" / "log_archives"
        self.summaries_dir = self.data_dir / "data" / "daily_summaries"

        self._cache_key: tuple[int, int, int, int] | None = None
        self._cache_value: dict[str, dict[str, Any]] = {}

    # ---------------------------- Public API ----------------------------
    def aggregate(self) -> dict[str, dict[str, Any]]:
        """Aggregate per-project metrics with mtime-based caching."""
        cache_key = self._compute_cache_key()
        if self._cache_key == cache_key and self._cache_value:
            return self._cache_value

        manifests = self._load_manifests()

        # Load detailed logs and archived logs
        detailed_ops = list(self._iter_file_operations())
        # Load daily summaries (consolidated) and convert to op-shaped records
        summary_ops = list(self._iter_daily_summaries())

        # De-duplicate: if a day has a summary, drop detailed entries for that day
        summary_days: set[str] = set()
        for rec in summary_ops:
            ts = rec.get("timestamp_str") or rec.get("timestamp") or ""
            day = ""
            try:
                v = ts
                if isinstance(v, str) and v.endswith("Z"):
                    v = v[:-1] + "+00:00"
                dt = datetime.fromisoformat(v)
                day = dt.date().isoformat()
            except Exception:
                pass
            if day:
                summary_days.add(day)

        filtered_detailed: list[dict[str, Any]] = []
        for rec in detailed_ops:
            ts = rec.get("timestamp") or rec.get("timestamp_str") or ""
            day = ""
            try:
                v = ts
                if isinstance(v, str) and v.endswith("Z"):
                    v = v[:-1] + "+00:00"
                dt = datetime.fromisoformat(v)
                day = dt.date().isoformat()
            except Exception:
                pass
            if day and day in summary_days:
                continue
            filtered_detailed.append(rec)

        file_ops = [*summary_ops, *filtered_detailed]

        results: dict[str, dict[str, Any]] = {}
        for mf in manifests:
            project_id = mf.get("projectId") or ""
            if not project_id:
                continue
            title = mf.get("title") or project_id
            status = mf.get("status")
            started_at = mf.get("startedAt")
            finished_at = mf.get("finishedAt")
            root_hint = (mf.get("paths") or {}).get("root") or ""

            # Prefer path-hint when it yields data; otherwise fall back to time-window
            ops_by_path = self._filter_ops_for_project(file_ops, root_hint)
            ops_by_time = self._filter_ops_by_time_window(
                file_ops, started_at, finished_at
            )
            # Combine path-matched detailed ops with summary/time-window ops that lack path hints
            if ops_by_path:
                proj_ops = list(ops_by_path)
                for r in ops_by_time:
                    if not (
                        r.get("source_dir") or r.get("dest_dir") or r.get("working_dir")
                    ):
                        proj_ops.append(r)
            else:
                proj_ops = ops_by_time
            images_processed, ops_by_type, ops_by_dest = self._sum_ops(proj_ops)
            timeseries_daily = self._daily_counts(proj_ops)
            # Compute work time from file operations (same method as dashboard data engine)
            try:
                ops_for_metrics: list[dict[str, Any]] = []
                for op in proj_ops:
                    op_copy = dict(op)
                    ts = op_copy.get("timestamp")
                    if isinstance(ts, datetime):
                        op_copy["timestamp"] = ts.isoformat()
                    elif not isinstance(ts, str):
                        ts_str = op_copy.get("timestamp_str")
                        if isinstance(ts_str, str):
                            op_copy["timestamp"] = ts_str
                    ops_for_metrics.append(op_copy)
                metrics = get_file_operation_metrics(ops_for_metrics)
                work_minutes = float(metrics.get("work_time_minutes") or 0.0)
            except Exception:
                work_minutes = 0.0

            images_per_hour = (
                round(images_processed / (max(work_minutes, 1.0) / 60.0), 2)
                if images_processed > 0
                else 0.0
            )
            tools = self._per_tool_metrics(proj_ops, started_at, finished_at)

            metrics = ProjectMetrics(
                project_id=project_id,
                title=title,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                images_processed=images_processed,
                operations_by_type=ops_by_type,
                operations_by_dest=ops_by_dest,
                images_per_hour=images_per_hour,
                work_minutes=work_minutes,  # NEW: Store work time
                timeseries_daily=timeseries_daily,
                baseline={"overall_iph_baseline": None, "per_tool": {}},
                tools=tools,
            )
            results[project_id] = metrics.to_dict()

        # Compute simple baselines across projects (with light outlier trimming)
        self._apply_baselines(results)

        self._cache_key = cache_key
        self._cache_value = results
        return results

    # ---------------------------- Helpers ----------------------------
    def _compute_cache_key(self) -> tuple[int, int, int, int]:
        def latest_mtime(path: Path) -> int:
            latest = 0
            if path.exists():
                for p in path.rglob("*"):
                    try:
                        latest = max(latest, int(p.stat().st_mtime))
                    except Exception:
                        pass
            return latest

        return (
            latest_mtime(self.projects_dir),
            latest_mtime(self.timer_projects_dir),
            latest_mtime(self.file_ops_dir),
            max(latest_mtime(self.log_archives_dir), latest_mtime(self.summaries_dir)),
        )

    def _load_manifests(self) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        if self.projects_dir.exists():
            for mf in sorted(self.projects_dir.glob("*.project.json")):
                try:
                    with open(mf) as f:
                        pj = json.load(f)
                    # Ensure required surface keys exist
                    pj.setdefault("paths", {})
                    manifests.append(pj)
                except Exception:
                    continue
        # Future hook: timer_data/projects/*.json if needed
        return manifests

    def _iter_file_operations(self):
        if self.file_ops_dir.exists():
            for log in self.file_ops_dir.glob("*.log"):
                yield from self._iter_file_ops_file(log)
        if self.log_archives_dir.exists():
            for gz in self.log_archives_dir.glob("*.gz"):
                yield from self._iter_file_ops_file(gz)

    def _iter_file_ops_file(self, path: Path):
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt") as f:
                    for line in f:
                        rec = self._safe_json(line)
                        if rec and rec.get("type") == "file_operation":
                            yield rec
            else:
                with open(path) as f:
                    for line in f:
                        rec = self._safe_json(line)
                        if rec and rec.get("type") == "file_operation":
                            yield rec
        except Exception:
            return

    def _iter_daily_summaries(self):
        """Yield synthetic file-operation-like records from daily summaries.
        Structure matches minimal fields used by aggregator downstream.
        """
        if not self.summaries_dir.exists():
            return
        for summary_file in self.summaries_dir.glob("daily_summary_*.json"):
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                date_str = summary.get("date")  # YYYYMMDD
                if not date_str or len(date_str) not in (8, 10):
                    continue
                # Normalize to YYYY-MM-DD for timestamp
                if "-" in date_str:
                    # already YYYY-MM-DD
                    iso_day = date_str
                else:
                    iso_day = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
                ts = f"{iso_day}T00:00:00"
                scripts = summary.get("scripts", {}) or {}
                for script_name, script_data in scripts.items():
                    ops = (script_data or {}).get("operations", {}) or {}
                    for op_name, file_count in ops.items():
                        try:
                            cnt = int(file_count or 0)
                        except Exception:
                            cnt = 0
                        if cnt <= 0:
                            continue
                        yield {
                            "timestamp_str": ts,
                            "script": script_name,
                            "session_id": f"daily_{date_str}",
                            "operation": op_name,
                            "file_count": cnt,
                            "source_dir": None,
                            "dest_dir": None,
                            "notes": f"Daily summary for {iso_day}",
                        }
            except Exception:
                continue

    @staticmethod
    def _safe_json(line: str) -> dict[str, Any] | None:
        try:
            return json.loads(line)
        except Exception:
            return None

    @staticmethod
    def _filter_ops_for_project(
        file_ops: list[dict[str, Any]], root_hint: str
    ) -> list[dict[str, Any]]:
        if not root_hint:
            return []
        root_hint = str(root_hint)
        out: list[dict[str, Any]] = []
        for r in file_ops:
            src = r.get("source_dir") or ""
            dst = r.get("dest_dir") or ""
            wd = r.get("working_dir") or ""
            if root_hint in src or root_hint in dst or root_hint in wd:
                out.append(r)
        return out

    @staticmethod
    def _parse_op_timestamp(rec: dict[str, Any]) -> datetime | None:
        ts = rec.get("timestamp") or rec.get("timestamp_str")
        if not ts:
            return None
        try:
            v = ts
            if isinstance(v, str) and v.endswith("Z"):
                v = v[:-1] + "+00:00"
            dt = datetime.fromisoformat(v)
            return normalize_to_naive_utc(dt)
        except Exception:
            return None

    def _filter_ops_by_time_window(
        self,
        file_ops: list[dict[str, Any]],
        started_at: str | None,
        finished_at: str | None,
    ) -> list[dict[str, Any]]:
        start_dt = _parse_iso_naive(started_at)
        end_dt = _parse_iso_naive(finished_at) or datetime.now()
        if not start_dt:
            return []
        out: list[dict[str, Any]] = []
        for r in file_ops:
            dt = self._parse_op_timestamp(r)
            if dt and start_dt <= dt <= end_dt:
                out.append(r)
        return out

    @staticmethod
    def _sum_ops(
        records: list[dict[str, Any]],
    ) -> tuple[int, dict[str, int], dict[str, dict[str, int]]]:
        total = 0
        by_type: dict[str, int] = {}
        by_dest: dict[str, dict[str, int]] = {}
        for r in records:
            op = r.get("operation") or "unknown"
            count = int(r.get("file_count") or 0)
            by_type[op] = by_type.get(op, 0) + count
            # Destination breakdown for move operations
            if op == "move":
                dest = (r.get("dest_dir") or "").strip().lower() or "unknown"
                g = by_dest.setdefault("move", {})
                g[dest] = g.get(dest, 0) + count

        # Count images processed using image-only signals and target destinations
        def _is_png_only(rec: dict[str, Any]) -> bool:
            n = rec.get("notes")
            notes = n if isinstance(n, str) else ""
            notes = notes.lower()
            if "image-only" in notes:
                return True
            files = rec.get("files") or rec.get("files_sample")
            if isinstance(files, list) and files:
                try:
                    return all(str(f).lower().endswith(".png") for f in files)
                except Exception:
                    return False
            return False

        png_records = [rec for rec in records if _is_png_only(rec)]
        if png_records:
            tmp = 0
            for rec in png_records:
                op = str(rec.get("operation") or "").lower()
                dest = str(rec.get("dest_dir") or "").lower()
                if (
                    op == "move"
                    and dest
                    in {
                        "selected",
                        "__selected",
                        "crop",
                        "__crop",
                        "__crop_auto",
                        "crop_auto",
                    }
                    or op == "crop"
                ):
                    tmp += int(rec.get("file_count") or 0)
            total = tmp
        else:
            # Fallback: count crop-only to match test expectations
            total = by_type.get("crop", 0)
        return total, by_type, by_dest

    def _per_tool_metrics(
        self,
        records: list[dict[str, Any]],
        started_at: str | None,
        finished_at: str | None,
    ) -> dict[str, dict[str, Any]]:
        """Compute per-tool totals and naive images/hour using the project window duration."""
        tools: dict[str, dict[str, Any]] = {}
        # Group counts by script, using same crop-priority rule
        grouped: dict[str, dict[str, int]] = {}
        for r in records:
            script = r.get("script") or "unknown"
            op = r.get("operation") or "unknown"
            count = int(r.get("file_count") or 0)
            if script not in grouped:
                grouped[script] = {"_crop": 0, "_all": 0}
            grouped[script]["_all"] += count
            if op == "crop":
                grouped[script]["_crop"] += count

        start_dt = _parse_iso_naive(started_at)
        end_dt = _parse_iso_naive(finished_at) or datetime.now()
        hours = (end_dt - start_dt).total_seconds() / 3600.0 if start_dt else 0.0
        hours = max(hours, 1e-6)
        for script, cnts in grouped.items():
            imgs = cnts["_crop"] if cnts["_crop"] > 0 else cnts["_all"]
            tools[script] = {
                "images_processed": imgs,
                "images_per_hour": round(imgs / hours, 2),
            }
        return tools

    def _apply_baselines(
        self, results: dict[str, dict[str, Any]], last_n: int = 5
    ) -> None:
        # Overall IPH baseline: mean of last N finished projects with non-zero IPH (light trim if many)
        finished = []
        for pj in results.values():
            iph = float(pj.get("throughput", {}).get("images_per_hour") or 0)
            finishedAt = pj.get("finishedAt")
            if iph > 0 and finishedAt:
                finished.append((finishedAt, iph, pj))
        finished.sort(key=lambda t: t[0])
        pool = [iph for _, iph, _ in finished[-last_n:]]
        if len(pool) >= 5:
            pool_sorted = sorted(pool)
            k = max(1, int(0.1 * len(pool_sorted)))
            pool_use = (
                pool_sorted[k:-k] if len(pool_sorted) - 2 * k >= 1 else pool_sorted
            )
        else:
            pool_use = pool
        overall_baseline = round(sum(pool_use) / len(pool_use), 2) if pool_use else 0.0

        # Per-tool baselines: average per tool across last N finished projects
        per_tool_vals: dict[str, list[float]] = {}
        for _, _, pj in finished[-last_n:]:
            for tool, stats in (pj.get("tools") or {}).items():
                v = float(stats.get("images_per_hour") or 0)
                if v > 0:
                    per_tool_vals.setdefault(tool, []).append(v)
        per_tool_baseline = {
            tool: round(sum(vs) / len(vs), 2)
            for tool, vs in per_tool_vals.items()
            if vs
        }

        # Attach to each project
        for pj in results.values():
            base = pj.get("baseline") or {}
            base["overall_iph_baseline"] = overall_baseline
            base["per_tool"] = per_tool_baseline
            pj["baseline"] = base

    @staticmethod
    def _daily_counts(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
        bucket: dict[str, int] = {}
        for r in records:
            ts = r.get("timestamp") or r.get("timestamp_str") or ""
            day = "unknown"
            try:
                v = ts
                if isinstance(v, str) and v.endswith("Z"):
                    v = v[:-1] + "+00:00"
                dt = datetime.fromisoformat(v)
                dt = normalize_to_naive_utc(dt)
                day = dt.date().isoformat()
            except Exception:
                # leave as 'unknown' if parse fails
                pass
            bucket[day] = bucket.get(day, 0) + int(r.get("file_count") or 0)
        # Drop 'unknown' from sparkline unless it is the only bucket
        if "unknown" in bucket and len(bucket) > 1:
            bucket.pop("unknown", None)
        return sorted(bucket.items(), key=lambda kv: kv[0])

    @staticmethod
    def _compute_images_per_hour(
        started_at: str | None, finished_at: str | None, images_processed: int
    ) -> float:
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


if __name__ == "__main__":
    main()
