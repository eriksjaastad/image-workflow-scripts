#!/usr/bin/env python3
"""
Current Project Dashboard (Flask)
=================================
Real-time dashboard for the ACTIVE project with predictive progress tracking.

- Serves on port 8082 by default
- Auto-refreshes via /api/progress (JSON) every 30 seconds
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import threading

from flask import Flask, jsonify, render_template, send_from_directory

# Ensure project root is importable, then use absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing data engines and aggregators (absolute paths)
from scripts.dashboard.project_metrics_aggregator import ProjectMetricsAggregator
from scripts.dashboard.data_engine import DashboardDataEngine
from scripts.utils.companion_file_utils import launch_browser


DATA_DIR = PROJECT_ROOT / "data"
PROJECTS_DIR = DATA_DIR / "projects"
CACHE_DIR = DATA_DIR / "dashboard_cache"

# Lightweight in-process caches
INVENTORY_CACHE: Dict[str, Dict[str, Any]] = {}


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        v = ts
        if isinstance(v, str) and v.endswith("Z"):
            v = v[:-1] + "+00:00"
        dt = datetime.fromisoformat(v)
        return dt
    except Exception:
        return None


def _count_pngs(path: Optional[str | Path]) -> int:
    try:
        if not path:
            return 0
        p = Path(path).expanduser()
        if not p.exists():
            return 0
        cnt = 0
        for f in p.rglob('*.png'):
            try:
                if f.is_file():
                    cnt += 1
            except Exception:
                continue
        return cnt
    except Exception:
        return 0


def _latest_mtime(path: Optional[str | Path]) -> int:
    try:
        if not path:
            return 0
        p = Path(path).expanduser()
        if not p.exists():
            return 0
        latest = 0
        for f in p.rglob('*'):
            try:
                latest = max(latest, int(f.stat().st_mtime))
            except Exception:
                continue
        return latest
    except Exception:
        return 0


def find_active_project() -> Optional[Dict[str, Any]]:
    """Find the active project with robust heuristics.

    Priority:
    1) status == 'active'
    2) finishedAt is missing/empty/None
    3) fallback to most recent startedAt
    """
    if not PROJECTS_DIR.exists():
        return None
    candidates: List[Tuple[datetime, Dict[str, Any], str]] = []
    for mf in sorted(PROJECTS_DIR.glob("*.project.json")):
        try:
            pj = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = (pj.get("status") or "").lower().strip()
        finished_at = pj.get("finishedAt")
        started_at = pj.get("startedAt")
        # Parse start for ordering; fall back to file mtime
        try:
            sdt = _parse_iso(started_at) or datetime.fromtimestamp(mf.stat().st_mtime)
        except Exception:
            sdt = datetime.fromtimestamp(mf.stat().st_mtime)

        is_active_status = status == "active"
        no_finish = finished_at in (None, "", [])
        if is_active_status or no_finish:
            pj.setdefault("counts", {})
            pj["manifestPath"] = str(mf)
            candidates.append((sdt, pj, str(mf)))

    if not candidates:
        return None
    # Choose most recent start time
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def compute_current_session_rate(file_ops_records: List[Dict[str, Any]], hours: float = 2.0) -> Dict[str, Any]:
    if not file_ops_records:
        return {"overall": 0.0, "by_op": {}, "by_stage": {"reviewed": 0.0, "selection": 0.0, "crop": 0.0, "sort": 0.0}}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    total = 0
    by_op: Dict[str, int] = {}
    stage_counts: Dict[str, int] = {"reviewed": 0, "selection": 0, "crop": 0, "sort": 0}
    seen: set[tuple] = set()

    for r in file_ops_records:
        ts = r.get("timestamp") or r.get("timestamp_str")
        if not ts:
            continue
        try:
            v = ts if isinstance(ts, str) else ""
            if v.endswith("Z"):
                v = v[:-1] + "+00:00"
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if dt < cutoff:
            continue
        count = int(r.get("file_count") or 0)
        op = str(r.get("operation") or "unknown").lower()
        # De-dup identical entries (common across rotated logs)
        files_sig = tuple((r.get("files") or r.get("files_sample") or [])[:5])
        sig = (dt.isoformat(), op, str(r.get("source_dir") or ''), str(r.get("dest_dir") or ''), count, files_sig)
        if sig in seen:
            continue
        seen.add(sig)
        total += count
        by_op[op] = by_op.get(op, 0) + count
        # Classify stage for stage-specific rates
        dest = str(r.get("dest_dir") or "").lower().strip()
        if op == "crop":
            stage_counts["crop"] += count
        elif op == "move":
            if dest == "selected":
                stage_counts["selection"] += count
                stage_counts["reviewed"] += count
            elif dest in {"__crop", "__crop_auto", "crop", "crop_auto"}:
                stage_counts["reviewed"] += count
            elif dest.startswith("character_group") or dest.startswith("_"):
                stage_counts["sort"] += count

    rate = total / max(hours, 1e-6)
    by_stage_rate = {k: round(v / max(hours, 1e-6), 2) for k, v in stage_counts.items()}
    return {
        "overall": round(rate, 2),
        "by_op": {k: round(v / max(hours, 1e-6), 2) for k, v in by_op.items()},
        "by_stage": by_stage_rate,
    }


def build_time_series_for_chart(project_metrics: Dict[str, Any], project_start: Optional[str], target_total: int) -> Dict[str, Any]:
    # Actual progress series from aggregator daily buckets
    actual_pts: List[Tuple[str, int]] = project_metrics.get("timeseries", {}).get("daily_files_processed", [])
    actual_cum: List[Tuple[str, int]] = []
    running = 0
    for day_str, cnt in actual_pts:
        running += int(cnt or 0)
        actual_cum.append((day_str, running))

    # Baseline predicted linear path using baseline overall IPH
    started_dt = _parse_iso(project_start) or datetime.now(timezone.utc)
    baseline_iph = float(project_metrics.get("baseline", {}).get("overall_iph_baseline") or 0.0)
    predicted: List[Tuple[str, int]] = []
    if baseline_iph > 0:
        # Build daily points for the next 30 days or until target
        max_days = 60
        imgs_per_day = baseline_iph * 24.0
        acc = 0.0
        for i in range(max_days):
            day = (started_dt + timedelta(days=i)).date().isoformat()
            acc = min(target_total, acc + imgs_per_day)
            predicted.append((day, int(round(acc))))

    return {"actual": actual_cum, "predicted": predicted}


def compute_milestones(total_images: int, processed_images: int, baseline_iph: float, started_at: Optional[str]) -> List[Dict[str, Any]]:
    milestones = []
    fractions = [0.25, 0.5, 0.75, 1.0]
    start_dt = _parse_iso(started_at) or datetime.now(timezone.utc)
    imgs_per_day = baseline_iph * 24.0 if baseline_iph > 0 else 0.0
    for frac in fractions:
        target_images = int(round(total_images * frac))
        target_date = None
        if imgs_per_day > 0:
            days_needed = target_images / imgs_per_day
            target_date = (start_dt + timedelta(days=days_needed)).date().isoformat()
        remaining_to_hit = max(0, target_images - processed_images)
        milestones.append({
            "label": f"{int(frac*100)}%",
            "target_images": target_images,
            "target_date": target_date,
            "remaining": remaining_to_hit,
        })
    return milestones


# -------------------------- Pattern Analysis --------------------------

def _load_manifest_for(project_id: str) -> Optional[Dict[str, Any]]:
    if not project_id:
        return None
    path = PROJECTS_DIR / f"{project_id}.project.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _project_records(engine: DashboardDataEngine, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    start_date = None
    started_at = manifest.get("startedAt")
    if started_at:
        try:
            sdt = _parse_iso(started_at)
            if sdt:
                start_date = sdt.date().isoformat()
        except Exception:
            start_date = None
    detailed_records = engine._load_from_detailed_logs(start_date=start_date, end_date=None)  # type: ignore[attr-defined]
    root_hint = (manifest.get("paths") or {}).get("root") or ""

    def rec_in_project(rec: Dict[str, Any]) -> bool:
        if root_hint:
            for key in ("source_dir", "dest_dir", "working_dir"):
                v = str(rec.get(key) or "")
                if root_hint in v:
                    return True
        ts = rec.get("timestamp") or rec.get("timestamp_str")
        if not ts:
            return False
        try:
            v = ts
            if isinstance(v, str) and v.endswith("Z"):
                v = v[:-1] + "+00:00"
            dt = datetime.fromisoformat(v)
            sdt = _parse_iso(manifest.get("startedAt")) or now - timedelta(days=365)
            edt = _parse_iso(manifest.get("finishedAt")) or now
            return sdt.replace(tzinfo=None) <= dt.replace(tzinfo=None) <= edt.replace(tzinfo=None)
        except Exception:
            return False

    return [r for r in detailed_records if rec_in_project(r)]


def _analyze_project_pattern(project_id: str, pj_metrics: Dict[str, Any], engine: DashboardDataEngine) -> Optional[Dict[str, Any]]:
    manifest = _load_manifest_for(project_id)
    if not manifest:
        return None

    counts = (manifest.get("counts") or {})
    total_images = int(counts.get("finalImages") or counts.get("initialImages") or 0)
    if total_images <= 0:
        return None

    # Daily progress percentage (use aggregator timeseries daily files processed)
    daily = pj_metrics.get("timeseries", {}).get("daily_files_processed") or []
    cum = []
    running = 0
    for day, cnt in daily:
        running += int(cnt or 0)
        cum.append((day, running))
    pct_by_day = [(d, (v / total_images) if total_images > 0 else 0.0) for d, v in cum]
    # Percent complete by day 3 and additional by days 4-6
    d1_3 = pct_by_day[min(2, len(pct_by_day)-1)][1] if pct_by_day else 0.0
    d4_6 = (pct_by_day[min(5, len(pct_by_day)-1)][1] - d1_3) if len(pct_by_day) >= 2 else 0.0

    # Time-of-day productivity (by hour)
    records = _project_records(engine, manifest)
    hourly_counts: Dict[int, int] = {h: 0 for h in range(24)}
    for r in records:
        ts = r.get("timestamp") or r.get("timestamp_str")
        try:
            v = ts
            if isinstance(v, str) and v.endswith("Z"):
                v = v[:-1] + "+00:00"
            dt = datetime.fromisoformat(v)
            hour = dt.hour
        except Exception:
            continue
        hourly_counts[hour] += int(r.get("file_count") or 0)

    def _avg_for_range(start: int, end: int) -> float:
        rng = list(range(start, end))
        total = sum(hourly_counts[h] for h in rng)
        hours = max(1, len(rng))
        return total / hours

    windows = {
        "morning": (6, 12),
        "afternoon": (12, 17),
        "evening": (17, 22),
        "late": (22, 24),
        "early": (0, 6),
    }
    window_rates = {k: _avg_for_range(a, b) for k, (a, b) in windows.items()}
    peak_window = max(window_rates.items(), key=lambda kv: kv[1])[0] if window_rates else None

    # Transition gap: selections completion day => first crop day after
    by_day_sel: Dict[str, int] = {}
    by_day_crop: Dict[str, int] = {}
    for r in records:
        op = str(r.get("operation") or "").lower()
        ts = r.get("timestamp") or r.get("timestamp_str")
        try:
            v = ts
            if isinstance(v, str) and v.endswith("Z"):
                v = v[:-1] + "+00:00"
            day = datetime.fromisoformat(v).date().isoformat()
        except Exception:
            continue
        cnt = int(r.get("file_count") or 0)
        if op == "crop":
            by_day_crop[day] = by_day_crop.get(day, 0) + cnt
        elif op == "move" and str(r.get("dest_dir") or "").lower().strip() == "selected":
            by_day_sel[day] = by_day_sel.get(day, 0) + cnt
    sel_cum = 0
    sel_done_day = None
    for day in sorted(by_day_sel.keys()):
        sel_cum += by_day_sel[day]
        if sel_cum >= total_images:
            sel_done_day = day
            break
    crop_start_day = None
    if sel_done_day:
        for day in sorted(by_day_crop.keys()):
            if day >= sel_done_day and by_day_crop[day] > 0:
                crop_start_day = day
                break
    gap_days = None
    if sel_done_day and crop_start_day:
        try:
            d1 = datetime.fromisoformat(sel_done_day)
            d2 = datetime.fromisoformat(crop_start_day)
            gap_days = (d2 - d1).days
        except Exception:
            gap_days = None

    avg_rate = float(pj_metrics.get("throughput", {}).get("images_per_hour") or 0.0)

    return {
        "projectId": project_id,
        "early_progress_pct": round(d1_3 * 100.0, 2),
        "late_progress_pct": round(d4_6 * 100.0, 2),
        "peak_window": peak_window,
        "hourly_counts": hourly_counts,
        "transition_gap_days": gap_days,
        "avg_rate_iph": avg_rate,
        "total_images": total_images,
    }


def _compute_patterns(projects_map: Dict[str, Dict[str, Any]], engine: DashboardDataEngine) -> Dict[str, Any]:
    per_project = []
    for project_id, pj in projects_map.items():
        if pj.get("finishedAt"):
            analysis = _analyze_project_pattern(project_id, pj, engine)
            if analysis:
                per_project.append(analysis)

    summary: Dict[str, Any] = {"projects_analyzed": len(per_project), "messages": []}
    if len(per_project) < 2:
        summary["messages"].append("Baseline still building – need 2+ completed projects for robust patterns.")
    if not per_project:
        return {"summary": summary, "patterns": {}}

    early = [p["early_progress_pct"] for p in per_project if p.get("early_progress_pct") is not None]
    late = [p["late_progress_pct"] for p in per_project if p.get("late_progress_pct") is not None]
    avg_early = sum(early) / len(early) if early else 0.0
    avg_late = sum(late) / len(late) if late else 0.0

    hourly_totals: Dict[int, float] = {h: 0.0 for h in range(24)}
    for p in per_project:
        hc = p.get("hourly_counts") or {}
        for h in range(24):
            hourly_totals[h] += float(hc.get(h, 0))
    for h in range(24):
        hourly_totals[h] = hourly_totals[h] / max(1, len(per_project))

    def _sum_range(start: int, end: int) -> float:
        return sum(hourly_totals[h] for h in range(start, end))
    window_avgs = {
        "morning": _sum_range(6, 12) / 6.0,
        "afternoon": _sum_range(12, 17) / 5.0,
        "evening": _sum_range(17, 22) / 5.0,
        "late": _sum_range(22, 24) / 2.0,
        "early": _sum_range(0, 6) / 6.0,
    }
    peak_window = max(window_avgs.items(), key=lambda kv: kv[1])[0] if window_avgs else None

    gaps = [p["transition_gap_days"] for p in per_project if p.get("transition_gap_days") is not None]
    avg_gap = (sum(gaps) / len(gaps)) if gaps else None

    trend = [{"projectId": p.get("projectId"), "imagesPerHour": p.get("avg_rate_iph", 0.0)} for p in per_project]

    return {
        "summary": summary,
        "patterns": {
            "slow_start": {"avg_early_pct": round(avg_early, 2), "avg_late_pct": round(avg_late, 2)},
            "time_of_day": {"hourly_avg": hourly_totals, "peak_window": peak_window, "windows": window_avgs},
            "transition_gap": {"avg_gap_days": avg_gap},
            "trend": trend,
        },
    }

def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.config['JSON_SORT_KEYS'] = False

    # Engines
    agg = ProjectMetricsAggregator(PROJECT_ROOT)
    engine = DashboardDataEngine(str(PROJECT_ROOT))

    @app.route("/")
    def index():
        return render_template("current_project.html")

    @app.route("/api/progress")
    def progress_api():
        # Active project manifest
        active = find_active_project()
        if not active:
            return jsonify({"error": "No active project found"}), 200

        project_id = active.get("projectId")
        title = active.get("title") or project_id
        started_at = active.get("startedAt")
        finished_at = active.get("finishedAt")
        counts = active.get("counts") or {}
        total_images = int(counts.get("initialImages") or 0)

        # Aggregate metrics across projects; then select this project
        projects = agg.aggregate()
        proj_metrics = projects.get(project_id) or {}
        totals = proj_metrics.get("totals", {}) or {}
        processed_images_raw = int(totals.get("images_processed") or 0)
        all_time_work_hours = float(totals.get("work_hours") or 0.0)
        ops_by_type: Dict[str, int] = totals.get("operations_by_type", {}) or {}
        ops_by_dest_all: Dict[str, Dict[str, int]] = totals.get("operations_by_dest", {}) or {}
        move_dest_counts: Dict[str, int] = ops_by_dest_all.get("move", {}) or {}

        # Stage completions (png-only counts already enforced upstream in aggregator):
        selection_done = int(move_dest_counts.get("selected", 0))
        reviewed_done = selection_done + int(move_dest_counts.get("__crop_auto", 0)) + int(move_dest_counts.get("__crop", 0)) + int(move_dest_counts.get("crop_auto", 0)) + int(move_dest_counts.get("crop", 0))
        crop_done = int(ops_by_type.get("crop", 0))
        def is_sort_bucket(name: str) -> bool:
            n = name.lower().strip()
            if n.startswith("character_group"):
                return True
            if n.startswith("_"):
                # Exclude known system dirs
                return n not in {"_trash", "_tmp", "_temp"}
            return False
        sort_done = sum(v for k, v in move_dest_counts.items() if is_sort_bucket(str(k)))
        # End-to-end completed is the bottleneck across stages
        processed_images = min(selection_done or 0, crop_done or 0, sort_done or 0)

        # Historical baseline (overall across completed projects)
        baseline_iph = float(proj_metrics.get("baseline", {}).get("overall_iph_baseline") or 0.0)

        # Stage baselines from finished projects (images/hour)
        def compute_stage_baselines(projects_map: Dict[str, Dict[str, Any]], cur_total: int) -> Dict[str, float]:
            sel_rates: List[float] = []
            rev_rates: List[float] = []
            crop_rates: List[float] = []
            sort_rates: List[float] = []
            sel_w: List[float] = []
            rev_w: List[float] = []
            crop_w: List[float] = []
            sort_w: List[float] = []
            for pid, pj in projects_map.items():
                if not pj.get("finishedAt"):
                    continue
                totals = pj.get("totals", {}) or {}
                work_hours = float(totals.get("work_hours") or 0.0)
                if work_hours <= 0:
                    continue
                # Weight by similarity of size (initial images)
                mf = _load_manifest_for(pid)
                size = int(((mf or {}).get('counts') or {}).get('initialImages') or 0)
                if cur_total > 0 and size > 0:
                    ratio = max(cur_total, size) / max(1, min(cur_total, size))
                    weight = 1.0 / ratio  # identical size = 1.0; 2x size => 0.5
                else:
                    weight = 1.0
                by_type = totals.get("operations_by_type", {}) or {}
                by_dest_all = totals.get("operations_by_dest", {}) or {}
                move_dest = by_dest_all.get("move", {}) or {}
                sel = int(move_dest.get("selected", 0))
                rev = sel + int(move_dest.get("__crop", 0)) + int(move_dest.get("__crop_auto", 0)) + int(move_dest.get("crop", 0)) + int(move_dest.get("crop_auto", 0))
                cr = int(by_type.get("crop", 0))
                so = sum(v for k, v in move_dest.items() if str(k).startswith("character_group"))
                sel_rates.append((sel / work_hours) * weight); sel_w.append(weight)
                rev_rates.append((rev / work_hours) * weight); rev_w.append(weight)
                crop_rates.append((cr / work_hours) * weight); crop_w.append(weight)
                sort_rates.append((so / work_hours) * weight); sort_w.append(weight)
            def wavg(vals: List[float], weights: List[float]) -> float:
                s = sum(vals)
                w = sum(weights) if weights else 0.0
                return round(s / w, 2) if w > 0 else 0.0
            return {
                "reviewed": wavg(rev_rates, rev_w),
                "selection": wavg(sel_rates, sel_w),
                "crop": wavg(crop_rates, crop_w),
                "sort": wavg(sort_rates, sort_w),
            }

        stage_baselines = compute_stage_baselines(projects, total_images)

        # Recent rate using detailed records via data engine
        # Pull recent records for last 2 hours from detailed logs and summaries
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=2)).date().isoformat()
        detailed_records = engine._load_from_detailed_logs(start_date=start_date, end_date=None)  # type: ignore[attr-defined]
        # Filter to project by path/time window similar to aggregator
        root_hint = (active.get("paths") or {}).get("root") or ""
        # Basic filter: by root hint or within project time window
        def rec_in_project(rec: Dict[str, Any]) -> bool:
            if root_hint:
                for key in ("source_dir", "dest_dir", "working_dir"):
                    v = str(rec.get(key) or "")
                    if root_hint in v:
                        return True
            # Fallback: by project start/finish window
            ts = rec.get("timestamp") or rec.get("timestamp_str")
            if not ts:
                return False
            try:
                v = ts
                if isinstance(v, str) and v.endswith("Z"):
                    v = v[:-1] + "+00:00"
                dt = datetime.fromisoformat(v)
                sdt = _parse_iso(started_at) or now - timedelta(days=30)
                edt = _parse_iso(finished_at) or now
                return sdt.replace(tzinfo=None) <= dt.replace(tzinfo=None) <= edt.replace(tzinfo=None)
            except Exception:
                return False

        proj_records = [r for r in detailed_records if rec_in_project(r)]
        recent = compute_current_session_rate(proj_records, hours=2.0)
        current_rate_overall = float(recent.get("overall") or 0.0)
        stage_rates = recent.get("by_stage", {}) or {}
        rev_rate = float(stage_rates.get("reviewed") or 0.0)
        sel_rate = float(stage_rates.get("selection") or 0.0)
        crop_rate = float(stage_rates.get("crop") or 0.0)
        sort_rate = float(stage_rates.get("sort") or 0.0)

        # Determine last active stage (by most recent event timestamp)
        def latest_ts_for(records: List[Dict[str, Any]], classifier: str) -> Optional[datetime]:
            latest: Optional[datetime] = None
            for r in records:
                ts = r.get("timestamp") or r.get("timestamp_str")
                if not ts:
                    continue
                try:
                    v = ts
                    if isinstance(v, str) and v.endswith("Z"):
                        v = v[:-1] + "+00:00"
                    dt = datetime.fromisoformat(v)
                except Exception:
                    continue
                op = str(r.get("operation") or "").lower()
                dest = str(r.get("dest_dir") or "").lower().strip()
                is_match = False
                if classifier == "reviewed":
                    is_match = (op == "move" and (dest in {"__selected", "selected"} or dest in {"__crop", "__crop_auto", "crop", "crop_auto"}))
                elif classifier == "crop":
                    is_match = (op == "crop")
                elif classifier == "sort":
                    is_match = (op == "move" and dest.startswith("character_group"))
                if is_match:
                    latest = dt if (latest is None or dt > latest) else latest
            return latest

        latest_rev = latest_ts_for(proj_records, "reviewed")
        latest_crop = latest_ts_for(proj_records, "crop")
        latest_sort = latest_ts_for(proj_records, "sort")
        recommended_topbar = "reviewed"
        latest_map = {"reviewed": latest_rev, "crop": latest_crop, "sort": latest_sort}
        try:
            recommended_topbar = max((k for k in latest_map.keys()), key=lambda k: latest_map[k] or datetime.fromtimestamp(0))
        except Exception:
            recommended_topbar = "reviewed"

        # Predictive metrics
        remaining = max(0, total_images - processed_images)
        # Stage remaining
        sel_rem = max(0, total_images - selection_done)
        rev_rem = max(0, total_images - reviewed_done)
        # Total to crop is everything sent to crop or crop_auto (support legacy and new names)
        total_to_crop = (
            int(move_dest_counts.get("__crop", 0)) + int(move_dest_counts.get("__crop_auto", 0)) +
            int(move_dest_counts.get("crop", 0)) + int(move_dest_counts.get("crop_auto", 0))
        )
        crop_rem = max(0, total_to_crop - crop_done)
        sort_rem = max(0, total_images - sort_done)
        # Bottleneck projection: max of per-stage time remaining
        per_stage_hours = []
        if rev_rate > 0:
            per_stage_hours.append(rev_rem / rev_rate)
        if sel_rate > 0:
            per_stage_hours.append(sel_rem / sel_rate)
        if crop_rate > 0:
            per_stage_hours.append(crop_rem / crop_rate)
        if sort_rate > 0:
            per_stage_hours.append(sort_rem / sort_rate)
        hours_remaining = max(per_stage_hours) if per_stage_hours else None
        predicted_finish = None
        if hours_remaining is not None:
            predicted_finish = (now + timedelta(hours=hours_remaining)).date().isoformat()

        # Phase-aware plan using billing hours/day (6h/day)
        BILLABLE_HOURS_PER_DAY = 6.0
        rate_review = rev_rate or stage_baselines.get("reviewed") or 0.0
        rate_crop = crop_rate or stage_baselines.get("crop") or 0.0
        rate_sort = sort_rate or stage_baselines.get("sort") or 0.0
        plan_hours_review = (rev_rem / rate_review) if rate_review > 0 else None
        # For crop, use crop_rem; for sort, use sort_rem after crop done
        plan_hours_crop = (crop_rem / rate_crop) if rate_crop > 0 else None
        plan_hours_sort = (sort_rem / rate_sort) if rate_sort > 0 else None
        total_plan_hours = sum([h for h in [plan_hours_review, plan_hours_crop, plan_hours_sort] if h is not None]) if any([plan_hours_review, plan_hours_crop, plan_hours_sort]) else None
        predicted_finish_billing = None
        if total_plan_hours is not None:
            days = total_plan_hours / BILLABLE_HOURS_PER_DAY
            predicted_finish_billing = (now + timedelta(days=days)).date().isoformat()

        # Pace status vs baseline
        pace_status = "UNKNOWN"
        if baseline_iph > 0 and current_rate_overall > 0:
            if current_rate_overall > baseline_iph * 1.1:
                pace_status = "AHEAD"
            elif current_rate_overall < baseline_iph * 0.9:
                pace_status = "BEHIND"
            else:
                pace_status = "ON TRACK"
        elif current_rate_overall == 0:
            pace_status = "PAUSED"

        # Determine current phase and compute phase-aware milestones
        def hours_remaining_for(count_remaining: int, rate: float) -> Optional[float]:
            return (count_remaining / rate) if rate and rate > 0 else None

        current_phase = "review" if reviewed_done < total_images else ("crop" if crop_done < total_to_crop else ("sort" if sort_done < total_images else "done"))
        phase_target = {
            "review": total_images,
            "crop": total_to_crop,
            "sort": total_images,
        }.get(current_phase, total_images)
        phase_done = {
            "review": reviewed_done,
            "crop": crop_done,
            "sort": sort_done,
        }.get(current_phase, processed_images)
        phase_rate = {
            "review": (rev_rate or stage_baselines.get("reviewed") or 0.0),
            "crop": (crop_rate or stage_baselines.get("crop") or 0.0),
            "sort": (sort_rate or stage_baselines.get("sort") or 0.0),
        }.get(current_phase, current_rate_overall)

        milestones = []
        for frac in [0.25, 0.5, 0.75, 1.0]:
            tgt = int(round(phase_target * frac))
            remaining_to_hit = max(0, tgt - phase_done)
            hrs = hours_remaining_for(remaining_to_hit, phase_rate)
            target_date = None
            if hrs is not None:
                days = hrs / 6.0
                target_date = (now + timedelta(days=days)).date().isoformat()
            milestones.append({
                "label": f"{int(frac*100)}%",
                "target_images": tgt,
                "target_date": target_date,
                "remaining": remaining_to_hit,
                "phase": current_phase,
                "est_hours": round(hrs, 2) if hrs is not None else None,
            })

        # Chart series
        chart = build_time_series_for_chart(proj_metrics, started_at, total_images)

        # Original estimate: based on baseline overall rate
        original_estimate = None
        if baseline_iph > 0 and total_images > 0 and started_at:
            hrs = total_images / baseline_iph
            original_estimate = (_parse_iso(started_at) + timedelta(hours=hrs)).date().isoformat()

        # Compute image-only all-time operation counts (delete should reflect PNGs only)
        def image_only_ops(records: List[Dict[str, Any]]) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for r in records:
                op = str(r.get("operation") or "").lower()
                cnt = int(r.get("file_count") or 0)
                # Prefer explicit image-only signals
                notes = str(r.get("notes") or "").lower()
                files = r.get("files") or r.get("files_sample") or []
                png_count = 0
                try:
                    if isinstance(files, list) and files:
                        png_count = sum(1 for f in files if str(f).lower().endswith('.png'))
                except Exception:
                    png_count = 0
                if 'image-only' in notes:
                    out[op] = out.get(op, 0) + cnt
                elif png_count > 0:
                    out[op] = out.get(op, 0) + png_count
                else:
                    # If we cannot verify, count only crops directly (they are image-native)
                    if op == 'crop':
                        out[op] = out.get(op, 0) + cnt
            return out

        try:
            all_records = engine._load_from_detailed_logs(start_date=None, end_date=None)  # type: ignore[attr-defined]
            proj_all_records = [r for r in all_records if rec_in_project(r)]
            ops_image_only = image_only_ops(proj_all_records)
        except Exception:
            ops_image_only = ops_by_type

        # Read-only inventory using manifest paths
        paths = active.get('paths') or {}
        root_dir = paths.get('root')
        selected_dir = paths.get('selectedDir') or str(PROJECT_ROOT / '__selected')
        crop_dir = paths.get('cropDir') or str(PROJECT_ROOT / '__crop')
        crop_auto_dir = str(PROJECT_ROOT / '__crop_auto')
        crop_cropped_dir = str(PROJECT_ROOT / '__cropped')
        crop_auto_cropped_dir = str(PROJECT_ROOT / '__cropped')
        delete_staging_dir = str(PROJECT_ROOT / '__delete_staging')

        # Cache key based on directory mtimes (so we don't rescan every 30s)
        inv_sig = (
            total_images,
            _latest_mtime(root_dir),
            _latest_mtime(selected_dir),
            _latest_mtime(crop_dir) + _latest_mtime(crop_auto_dir),
            _latest_mtime(crop_cropped_dir) + _latest_mtime(crop_auto_cropped_dir),
            _latest_mtime(delete_staging_dir),
        )
        inv_payload: Dict[str, Any]
        cache_hit = False
        try:
            cache = INVENTORY_CACHE.get(project_id)
            if cache and cache.get('sig') == inv_sig and (datetime.utcnow().timestamp() - cache.get('ts', 0)) < 600:
                inv_payload = cache.get('payload', {})
                cache_hit = True
            else:
                raise KeyError
        except Exception:
            # Recompute counts only when signature changed or cache stale
            inv_remaining_root = _count_pngs(root_dir)
            inv_selected = _count_pngs(selected_dir)
            inv_crop_queue = _count_pngs(crop_dir) + _count_pngs(crop_auto_dir)
            inv_cropped_done = _count_pngs(crop_cropped_dir) + _count_pngs(crop_auto_cropped_dir)
            inv_deleted = _count_pngs(delete_staging_dir)
            inv_outside_root = inv_selected + inv_crop_queue + inv_cropped_done + inv_deleted
            inv_reviewed_total = max(0, (total_images - inv_remaining_root))
            inv_payload = {
                "initial": total_images,
                "remaining_root": inv_remaining_root,
                "selected": inv_selected,
                "crop_queue": inv_crop_queue,
                "cropped_done": inv_cropped_done,
                "deleted": inv_deleted,
                "outside_root": inv_outside_root,
                "reviewed_total": inv_reviewed_total,
            }
            INVENTORY_CACHE[project_id] = {"sig": inv_sig, "payload": inv_payload, "ts": datetime.utcnow().timestamp()}

        resp = {
            "project": {
                "projectId": project_id,
                "title": title,
                "startedAt": started_at,
                "totalImages": total_images,
                "processedImages": processed_images,
                "percentComplete": round((processed_images / total_images) * 100.0, 2) if total_images > 0 else 0.0,
            },
            "pace": {
                "baselineImagesPerHour": baseline_iph,
                "currentImagesPerHour": current_rate_overall,
                "status": pace_status,
            },
            "prediction": {
                "hoursRemaining": round(hours_remaining, 2) if hours_remaining is not None else None,
                "predictedFinishDate": predicted_finish,
                "originalEstimateDate": original_estimate,
                "billingPredictedFinishDate": predicted_finish_billing,
            },
            "stages": {
                "reviewed": {"done": reviewed_done, "ratePerHour": rev_rate, "remaining": rev_rem, "hoursRemaining": round((rev_rem/(rev_rate or stage_baselines.get('reviewed') or 0.0)),2) if (rev_rate or stage_baselines.get('reviewed')) else None},
                "selection": {"done": selection_done, "ratePerHour": sel_rate, "remaining": sel_rem, "hoursRemaining": round((sel_rem/(sel_rate or stage_baselines.get('selection') or 0.0)),2) if (sel_rate or stage_baselines.get('selection')) else None},
                "crop": {"done": crop_done, "ratePerHour": crop_rate, "remaining": crop_rem, "hoursRemaining": round((crop_rem/(crop_rate or stage_baselines.get('crop') or 0.0)),2) if (crop_rate or stage_baselines.get('crop')) else None},
                "sort": {"done": sort_done, "ratePerHour": sort_rate, "remaining": sort_rem, "hoursRemaining": round((sort_rem/(sort_rate or stage_baselines.get('sort') or 0.0)),2) if (sort_rate or stage_baselines.get('sort')) else None},
            },
            "stageBaselines": stage_baselines,
            "plan": {
                "billableHoursPerDay": BILLABLE_HOURS_PER_DAY,
                "hours": {
                    "review": round(plan_hours_review, 2) if plan_hours_review is not None else None,
                    "crop": round(plan_hours_crop, 2) if plan_hours_crop is not None else None,
                    "sort": round(plan_hours_sort, 2) if plan_hours_sort is not None else None,
                    "total": round(total_plan_hours, 2) if total_plan_hours is not None else None
                }
            },
            "topBar": {
                "reviewed": reviewed_done,
                "end_to_end": processed_images,
                "total": total_images,
                "recommended": recommended_topbar,
            },
            "milestones": milestones,
            "timeseries": chart,
            "allTime": {
                "workHours": all_time_work_hours,
                "operationsByType": ops_image_only,
            },
            "inventory": inv_payload,
        }

        # No-cache headers managed by Flask config in upstream dashboards; keep simple here
        return jsonify(resp)

    @app.route("/api/patterns")
    def patterns_api():
        """Return historical pattern analysis with light caching."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        cache_path = CACHE_DIR / "pattern_analysis.json"

        # Build cache key from finished projects (projectId:finishedAt)
        projects = agg.aggregate()
        finished_pairs = []
        for pj in projects.values():
            if pj.get("finishedAt"):
                finished_pairs.append(f"{pj.get('projectId')}|{pj.get('finishedAt')}")
        cache_key = {"finished": sorted(finished_pairs), "version": 1}

        # Load cache if valid
        try:
            if cache_path.exists():
                cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
                if cache_data.get("cache_key") == cache_key:
                    return jsonify(cache_data.get("payload", {}))
        except Exception:
            pass

        # Compute fresh patterns
        payload = _compute_patterns(projects, engine)

        # Write cache best-effort
        try:
            cache_blob = {"cache_key": cache_key, "payload": payload, "generatedAt": datetime.utcnow().isoformat() + "Z"}
            cache_path.write_text(json.dumps(cache_blob, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        return jsonify(payload)

    # Static route for Chart.js if needed (fallback if not using CDN)
    @app.route("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(str(Path(__file__).parent / "static"), filename)

    return app


def main():
    app = create_app()
    host = "127.0.0.1"
    port = 8082
    # Auto-launch browser shortly after server starts (local only)
    threading.Thread(target=launch_browser, args=(host, port, 1.2), daemon=True).start()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()


