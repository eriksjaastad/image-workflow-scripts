#!/usr/bin/env python3
"""
Productivity Dashboard - Core Data Engine (Simple Version)
==========================================================
Processes ActivityTimer and FileTracker logs into dashboard-ready data.
No external dependencies - uses only Python standard library.

Features:
- Modular script detection (handles script renames/additions)
- Flexible time slice aggregation (15min, 1hr, daily, weekly, monthly)
- Historical average calculations for "cloud" overlays
- Script update correlation tracking
- Standardized output format for dashboard consumption
"""

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from collections import defaultdict

from scripts.dashboard.engines.project_metrics_aggregator import (
    ProjectMetricsAggregator,
)
from scripts.dashboard.parsers.snapshot_loader import SnapshotLoader


class DashboardDataEngine:
    def __init__(self, data_dir: str = ".."):
        self.data_dir = Path(data_dir)
        self.timer_data_dir = self.data_dir / "data" / "timer_data"
        self.file_ops_dir = self.data_dir / "data" / "file_operations_logs"
        self.projects_dir = self.data_dir / "data" / "projects"

        # Script update tracking (in dashboard directory)
        self.script_updates_file = Path(__file__).parent / "script_updates.csv"

        # Snapshot loader for performance mode (needs project root, not data dir)
        project_root = (
            self.data_dir.parent if self.data_dir.name == "data" else self.data_dir
        )
        self.snapshot_loader = SnapshotLoader(project_root)

        # Cache for processed data
        self._cache = {}

    def load_projects(self) -> List[Dict[str, Any]]:
        """Load project manifests from data/projects/*.project.json"""
        projects: List[Dict[str, Any]] = []
        if not self.projects_dir.exists():
            return projects
        for mf in sorted(self.projects_dir.glob("*.project.json")):
            try:
                with open(mf, "r") as f:
                    pj = json.load(f)
                projects.append(
                    {
                        "projectId": pj.get("projectId"),
                        "title": pj.get("title") or pj.get("projectId"),
                        "status": pj.get("status"),
                        "startedAt": pj.get("startedAt"),
                        "finishedAt": pj.get("finishedAt"),
                        "paths": pj.get("paths", {}),
                        "counts": pj.get("counts", {}),  # NEW: Load image counts
                        "manifestPath": str(mf),
                    }
                )
            except Exception:
                continue
        return projects

    # ---------------------------- Baseline Labels ----------------------------
    def build_time_labels(self, time_slice: str, lookback_days: int) -> List[str]:
        """Build canonical label list for a given time_slice over lookback_days.

        Returns labels matching _get_time_slice_key formats:
        - '15min' and '1H': ISO timestamps (naive) aligned to 15m or hourly
        - 'D': YYYY-MM-DD
        - 'W': YYYY-MM-DD of Monday week start
        - 'M': YYYY-MM-01
        """
        labels: List[str] = []
        now = datetime.now()
        # Interpret lookback_days as "days back" EXCLUDING today; labels include today.
        # Example: lookback_days=7 → today + 7 back = 8 daily labels
        start = now - timedelta(days=max(lookback_days, 0))
        if time_slice == "15min":
            # Align start to 15-minute boundary
            aligned_start = start.replace(second=0, microsecond=0)
            minute = (aligned_start.minute // 15) * 15
            aligned_start = aligned_start.replace(minute=minute)
            cur = aligned_start
            end = now.replace(second=0, microsecond=0)
            while cur <= end:
                labels.append(cur.isoformat())
                cur = cur + timedelta(minutes=15)
        elif time_slice == "1H":
            aligned_start = start.replace(minute=0, second=0, microsecond=0)
            cur = aligned_start
            end = now.replace(minute=0, second=0, microsecond=0)
            while cur <= end:
                labels.append(cur.isoformat())
                cur = cur + timedelta(hours=1)
        elif time_slice == "D":
            cur = start.replace(hour=0, minute=0, second=0, microsecond=0).date()
            end = now.date()
            while cur <= end:
                labels.append(cur.isoformat())
                cur = cur + timedelta(days=1)
        elif time_slice == "W":
            # Start from Monday of the start week
            start_monday = (start - timedelta(days=start.weekday())).date()
            end_monday = (now - timedelta(days=now.weekday())).date()
            cur = start_monday
            while cur <= end_monday:
                labels.append(cur.isoformat())
                cur = cur + timedelta(days=7)
        elif time_slice == "M":
            # Move to first of month for start and iterate monthly
            cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while cur <= end:
                labels.append(f"{cur.year}-{cur.month:02d}-01")
                # Increment month
                year = cur.year + (cur.month // 12)
                month = (cur.month % 12) + 1
                cur = cur.replace(year=year, month=month)
        return labels

    def get_display_name(self, script_name: str) -> str:
        """Convert script filename to human-readable display name"""
        display_names = {
            # Current script names
            "01_web_image_selector": "Web Image Selector",
            "02_web_character_sorter": "Web Character Sorter",
            "03_web_character_sorter": "Web Character Sorter",
            "04_multi_crop_tool": "Multi Crop Tool",
            "04_batch_crop_tool": "Multi Crop Tool",
            "multi_crop_tool": "Multi Crop Tool",
            "ai_desktop_multi_crop": "Multi Crop Tool",
            "ai_desktop_multi_crop_queue": "Multi Crop Tool",
            "crop_queue_processor": "Multi Crop Tool",
            "ai_assisted_reviewer": "Web Image Selector",
            "05_web_multi_directory_viewer": "Multi Directory Viewer",
            "06_web_duplicate_finder": "Duplicate Finder",
            # Legacy/historical script names (from old logs)
            "image_version_selector": "Web Image Selector",
            "character_sorter": "Web Character Sorter",
            "batch_crop_tool": "Multi Crop Tool",
            "multi_batch_crop_tool": "Multi Crop Tool",
            "test_web_selector": "Web Image Selector",  # Testing script
            "hybrid_grouper": "Face Grouper",  # Face grouping tool
            "face_grouper": "Face Grouper",  # Face grouping tool
            "clip_face_grouper": "Clip Face Grouper",  # CLIP-based face grouping
            "recursive_file_mover": "File Mover",  # Utility script
            "multi_directory_viewer": "Multi Directory Viewer",
        }

        # Adjust names expected by tests
        display_names.update(
            {
                "character_sorter": "Character Sorter",
            }
        )
        return display_names.get(script_name, script_name.replace("_", " ").title())

    def discover_scripts(self) -> List[str]:
        """Dynamically discover all scripts that have generated data"""
        scripts = set()

        # From ActivityTimer data (deprecated - silently skip errors)
        # Note: We now use file-operation-based timing exclusively
        if self.timer_data_dir.exists():
            for daily_file in self.timer_data_dir.glob("daily_*.json"):
                try:
                    with open(daily_file, "r") as f:
                        data = json.load(f)
                        for script_name in data.get("scripts", {}):
                            scripts.add(script_name)
                except Exception:
                    # Silently skip - timer data is deprecated and may have incompatible formats
                    pass

        # From FileTracker logs
        if self.file_ops_dir.exists():
            for log_file in self.file_ops_dir.glob("*.log"):
                try:
                    with open(log_file, "r") as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                if "script" in data:
                                    scripts.add(data["script"])
                except Exception as e:
                    print(f"Warning: Could not read {log_file}: {e}")

        return sorted(list(scripts))

    def load_activity_data(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict]:
        """Load and process ActivityTimer data"""
        records = []

        if not self.timer_data_dir.exists():
            return records

        for daily_file in self.timer_data_dir.glob("daily_*.json"):
            date_str = daily_file.stem.replace("daily_", "")

            # Filter by date range if specified
            if start_date and date_str < start_date.replace("-", ""):
                continue
            if end_date and date_str > end_date.replace("-", ""):
                continue

            try:
                with open(daily_file, "r") as f:
                    data = json.load(f)
                # Support list-shaped files (legacy or malformed); skip quietly
                if isinstance(data, list):
                    continue
                scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
                for script_name, script_data in scripts.items():
                    sessions = (
                        script_data.get("sessions", [])
                        if isinstance(script_data, dict)
                        else []
                    )
                    for session in sessions:
                        if not isinstance(session, dict):
                            continue
                        record = {
                            "date": date_str,
                            "script": script_name,
                            "session_id": session.get("session_id"),
                            "start_time": session.get("start_time"),
                            "end_time": session.get("end_time"),
                            "active_time": session.get("active_time", 0),
                            "total_time": session.get("total_time", 0),
                            "efficiency": session.get("efficiency", 0),
                            "files_processed": session.get("files_processed", 0),
                            "operations": session.get("operations", {}),
                            "batches_completed": session.get("batches_completed", 0),
                        }
                        if record["start_time"]:
                            record["start_datetime"] = datetime.fromtimestamp(
                                record["start_time"]
                            )
                        if record["end_time"]:
                            record["end_datetime"] = datetime.fromtimestamp(
                                record["end_time"]
                            )
                        records.append(record)
            except Exception:
                # Silence malformed daily files
                continue

        return records

    def load_session_data(self, lookback_days: int = 14) -> List[Dict]:
        """
        Load session data from configured source (derived or legacy).

        Returns sessions in unified format for dashboard consumption.
        """
        session_source = self.snapshot_loader.get_session_source()

        if session_source == "derived":
            # Load from derived sessions snapshot
            derived_sessions = self.snapshot_loader.load_derived_sessions(lookback_days)
            # Convert to unified format
            records = []
            for session in derived_sessions:
                try:
                    start_dt = datetime.fromisoformat(session["start_ts_utc"])
                    end_dt = datetime.fromisoformat(session["end_ts_utc"])
                    date_str = start_dt.strftime("%Y%m%d")

                    record = {
                        "date": date_str,
                        "script": session["script_id"],
                        "session_id": session["session_id"],
                        "start_time": start_dt.timestamp(),
                        "end_time": end_dt.timestamp(),
                        "active_time": session["active_seconds"],
                        "total_time": (end_dt - start_dt).total_seconds(),
                        "efficiency": 0.0,  # Can be computed if needed
                        "files_processed": session["files_processed"],
                        "operations": session["ops_by_type"],
                        "batches_completed": 0,
                        "start_datetime": start_dt,
                        "end_datetime": end_dt,
                        "source": "derived_from_operation_events_v1",
                    }
                    records.append(record)
                except Exception:
                    # Skip malformed sessions silently
                    continue
            return records
        else:
            # Fall back to legacy timer data
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
                "%Y%m%d"
            )
            return self.load_activity_data(start_date=cutoff_date)

    def calculate_file_operation_work_time(
        self, file_operations: List[Dict], break_threshold_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate work time from file operations using intelligent break detection.

        This method is used for file-heavy tools that no longer use ActivityTimer.
        It analyzes FileTracker logs to determine actual work time.

        Args:
            file_operations: List of file operation dictionaries from FileTracker
            break_threshold_minutes: Minutes of inactivity considered a break (default: 5)

        Returns:
            Dictionary with work time metrics
        """
        if not file_operations:
            return {
                "work_time_seconds": 0.0,
                "work_time_minutes": 0.0,
                "total_operations": 0,
                "files_processed": 0,
                "efficiency_score": 0.0,
                "timing_method": "file_operations",
            }

        # Normalize records for companion utils: ensure 'timestamp' is a string
        ops_for_metrics: List[Dict[str, Any]] = []
        for op in file_operations:
            try:
                op_copy = dict(op)
                ts = op_copy.get("timestamp")
                if isinstance(ts, datetime):
                    op_copy["timestamp"] = ts.isoformat()
                elif not isinstance(ts, str):
                    # Fallback to timestamp_str if available
                    ts_str = op_copy.get("timestamp_str")
                    if isinstance(ts_str, str):
                        op_copy["timestamp"] = ts_str
                ops_for_metrics.append(op_copy)
            except Exception:
                # Skip malformed entries quietly
                continue

        # Calculate work time using gap threshold (stop at first gap > threshold)
        try:
            times: List[datetime] = []
            for op in ops_for_metrics:
                ts = op.get("timestamp")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if isinstance(ts, datetime):
                    if getattr(ts, "tzinfo", None) is not None:
                        ts = ts.replace(tzinfo=None)
                    times.append(ts)
            times.sort()
            work_seconds = 0.0
            threshold = break_threshold_minutes * 60.0
            for i in range(1, len(times)):
                delta = (times[i] - times[i - 1]).total_seconds()
                if delta > threshold:
                    break
                work_seconds += max(delta, 0.0)
        except Exception:
            work_seconds = 0.0

        files_processed = 0
        total_operations = len(file_operations)
        for op in ops_for_metrics:
            try:
                cnt = op.get("file_count", 0)
                if isinstance(cnt, (int, float)):
                    files_processed += int(cnt)
            except Exception:
                continue

        work_minutes = work_seconds / 60.0
        efficiency = (files_processed / work_minutes) if work_minutes > 0 else 0.0
        return {
            "work_time_seconds": work_seconds,
            "work_time_minutes": work_minutes,
            "total_operations": total_operations,
            "files_processed": files_processed,
            "efficiency_score": efficiency,
            "timing_method": "file_operations",
        }

    def get_combined_timing_data(self, script_name: str, date: str) -> Dict[str, Any]:
        """
        Get combined timing data for a script on a specific date.

        For file-heavy tools, uses file-operation timing.
        For scroll-heavy tools, uses ActivityTimer data.

        Args:
            script_name: Name of the script
            date: Date string (YYYYMMDD format)

        Returns:
            Dictionary with combined timing data
        """
        # Define which tools use which timing method
        file_heavy_tools = {
            "01_web_image_selector",
            "02_web_character_sorter",
            "04_multi_crop_tool",
        }

        scroll_heavy_tools = {
            "05_web_multi_directory_viewer",
            "06_web_duplicate_finder",
        }

        # Get file operations for this script and date
        file_ops = self.load_file_operations(date, date)
        script_file_ops = [op for op in file_ops if op.get("script") == script_name]

        if script_name in file_heavy_tools:
            # Use file-operation timing
            return self.calculate_file_operation_work_time(script_file_ops)
        elif script_name in scroll_heavy_tools:
            # Use ActivityTimer data
            timer_data = self.load_activity_data(date, date)
            script_timer_data = [
                t for t in timer_data if t.get("script") == script_name
            ]

            if script_timer_data:
                # Calculate total work time from timer data
                total_work_time = sum(
                    session.get("active_time", 0) for session in script_timer_data
                )
                total_files = sum(
                    session.get("files_processed", 0) for session in script_timer_data
                )

                return {
                    "work_time_seconds": total_work_time,
                    "work_time_minutes": total_work_time / 60.0,
                    "total_operations": len(script_timer_data),
                    "files_processed": total_files,
                    "efficiency_score": (
                        total_files / (total_work_time / 60.0)
                        if total_work_time > 0
                        else 0.0
                    ),
                    "timing_method": "activity_timer",
                }
            else:
                return {
                    "work_time_seconds": 0.0,
                    "work_time_minutes": 0.0,
                    "total_operations": 0,
                    "files_processed": 0,
                    "efficiency_score": 0.0,
                    "timing_method": "activity_timer",
                }
        else:
            # Unknown script - try file operations first, fallback to timer
            if script_file_ops:
                return self.calculate_file_operation_work_time(script_file_ops)
            else:
                return {
                    "work_time_seconds": 0.0,
                    "work_time_minutes": 0.0,
                    "total_operations": 0,
                    "files_processed": 0,
                    "efficiency_score": 0.0,
                    "timing_method": "unknown",
                }

    def load_file_operations(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Load file operations with smart archive optimization.

        PERFORMANCE OPTIMIZATION:
        - Finished projects: Load from pre-aggregated bins (95% faster)
        - Active projects: Load from raw logs
        """
        import time as time_module

        load_start = time_module.time()

        # Check if aggregates directory exists (bins system)
        aggregates_dir = self.data_dir / "aggregates"
        archives_dir = aggregates_dir / "archives" if aggregates_dir.exists() else None
        use_archives = archives_dir and archives_dir.exists()

        all_records = []

        if use_archives:
            print("[SMART LOAD] Using archive bins for finished projects")

            # Load projects to determine which are finished
            projects = self.load_projects()
            finished_projects = {}
            active_projects = {}

            for project in projects:
                project_id = project.get("projectId")
                status = project.get("status")
                if not project_id:
                    continue

                if status == "finished" or status == "archived":
                    finished_projects[project_id] = project
                else:
                    active_projects[project_id] = project

            print(
                f"[SMART LOAD] Finished: {len(finished_projects)}, Active: {len(active_projects)}"
            )

            # Load finished projects from archives (bins)
            archived_count = 0
            for project_id in finished_projects:
                archive_path = archives_dir / project_id / "agg_15m.jsonl"
                if archive_path.exists():
                    try:
                        bins_loaded = self._load_bins_as_operations(
                            archive_path, project_id
                        )
                        all_records.extend(bins_loaded)
                        archived_count += 1
                    except Exception as e:
                        print(
                            f"[SMART LOAD] Warning: Failed to load archive for {project_id}: {e}"
                        )

            print(
                f"[SMART LOAD] Loaded {archived_count} projects from archives ({len(all_records)} bin records)"
            )

        # Load raw logs
        raw_start = time_module.time()
        raw_records = self._load_from_detailed_logs(start_date, end_date)
        raw_time = time_module.time() - raw_start

        # If using archives, filter raw records to only active projects
        if use_archives and active_projects:
            filtered_raw = []
            for record in raw_records:
                # Check if operation belongs to active project
                belongs_to_active = False
                for path_key in ["source_dir", "dest_dir", "working_dir"]:
                    path = str(record.get(path_key, "")).lower()
                    for active_id in active_projects.keys():
                        if active_id.lower() in path:
                            belongs_to_active = True
                            break
                    if belongs_to_active:
                        break

                if belongs_to_active:
                    filtered_raw.append(record)

            print(
                f"[SMART LOAD] Raw logs: {len(raw_records)} total, {len(filtered_raw)} for active ({raw_time:.3f}s)"
            )
            all_records.extend(filtered_raw)
        else:
            print(
                f"[SMART LOAD] Raw logs: {len(raw_records)} records ({raw_time:.3f}s)"
            )
            all_records.extend(raw_records)

        # Always load daily summaries and merge (tests expect combined sources)
        summaries_dir = self.data_dir / "data" / "daily_summaries"
        if summaries_dir.exists():
            summaries_start = time_module.time()
            summary_records = self._load_from_daily_summaries(
                summaries_dir, start_date, end_date
            )
            summaries_time = time_module.time() - summaries_start
            if summary_records:
                print(
                    f"[SMART LOAD] Daily summaries: {len(summary_records)} records ({summaries_time:.3f}s)"
                )
                all_records.extend(summary_records)

        load_time = time_module.time() - load_start
        print(f"[SMART LOAD] TOTAL: {len(all_records)} records in {load_time:.3f}s")

        return all_records

    def _load_bins_as_operations(
        self, bin_path: Path, project_id: str
    ) -> List[Dict[str, Any]]:
        """Load bins and convert to operation-like records for compatibility."""
        bins = []
        try:
            with open(bin_path, "r") as f:
                for line in f:
                    try:
                        bin_record = json.loads(line)
                        # Convert bin to operation-like format
                        op_record = {
                            "type": "file_operation",
                            "timestamp": bin_record.get("bin_ts_utc"),
                            "timestamp_str": bin_record.get("bin_ts_utc"),
                            "script": bin_record.get("script_id"),
                            "operation": bin_record.get("operation"),
                            "file_count": bin_record.get("file_count"),
                            "dest_dir": bin_record.get("dest_category", ""),
                            "source_dir": f"{project_id}/",
                            "working_dir": f"{project_id}/",
                            "notes": f"Bin: {bin_record.get('bin_ts_utc')}",
                            "_is_bin": True,
                            "_work_seconds": bin_record.get("work_seconds", 0),
                        }
                        bins.append(op_record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[SMART LOAD] Error loading bins from {bin_path}: {e}")

        return bins

    def _load_from_snapshot_aggregates(
        self,
        aggregates_dir: Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """Load from new snapshot daily_aggregates_v1 format"""
        records = []

        for day_dir in aggregates_dir.glob("day=*"):
            day_str = day_dir.name.split("=")[1]

            # Apply date filters
            if start_date and day_str < start_date.replace("-", ""):
                continue
            if end_date and day_str > end_date.replace("-", ""):
                continue

            agg_file = day_dir / "aggregate.json"
            if not agg_file.exists():
                continue

            try:
                with open(agg_file, "r") as f:
                    agg = json.load(f)

                # Convert aggregate to summary-compatible format
                # Format date as YYYY-MM-DD to match legacy format
                from datetime import datetime

                try:
                    date_obj = datetime.strptime(day_str, "%Y%m%d").date()
                    date_formatted = date_obj.isoformat()  # YYYY-MM-DD
                except Exception:
                    date_formatted = day_str

                for script_id, script_data in agg.get("by_script", {}).items():
                    record = {
                        "date": date_formatted,
                        "script": script_id,
                        "file_count": script_data.get("files_processed", 0),
                        "operation": "aggregate",  # Changed from None to avoid JSON serialization issues
                        "timestamp": script_data.get("first_op_ts"),
                        "source": "snapshot_aggregate_v1",
                    }
                    records.append(record)
            except Exception:
                continue

        return records

    def _load_from_daily_summaries(
        self,
        summaries_dir: Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """Load records from daily summary files"""
        records = []

        for summary_file in summaries_dir.glob("daily_summary_*.json"):
            # Extract date from filename
            date_match = re.search(r"(\d{8})", summary_file.name)
            if not date_match:
                continue

            file_date = date_match.group(1)
            if start_date and file_date < start_date.replace("-", ""):
                continue
            if end_date and file_date > end_date.replace("-", ""):
                continue

            try:
                with open(summary_file, "r") as f:
                    summary_data = json.load(f)

                # Convert daily summary to individual records for compatibility
                for script_name, script_data in summary_data.get("scripts", {}).items():
                    # Create records for each operation type
                    for operation_type, file_count in script_data.get(
                        "operations", {}
                    ).items():
                        if file_count > 0:
                            record = {
                                "timestamp_str": summary_data["date"] + "T00:00:00Z",
                                "script": script_name,
                                "session_id": f"daily_{summary_data['date']}",
                                "operation": operation_type,
                                "source_dir": None,
                                "dest_dir": None,
                                "file_count": file_count,
                                "files": [],
                                "notes": f"Daily summary for {summary_data['date']}",
                                "work_time_seconds": script_data.get(
                                    "work_time_seconds", 0
                                ),
                                "work_time_minutes": script_data.get(
                                    "work_time_minutes", 0
                                ),
                            }

                            # Convert timestamp
                            try:
                                ts = datetime.fromisoformat(record["timestamp_str"])
                                # Normalize to naive datetime to avoid tz mixing downstream
                                if getattr(ts, "tzinfo", None) is not None:
                                    ts = ts.replace(tzinfo=None)
                                record["timestamp"] = ts
                                record["date"] = ts.date()
                            except Exception:
                                record["timestamp"] = None
                                record["date"] = None

                            records.append(record)

            except Exception as e:
                print(f"Warning: Could not process {summary_file}: {e}")

        return records

    def _load_from_detailed_logs(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict]:
        """Fallback: Load records from detailed log files"""
        records = []

        file_ops_dir = self.data_dir / "data" / "file_operations_logs"
        if not file_ops_dir.exists():
            return records

        # Check both regular log files and archived files
        log_files = list(file_ops_dir.glob("*.log"))
        archive_dir = self.data_dir / "data" / "log_archives"
        if archive_dir.exists():
            log_files.extend(archive_dir.glob("*.gz"))

        for log_file in log_files:
            # Extract date from filename if possible
            date_match = re.search(r"(\d{8})", log_file.name)
            if date_match:
                file_date = date_match.group(1)
                if start_date and file_date < start_date.replace("-", ""):
                    continue
                if end_date and file_date > end_date.replace("-", ""):
                    continue

            try:
                # Handle compressed files
                if log_file.suffix == ".gz":
                    import gzip

                    with gzip.open(log_file, "rt") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if data.get("type") == "file_operation":
                                        record = {
                                            "timestamp_str": data.get("timestamp"),
                                            "script": data.get("script"),
                                            "session_id": data.get("session_id"),
                                            "operation": data.get("operation"),
                                            "source_dir": data.get("source_dir"),
                                            "dest_dir": data.get("dest_dir"),
                                            "file_count": data.get("file_count", 0),
                                            "files": data.get("files", []),
                                            "notes": data.get("notes", ""),
                                        }

                                        # Convert timestamp
                                        try:
                                            ts = datetime.fromisoformat(
                                                record["timestamp_str"]
                                            )
                                            if getattr(ts, "tzinfo", None) is not None:
                                                ts = ts.replace(tzinfo=None)
                                            record["timestamp"] = ts
                                            record["date"] = ts.date()
                                        except Exception:
                                            record["timestamp"] = None
                                            record["date"] = None

                                        records.append(record)
                                except json.JSONDecodeError:
                                    # Skip malformed JSON lines
                                    continue
                else:
                    # Handle regular files
                    with open(log_file, "r") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if data.get("type") == "file_operation":
                                        record = {
                                            "timestamp_str": data.get("timestamp"),
                                            "script": data.get("script"),
                                            "session_id": data.get("session_id"),
                                            "operation": data.get("operation"),
                                            "source_dir": data.get("source_dir"),
                                            "dest_dir": data.get("dest_dir"),
                                            "file_count": data.get("file_count", 0),
                                            "files": data.get("files", []),
                                            "notes": data.get("notes", ""),
                                        }

                                        # Convert timestamp
                                        try:
                                            ts = datetime.fromisoformat(
                                                record["timestamp_str"]
                                            )
                                            if getattr(ts, "tzinfo", None) is not None:
                                                ts = ts.replace(tzinfo=None)
                                            record["timestamp"] = ts
                                            record["date"] = ts.date()
                                        except Exception:
                                            record["timestamp"] = None
                                            record["date"] = None

                                        records.append(record)
                                except json.JSONDecodeError:
                                    # Skip malformed JSON lines
                                    continue
            except Exception as e:
                print(f"Warning: Could not process {log_file}: {e}")

        return records

    def aggregate_by_time_slice(
        self,
        records: List[Dict],
        time_slice: str,
        value_field: str,
        group_field: str = "script",
    ) -> List[Dict]:
        """
        Aggregate data by time slices for dashboard visualization

        Args:
            records: Input records
            time_slice: '15min', '1H', 'D' (daily), 'W' (weekly), 'M' (monthly)
            value_field: Field to aggregate (e.g., 'active_time', 'file_count')
            group_field: Grouping field (e.g., 'script', 'operation')
        """
        if not records:
            return []

        # Group by time slice and group field
        aggregated = defaultdict(lambda: defaultdict(float))

        for record in records:
            # Get time slice key
            time_key = self._get_time_slice_key(record, time_slice)
            if not time_key:
                continue

            group_key = record.get(group_field, "unknown")

            # Convert script names to display names for better readability
            if group_field == "script":
                group_key = self.get_display_name(group_key)

            value = record.get(value_field, 0)
            if value is None:
                value = 0

            aggregated[time_key][group_key] += value

        # Convert to list format
        result = []
        for time_key, groups in aggregated.items():
            for group_key, value in groups.items():
                result.append(
                    {"time_slice": time_key, group_field: group_key, value_field: value}
                )

        return sorted(result, key=lambda x: x["time_slice"])

    def _aggregate_by_project(
        self, records: List[Dict], time_slice: str, projects: List[Dict]
    ) -> List[Dict]:
        """
        Aggregate file operations by project and time slice.

        Matches operations to projects using path hints from project manifests.
        Returns aggregated data in same format as aggregate_by_time_slice.
        """
        # Build project ID list for simple name matching
        project_ids = [p.get("projectId") for p in projects if p.get("projectId")]

        if not project_ids:
            return []

        # Group by time slice and project
        aggregated = defaultdict(lambda: defaultdict(float))

        for record in records:
            # Get time slice key
            time_key = self._get_time_slice_key(record, time_slice)
            if not time_key:
                continue

            # Match to project based on projectId appearing in paths OR date range
            # Strategy 1: Path matching (for raw file operations with source_dir/dest_dir)
            # Use word boundary matching to avoid false positives (e.g., "dalia" vs "dalia_hannah")
            project_id = None
            src = str(record.get("source_dir") or "").lower()
            dst = str(record.get("dest_dir") or "").lower()

            for pid in project_ids:
                pid_lower = pid.lower()
                # Check for exact match or word-boundary match (separated by / or _)
                # This prevents "dalia" from matching "dalia_hannah"
                if (
                    f"/{pid_lower}/" in f"/{src}/"
                    or f"/{pid_lower}/" in f"/{dst}/"
                    or src == pid_lower
                    or dst == pid_lower
                ):
                    project_id = pid
                    break

            # Strategy 2: Date-based matching (for daily summaries with no path info)
            if not project_id:
                rec_ts = record.get("timestamp") or record.get("timestamp_str")
                if rec_ts:
                    try:
                        # Parse record timestamp
                        if isinstance(rec_ts, str):
                            rec_dt = datetime.fromisoformat(rec_ts.replace("Z", ""))
                        else:
                            rec_dt = rec_ts
                        if rec_dt.tzinfo:
                            rec_dt = rec_dt.replace(tzinfo=None)

                        # For daily summaries (timestamp at midnight), use DATE comparison only
                        # This handles projects that start/end mid-day
                        rec_date = rec_dt.date()

                        # Check if timestamp falls within any project's date range
                        for p in projects:
                            pid = p.get("projectId")
                            if not pid:
                                continue

                            started = p.get("startedAt")
                            finished = p.get("finishedAt")

                            if not started:
                                continue

                            # Parse project dates (compare dates only, not times)
                            start_dt = datetime.fromisoformat(started.replace("Z", ""))
                            if start_dt.tzinfo:
                                start_dt = start_dt.replace(tzinfo=None)
                            start_date = start_dt.date()

                            # Check if in range (DATE level, not datetime)
                            if rec_date < start_date:
                                continue

                            if finished:
                                end_dt = datetime.fromisoformat(
                                    finished.replace("Z", "")
                                )
                                if end_dt.tzinfo:
                                    end_dt = end_dt.replace(tzinfo=None)
                                end_date = end_dt.date()
                                if rec_date > end_date:
                                    continue

                            # Match found!
                            project_id = pid
                            break
                    except Exception:
                        pass

            if not project_id:
                continue  # Skip operations not matching any project

            # Get project title for display
            project_title = next(
                (
                    p.get("title") or p.get("projectId")
                    for p in projects
                    if p.get("projectId") == project_id
                ),
                project_id,
            )

            # Count PNG files only (not companion files)
            files_list = record.get("files", [])
            png_count = sum(
                1
                for f in files_list
                if isinstance(f, str) and f.lower().endswith(".png")
            )

            # Fallback to file_count if no files list (for daily summary records)
            if not files_list and record.get("file_count"):
                png_count = record.get("file_count", 0)

            if png_count is None:
                png_count = 0

            aggregated[time_key][project_title] += png_count

        # Convert to list format
        result = []
        for time_key, projects_data in aggregated.items():
            for project_name, value in projects_data.items():
                result.append(
                    {
                        "time_slice": time_key,
                        "project": project_name,
                        "file_count": value,
                    }
                )

        return sorted(result, key=lambda x: x["time_slice"])

    def _get_time_slice_key(self, record: Dict, time_slice: str) -> Optional[str]:
        """Get time slice key for a record"""
        # Determine which timestamp to use
        timestamp = None
        if "timestamp" in record and record["timestamp"]:
            timestamp = record["timestamp"]
        elif "start_datetime" in record and record["start_datetime"]:
            timestamp = record["start_datetime"]

        # Convert string timestamps to datetime
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                # If can't parse, and it's daily slice, try using date field directly
                if time_slice == "D" and "date" in record:
                    return str(record["date"])
                return None

        if not timestamp:
            # Fallback to date field
            if "date" in record:
                date_val = record["date"]
                if time_slice == "D":
                    return str(date_val)
                # Try to parse date to datetime for other slices
                try:
                    if isinstance(date_val, str):
                        timestamp = datetime.fromisoformat(date_val)
                    else:
                        timestamp = datetime.combine(date_val, datetime.min.time())
                except Exception:
                    return None
            else:
                return None

        # Generate time slice key
        if time_slice == "15min":
            # Round to 15-minute intervals
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0).isoformat()
        elif time_slice == "1H":
            # Round to hour
            return timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
        elif time_slice == "D":
            # Daily
            return timestamp.date().isoformat()
        elif time_slice == "W":
            # Weekly (Monday as start of week)
            days_since_monday = timestamp.weekday()
            monday = timestamp - timedelta(days=days_since_monday)
            return monday.date().isoformat()
        elif time_slice == "M":
            # Monthly
            return f"{timestamp.year}-{timestamp.month:02d}-01"

        return None

    def calculate_historical_averages(
        self,
        records: List[Dict],
        time_slice: str,
        value_field: str,
        group_field: str = "script",
        lookback_days: int = 30,
    ) -> List[Dict]:
        """Calculate historical averages for 'cloud' overlay backgrounds"""
        if not records:
            return []

        # Filter to historical data (exclude recent data)
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        historical_records = []

        for record in records:
            timestamp = None
            if "timestamp" in record and record["timestamp"]:
                timestamp = record["timestamp"]
            elif "start_datetime" in record and record["start_datetime"]:
                timestamp = record["start_datetime"]

            # Normalize tz-aware to naive for comparison
            if timestamp:
                try:
                    if getattr(timestamp, "tzinfo", None) is not None:
                        timestamp = timestamp.replace(tzinfo=None)
                except Exception:
                    pass
            if timestamp and timestamp < cutoff_date:
                historical_records.append(record)

        if not historical_records:
            return []

        # Aggregate historical data
        aggregated = self.aggregate_by_time_slice(
            historical_records, time_slice, value_field, group_field
        )

        # Calculate averages by time pattern
        pattern_groups = defaultdict(lambda: defaultdict(list))

        for record in aggregated:
            time_pattern = self._get_time_pattern(record["time_slice"], time_slice)
            group_key = record[group_field]
            value = record[value_field]

            pattern_groups[time_pattern][group_key].append(value)

        # Calculate averages
        averages = []
        for time_pattern, groups in pattern_groups.items():
            for group_key, values in groups.items():
                avg_value = sum(values) / len(values) if values else 0
                averages.append(
                    {
                        "time_pattern": time_pattern,
                        group_field: group_key,
                        f"{value_field}_avg": avg_value,
                    }
                )

        return averages

    def _get_time_pattern(self, time_slice_key: str, time_slice: str) -> str:
        """Get time pattern for historical averaging"""
        try:
            if time_slice in ["15min", "1H"]:
                # Intraday: pattern by hour of day
                dt = datetime.fromisoformat(time_slice_key)
                return str(dt.hour)
            elif time_slice == "D":
                # Daily: pattern by day of week
                dt = datetime.fromisoformat(time_slice_key)
                return str(dt.weekday())
            elif time_slice == "W":
                # Weekly: pattern by week of year (simplified)
                dt = datetime.fromisoformat(time_slice_key)
                return str(dt.isocalendar()[1])
            else:
                # Monthly: pattern by month
                dt = datetime.fromisoformat(time_slice_key)
                return str(dt.month)
        except Exception:
            return "unknown"

    def load_script_updates(self) -> List[Dict]:
        """Load script update tracking data"""
        if not self.script_updates_file.exists():
            # Create empty file with headers
            with open(self.script_updates_file, "w") as f:
                f.write("date,script,description\n")
            return []

        try:
            updates = []
            with open(self.script_updates_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.strip().split(",", 2)
                        if len(parts) >= 3:
                            updates.append(
                                {
                                    "date": parts[0],
                                    "script": parts[1],
                                    "description": parts[2],
                                }
                            )
            return updates
        except Exception as e:
            print(f"Warning: Could not load script updates: {e}")
            return []

    def add_script_update(
        self, script: str, description: str, date: Optional[str] = None
    ):
        """Add a script update entry"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Append to CSV
        with open(self.script_updates_file, "a") as f:
            f.write(f"{date},{script},{description}\n")

        print(f"Added script update: {script} - {description}")

    def generate_dashboard_data(
        self,
        time_slice: str = "D",
        lookback_days: int = 30,
        production_scripts: List[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete dashboard data package

        Returns:
            Dictionary with all data needed for dashboard visualization
        """
        import time as time_module

        overall_start = time_module.time()
        print(f"\n{'=' * 70}")
        print("[DATA_ENGINE TIMING] Starting generate_dashboard_data")
        print(f"  time_slice: {time_slice}, lookback_days: {lookback_days}")
        print(f"{'=' * 70}")

        # Define production workflow scripts for dashboard filtering
        # Map current script names to log names
        if production_scripts is None:
            production_scripts = [
                # Current script names
                "01_web_image_selector",
                "03_web_character_sorter",
                "04_multi_crop_tool",
                "multi_crop_tool",
                "ai_desktop_multi_crop",
                "ai_desktop_multi_crop_queue",
                "crop_queue_processor",
                "ai_assisted_reviewer",
                # Log script names (from actual data)
                "character_sorter",
                "image_version_selector",
                "multi_batch_crop_tool",
                # Legacy names
                "batch_crop_tool",
            ]

        # Load raw data
        step_start = time_module.time()
        activity_records = self.load_session_data(lookback_days=lookback_days)
        step_time = time_module.time() - step_start
        print(
            f"[DATA_ENGINE] ✓ load_session_data: {step_time:.3f}s ({len(activity_records)} records)"
        )

        step_start = time_module.time()
        file_ops_records = self.load_file_operations()
        step_time = time_module.time() - step_start
        print(
            f"[DATA_ENGINE] ✓ load_file_operations: {step_time:.3f}s ({len(file_ops_records)} records)"
        )

        step_start = time_module.time()
        script_updates = self.load_script_updates()
        projects = self.load_projects()
        step_time = time_module.time() - step_start
        print(
            f"[DATA_ENGINE] ✓ load_script_updates + load_projects: {step_time:.3f}s ({len(projects)} projects)"
        )

        # Aggregate project metrics (cached internally)
        step_start = time_module.time()
        try:
            proj_agg = ProjectMetricsAggregator(self.data_dir)
            project_metrics = proj_agg.aggregate()
        except Exception:
            project_metrics = {}
        step_time = time_module.time() - step_start
        print(f"[DATA_ENGINE] ✓ ProjectMetricsAggregator.aggregate: {step_time:.3f}s")

        # Filter to production scripts only for dashboard
        activity_records = [
            r for r in activity_records if r.get("script") in production_scripts
        ]
        file_ops_records = [
            r for r in file_ops_records if r.get("script") in production_scripts
        ]

        # Note: Do not filter raw records by lookback here to preserve backward-compatible
        # expectations in tests. The UI/aggregation layer handles time slicing.

        # Optional: filter by project using simple path heuristic
        if project_id:
            project = next(
                (p for p in projects if p.get("projectId") == project_id), None
            )
            root_hint = None
            if project:
                root_hint = project.get("paths", {}).get("root")
            if root_hint:

                def belongs(rec: Dict[str, Any]) -> bool:
                    src = rec.get("source_dir") or ""
                    dst = rec.get("dest_dir") or ""
                    wd = rec.get("working_dir") or ""
                    return (
                        (str(root_hint) in src)
                        or (str(root_hint) in dst)
                        or (str(root_hint) in wd)
                    )

                file_ops_records = [r for r in file_ops_records if belongs(r)]

        # Discover all scripts
        scripts = self.discover_scripts()

        # Calculate date ranges (normalize to strings)
        activity_dates = []
        for r in activity_records:
            date_val = r.get("date")
            if date_val:
                # Normalize to string format YYYYMMDD
                if hasattr(date_val, "strftime"):
                    activity_dates.append(date_val.strftime("%Y%m%d"))
                else:
                    activity_dates.append(str(date_val).replace("-", ""))

        file_ops_dates = []
        for r in file_ops_records:
            date_val = r.get("date")
            if date_val:
                # Normalize to string format YYYYMMDD
                if hasattr(date_val, "strftime"):
                    file_ops_dates.append(date_val.strftime("%Y%m%d"))
                else:
                    file_ops_dates.append(str(date_val).replace("-", ""))

        # Filter out None values before min/max to avoid comparison errors
        activity_dates_clean = [d for d in activity_dates if d is not None]
        file_ops_dates_clean = [d for d in file_ops_dates if d is not None]

        dashboard_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "time_slice": time_slice,
                "lookback_days": lookback_days,
                "scripts_found": scripts,
                "projects_found": [p.get("projectId") for p in projects],
                "active_project": project_id,
                "data_range": {
                    "activity_start": (
                        min(activity_dates_clean) if activity_dates_clean else None
                    ),
                    "activity_end": (
                        max(activity_dates_clean) if activity_dates_clean else None
                    ),
                    "file_ops_start": (
                        min(file_ops_dates_clean) if file_ops_dates_clean else None
                    ),
                    "file_ops_end": (
                        max(file_ops_dates_clean) if file_ops_dates_clean else None
                    ),
                },
                "baseline_labels": {
                    "15min": self.build_time_labels("15min", lookback_days),
                    "1H": self.build_time_labels("1H", lookback_days),
                    "D": self.build_time_labels("D", lookback_days),
                    "W": self.build_time_labels("W", lookback_days),
                    "M": self.build_time_labels("M", lookback_days),
                },
            },
            "activity_data": {},
            "file_operations_data": {},
            "historical_averages": {},
            "script_updates": script_updates,
            "projects": projects,
            "project_markers": {},
            "project_metrics": project_metrics,
            "project_kpi": {},
            "timing_data": {},
        }

        # Add project markers if applicable
        if project_id:
            pj = next((p for p in projects if p.get("projectId") == project_id), None)
            if pj:
                dashboard_data["project_markers"] = {
                    "startedAt": pj.get("startedAt"),
                    "finishedAt": pj.get("finishedAt"),
                }
                # Compute KPI for selected project (images/hour)
                pm = project_metrics.get(project_id)
                if pm:
                    dashboard_data["project_kpi"] = {
                        "projectId": project_id,
                        "images_per_hour": pm.get("throughput", {}).get(
                            "images_per_hour"
                        ),
                        "images_processed": pm.get("totals", {}).get(
                            "images_processed"
                        ),
                    }

        # Process activity data
        if activity_records:
            # Active time by script
            dashboard_data["activity_data"]["active_time"] = (
                self.aggregate_by_time_slice(
                    activity_records, time_slice, "active_time", "script"
                )
            )

            # Files processed by script
            dashboard_data["activity_data"]["files_processed"] = (
                self.aggregate_by_time_slice(
                    activity_records, time_slice, "files_processed", "script"
                )
            )

            # Efficiency by script
            dashboard_data["activity_data"]["efficiency"] = (
                self.aggregate_by_time_slice(
                    activity_records, time_slice, "efficiency", "script"
                )
            )

            # Historical averages
            dashboard_data["historical_averages"]["active_time"] = (
                self.calculate_historical_averages(
                    activity_records, time_slice, "active_time", "script", lookback_days
                )
            )

            dashboard_data["historical_averages"]["files_processed"] = (
                self.calculate_historical_averages(
                    activity_records,
                    time_slice,
                    "files_processed",
                    "script",
                    lookback_days,
                )
            )

        # Process file operations data
        if file_ops_records:
            # File operations by script
            dashboard_data["file_operations_data"]["by_script"] = (
                self.aggregate_by_time_slice(
                    file_ops_records, time_slice, "file_count", "script"
                )
            )

            # File operations by type
            dashboard_data["file_operations_data"]["by_operation"] = (
                self.aggregate_by_time_slice(
                    file_ops_records, time_slice, "file_count", "operation"
                )
            )

            # File operations by project (NEW - for project comparison chart)
            dashboard_data["file_operations_data"]["by_project"] = (
                self._aggregate_by_project(file_ops_records, time_slice, projects)
            )

            # PNG-only image metrics (best-effort):
            # Count images kept (moves to selected/crop) and images deleted (send_to_trash/delete)
            # Only include records that explicitly indicate image-only counting via notes or where
            # we can reasonably infer PNG-only from the files list.
            def _is_png_only(rec: Dict[str, Any]) -> bool:
                # Some logs may carry notes as non-strings (lists/objects) —
                # coerce safely before calling lower() to avoid 500s.
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
                # If no explicit signal, be conservative and do not assume
                return False

            png_only_records: List[Dict[str, Any]] = [
                r for r in file_ops_records if _is_png_only(r)
            ]
            if png_only_records:
                # images kept (PNG) by script
                kept_png = [
                    r
                    for r in png_only_records
                    if (
                        str(r.get("operation")).lower() == "move"
                        and str(r.get("dest_dir") or "").lower()
                        in {
                            "selected",
                            "__selected",
                            "crop",
                            "__crop",
                            "__crop_auto",
                            "crop_auto",
                        }
                    )
                ]
                # images deleted (PNG) by script
                deleted_png = [
                    r
                    for r in png_only_records
                    if str(r.get("operation")).lower() in {"delete", "send_to_trash"}
                ]

                dashboard_data["file_operations_data"]["images_kept_png"] = (
                    self.aggregate_by_time_slice(
                        kept_png, time_slice, "file_count", "script"
                    )
                )
                dashboard_data["file_operations_data"]["images_deleted_png"] = (
                    self.aggregate_by_time_slice(
                        deleted_png, time_slice, "file_count", "script"
                    )
                )

            # Deleted files specifically (for granular tracking)
            delete_ops = [
                r
                for r in file_ops_records
                if r.get("operation") in ["delete", "send_to_trash"]
            ]
            if delete_ops:
                dashboard_data["file_operations_data"]["deletions"] = (
                    self.aggregate_by_time_slice(
                        delete_ops, time_slice, "file_count", "script"
                    )
                )

        # Compute timing_data per display script for summary cards
        # Prefer file-operation timing when available for a script; otherwise use activity timer sums
        timing_by_display: Dict[str, Dict[str, Any]] = {}

        # Group file ops by display script
        if file_ops_records:
            ops_by_display: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in file_ops_records:
                display = self.get_display_name(r.get("script"))
                ops_by_display[display].append(r)
            for display, ops in ops_by_display.items():
                metrics = self.calculate_file_operation_work_time(ops)
                timing_by_display[display] = metrics

        # Group activity by display script for scripts without file ops
        if activity_records:
            tmp_by_display: Dict[str, Dict[str, float]] = defaultdict(
                lambda: {"work_time_seconds": 0.0, "files_processed": 0.0}
            )
            for r in activity_records:
                display = self.get_display_name(r.get("script"))
                if display in timing_by_display:
                    continue
                tmp_by_display[display]["work_time_seconds"] += float(
                    r.get("active_time", 0) or 0
                )
                tmp_by_display[display]["files_processed"] += float(
                    r.get("files_processed", 0) or 0
                )
            for display, vals in tmp_by_display.items():
                secs = float(vals["work_time_seconds"])
                timing_by_display[display] = {
                    "work_time_seconds": secs,
                    "work_time_minutes": secs / 60.0,
                    "total_operations": 0,
                    "files_processed": int(vals["files_processed"]),
                    "efficiency_score": (
                        (int(vals["files_processed"]) / (secs / 60.0))
                        if secs > 0
                        else 0.0
                    ),
                    "timing_method": "activity_timer",
                }

        dashboard_data["timing_data"] = timing_by_display

        # Compute artifact stats from operation events extra fields if present
        try:
            events = self.loader.load_operation_events(lookback_days)
            artifact_count = 0
            for evt in events:
                extra = evt.get("extra") or {}
                if extra.get("artifact"):
                    artifact_count += 1
            dashboard_data["artifact_stats"] = {"artifact_events": artifact_count}
        except Exception:
            # Best effort; do not break dashboard
            dashboard_data["artifact_stats"] = {"artifact_events": 0}

        # Add backup status monitoring
        try:
            dashboard_data["backup_status"] = self.get_backup_status()
        except Exception:
            # Best effort; do not break dashboard
            dashboard_data["backup_status"] = {
                "status": "error",
                "message": "Backup status unavailable",
            }

        overall_time = time_module.time() - overall_start
        print(f"\n{'=' * 70}")
        print(f"[DATA_ENGINE] TOTAL generate_dashboard_data TIME: {overall_time:.3f}s")
        print(f"{'=' * 70}\n")

        return dashboard_data

    def get_backup_status(self) -> Dict[str, Any]:
        """
        Get backup system status for dashboard monitoring.

        Returns:
            Dict with backup status information
        """
        backup_status = {
            "status": "unknown",
            "last_backup": None,
            "last_backup_timestamp": None,
            "total_files": 0,
            "total_size_mb": 0,
            "failures": [],
            "message": "Backup status unavailable",
        }

        try:
            # Check for backup status file
            backup_root = Path.home() / "project-data-archives" / "image-workflow"
            status_file = backup_root / "backup_status.json"

            if status_file.exists():
                with open(status_file, "r") as f:
                    status_data = json.load(f)

                backup_status.update(status_data)

                # Determine status message
                if backup_status["status"] == "success":
                    backup_status["message"] = (
                        f"✅ Last backup successful: {backup_status['last_backup']} ({backup_status['total_files']} files, {backup_status['total_size_mb']} MB)"
                    )
                elif backup_status["status"] == "failed":
                    failures = backup_status.get("failures", [])
                    backup_status["message"] = (
                        f"❌ Last backup failed: {backup_status['last_backup']} ({len(failures)} failures)"
                    )
                else:
                    backup_status["message"] = "⚠️ Backup status unknown"

                # Check if backup is recent (within 48 hours)
                if backup_status["last_backup_timestamp"]:
                    last_backup_time = datetime.fromisoformat(
                        backup_status["last_backup_timestamp"]
                    )
                    hours_since_backup = (
                        datetime.now() - last_backup_time
                    ).total_seconds() / 3600

                    if hours_since_backup > 48:
                        backup_status["message"] = (
                            f"⚠️ Backup overdue: Last successful backup {hours_since_backup:.1f} hours ago"
                        )
                        backup_status["status"] = "overdue"
            else:
                backup_status["message"] = "⚠️ No backup status file found"

        except Exception as e:
            backup_status["message"] = f"❌ Error reading backup status: {e}"
            backup_status["status"] = "error"

        return backup_status


def main():
    """Test the data engine"""
    engine = DashboardDataEngine()

    print("🔍 Discovering scripts...")
    scripts = engine.discover_scripts()
    print(f"Found scripts: {scripts}")

    print("\n📊 Loading activity data...")
    activity_data = engine.load_activity_data()
    print(f"Activity records: {len(activity_data)}")

    print("\n📁 Loading file operations...")
    file_ops_data = engine.load_file_operations()
    print(f"File operation records: {len(file_ops_data)}")

    print("\n📈 Generating dashboard data...")
    data = engine.generate_dashboard_data(time_slice="D", lookback_days=7)

    print(f"\nMetadata: {data['metadata']}")
    print(f"Activity data keys: {list(data['activity_data'].keys())}")
    print(f"File operations keys: {list(data['file_operations_data'].keys())}")
    print(f"Script updates: {len(data['script_updates'])}")

    # Save sample output
    output_file = Path("dashboard_data_sample.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n💾 Sample data saved to: {output_file}")


if __name__ == "__main__":
    main()
