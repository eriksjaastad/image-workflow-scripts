#!/usr/bin/env python3
"""
Snapshot Data Loader
====================
Loads pre-aggregated snapshot data for dashboard performance mode.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


class SnapshotLoader:
    """Loads normalized snapshot data from v1 format."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.snapshot_dir = self.project_root / "data" / "snapshot"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        config_path = self.project_root / "configs" / "metrics_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        # Defaults
        return {
            "metrics": {
                "session_source": "derived",
                "include_personal_focus_timers": False,
                "gap_min_minutes": 5,
                "max_gap_contrib_seconds": 60,
            },
            "performanceMode": True,
            "lookbackDays": 14,
        }

    def load_derived_sessions(self, lookback_days: int = 14) -> List[Dict[str, Any]]:
        """Load derived sessions from snapshot."""
        sessions_dir = self.snapshot_dir / "derived_sessions_v1"
        if not sessions_dir.exists():
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        sessions = []

        for day_dir in sessions_dir.glob("day=*"):
            day_str = day_dir.name.split("=")[1]
            day_date = datetime.strptime(day_str, "%Y%m%d").replace(tzinfo=timezone.utc)

            if day_date < cutoff:
                continue

            sessions_file = day_dir / "sessions.jsonl"
            if sessions_file.exists():
                with open(sessions_file) as f:
                    for line in f:
                        if line.strip():
                            sessions.append(json.loads(line))

        return sessions

    def load_daily_aggregates(
        self, lookback_days: int = 14
    ) -> Dict[str, Dict[str, Any]]:
        """Load daily aggregates from snapshot."""
        agg_dir = self.snapshot_dir / "daily_aggregates_v1"
        if not agg_dir.exists():
            return {}

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        aggregates = {}

        for day_dir in agg_dir.glob("day=*"):
            day_str = day_dir.name.split("=")[1]
            day_date = datetime.strptime(day_str, "%Y%m%d").replace(tzinfo=timezone.utc)

            if day_date < cutoff:
                continue

            agg_file = day_dir / "aggregate.json"
            if agg_file.exists():
                with open(agg_file) as f:
                    aggregates[day_str] = json.load(f)

        return aggregates

    def get_session_source(self) -> str:
        """Get configured session source."""
        return self.config.get("metrics", {}).get("session_source", "derived")

    def is_performance_mode(self) -> bool:
        """Check if performance mode is enabled."""
        return self.config.get("performanceMode", True)
