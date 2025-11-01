#!/usr/bin/env python3
"""
Bins Reader
===========
Helper module for reading 15-minute pre-aggregated bins.

Used by data_engine when performance_mode is enabled.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class BinsReader:
    """Reader for 15-minute aggregated bins."""

    def __init__(self, data_dir: Path, config: dict[str, Any]):
        self.data_dir = data_dir
        self.config = config
        self.aggregates_dir = data_dir / "aggregates"
        self.daily_dir = self.aggregates_dir / "daily"
        self.overall_path = self.aggregates_dir / "overall" / "agg_15m_cumulative.jsonl"

    def load_config(self) -> dict[str, Any]:
        """Load bins configuration."""
        config_path = self.data_dir.parent / "configs" / "bins_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "enabled": False,
            "performance_mode": {"use_15m_bins": False, "bin_charts": []},
        }

    def is_enabled(self) -> bool:
        """Check if bins system is enabled."""
        return self.config.get("enabled", False) and self.config.get(
            "performance_mode", {}
        ).get("use_15m_bins", False)

    def should_use_bins_for_chart(self, chart_name: str) -> bool:
        """Check if specific chart should use bins."""
        if not self.is_enabled():
            return False
        bin_charts = self.config.get("performance_mode", {}).get("bin_charts", [])
        return chart_name in bin_charts

    def load_bins_for_range(
        self, start_date: datetime, end_date: datetime, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Load bins for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            project_id: Optional project filter

        Returns:
            List of bin records
        """
        bins = []
        current_date = start_date.date()
        end = end_date.date()

        while current_date <= end:
            day_str = current_date.strftime("%Y%m%d")
            bin_path = self.daily_dir / f"day={day_str}" / "agg_15m.jsonl"

            if bin_path.exists():
                try:
                    with open(bin_path) as f:
                        for line in f:
                            try:
                                bin_record = json.loads(line)

                                # Filter by project if specified
                                if (
                                    project_id
                                    and bin_record.get("project_id") != project_id
                                ):
                                    continue

                                bins.append(bin_record)
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"Warning: Error reading {bin_path}: {e}")

            current_date += timedelta(days=1)

        return bins

    def aggregate_bins_to_time_slice(
        self, bins: list[dict[str, Any]], time_slice: str, groupby: str
    ) -> list[dict[str, Any]]:
        """Aggregate bins to a specific time slice and grouping.

        Args:
            bins: List of 15-minute bin records
            time_slice: Target time slice ('15min', '1H', 'D', 'W', 'M')
            groupby: Grouping field ('script_id', 'operation', 'project_id')

        Returns:
            List of aggregated records matching data_engine format
        """
        # If time_slice is 15min, bins are already at right granularity
        if time_slice == "15min":
            return self._format_bins_for_output(bins, groupby)

        # Otherwise, re-aggregate to target time slice
        aggregated = defaultdict(
            lambda: defaultdict(
                lambda: {"file_count": 0, "event_count": 0, "work_seconds": 0.0}
            )
        )

        for bin_record in bins:
            # Parse bin timestamp
            bin_ts_str = bin_record.get("bin_ts_utc", "")
            try:
                bin_ts = datetime.fromisoformat(bin_ts_str.replace("Z", "+00:00"))
            except Exception:
                continue

            # Get time slice key
            ts_key = self._get_time_slice_key(bin_ts, time_slice)
            if not ts_key:
                continue

            # Get grouping key
            group_key = bin_record.get(groupby, "unknown")

            # Accumulate
            aggregated[ts_key][group_key]["file_count"] += bin_record.get(
                "file_count", 0
            )
            aggregated[ts_key][group_key]["event_count"] += bin_record.get(
                "event_count", 0
            )
            aggregated[ts_key][group_key]["work_seconds"] += bin_record.get(
                "work_seconds", 0.0
            )

        # Convert to output format
        output = []
        for ts_key, groups in aggregated.items():
            for group_key, values in groups.items():
                output.append(
                    {
                        "time_slice": ts_key,
                        groupby: group_key,
                        "file_count": values["file_count"],
                        "event_count": values["event_count"],
                        "work_time_minutes": round(values["work_seconds"] / 60.0, 2),
                    }
                )

        return output

    def _format_bins_for_output(
        self, bins: list[dict[str, Any]], groupby: str
    ) -> list[dict[str, Any]]:
        """Format bins for output (15min granularity)."""
        output = []
        for bin_record in bins:
            # Get time slice key (bin timestamp)
            bin_ts_str = bin_record.get("bin_ts_utc", "")
            try:
                bin_ts = datetime.fromisoformat(bin_ts_str.replace("Z", "+00:00"))
                ts_key = bin_ts.isoformat()
            except Exception:
                ts_key = bin_ts_str

            # Get grouping key
            group_key = bin_record.get(groupby, "unknown")

            output.append(
                {
                    "time_slice": ts_key,
                    groupby: group_key,
                    "file_count": bin_record.get("file_count", 0),
                    "event_count": bin_record.get("event_count", 0),
                    "work_time_minutes": round(
                        bin_record.get("work_seconds", 0.0) / 60.0, 2
                    ),
                }
            )

        return output

    def _get_time_slice_key(self, dt: datetime, time_slice: str) -> str:
        """Get time slice key for a datetime (matches data_engine logic)."""
        if time_slice == "15min":
            # Floor to 15-minute boundary
            minute = (dt.minute // 15) * 15
            aligned = dt.replace(minute=minute, second=0, microsecond=0)
            return aligned.isoformat()

        if time_slice == "1H":
            aligned = dt.replace(minute=0, second=0, microsecond=0)
            return aligned.isoformat()

        if time_slice == "D":
            return dt.date().isoformat()

        if time_slice == "W":
            # Monday of the week
            monday = dt.date() - timedelta(days=dt.weekday())
            return monday.isoformat()

        if time_slice == "M":
            return dt.date().replace(day=1).isoformat()

        return ""

    def load_by_script(
        self,
        start_date: datetime,
        end_date: datetime,
        time_slice: str,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load file operations by script from bins.

        Returns:
            List of records with format:
            {'time_slice': '...', 'script': '...', 'file_count': N, ...}
        """
        bins = self.load_bins_for_range(start_date, end_date, project_id)
        return self.aggregate_bins_to_time_slice(bins, time_slice, "script_id")

    def load_by_operation(
        self,
        start_date: datetime,
        end_date: datetime,
        time_slice: str,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load file operations by operation type from bins.

        Returns:
            List of records with format:
            {'time_slice': '...', 'operation': '...', 'file_count': N, ...}
        """
        bins = self.load_bins_for_range(start_date, end_date, project_id)
        return self.aggregate_bins_to_time_slice(bins, time_slice, "operation")

    def load_by_project(
        self, start_date: datetime, end_date: datetime, time_slice: str
    ) -> list[dict[str, Any]]:
        """Load file operations by project from bins.

        Returns:
            List of records with format:
            {'time_slice': '...', 'project': '...', 'file_count': N, ...}
        """
        bins = self.load_bins_for_range(start_date, end_date)
        return self.aggregate_bins_to_time_slice(bins, time_slice, "project_id")


def main():
    """Test bins reader."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    # Load config
    config_path = project_root / "configs" / "bins_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        print("Config not found")
        sys.exit(1)

    reader = BinsReader(data_dir, config)

    print(f"Bins enabled: {reader.is_enabled()}")
    print(f"Daily dir: {reader.daily_dir}")
    print(f"Overall path: {reader.overall_path}")

    # Test load last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print(f"\nLoading bins from {start_date.date()} to {end_date.date()}...")
    bins = reader.load_bins_for_range(start_date, end_date)
    print(f"Loaded {len(bins)} bins")

    if bins:
        print("\nSample bin:")
        print(json.dumps(bins[0], indent=2))

        # Test aggregation
        print("\nAggregating by script (daily)...")
        by_script = reader.load_by_script(start_date, end_date, "D")
        print(f"Result: {len(by_script)} records")
        if by_script:
            print(json.dumps(by_script[0], indent=2))


if __name__ == "__main__":
    main()
