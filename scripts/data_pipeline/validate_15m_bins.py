#!/usr/bin/env python3
"""
15-Minute Bin Validator
=======================
Validates aggregated 15-minute bins against raw logs to ensure correctness.

Validation checks:
1. Total file counts match (±1% tolerance for rounding)
2. Total event counts match exactly
3. Work seconds match (±1% tolerance)
4. No duplicate dedupe_keys
5. All bin timestamps are valid and 15-minute aligned
6. Min/max timestamps within expected range
7. All required fields present and correct types

Exit codes:
0 = All validations passed
1 = Validation failures detected
2 = Error running validation
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.data_pipeline.aggregate_to_15m import (
    count_png_files,
    load_raw_logs_for_day,
    parse_timestamp_to_utc,
)
from scripts.utils.companion_file_utils import get_file_operation_metrics


class ValidationError:
    """Container for validation errors."""

    def __init__(self, check_name: str, message: str, severity: str = "error"):
        self.check_name = check_name
        self.message = message
        self.severity = severity  # 'error', 'warning'

    def __str__(self):
        icon = "✗" if self.severity == "error" else "⚠"
        return f"  {icon} {self.check_name}: {self.message}"


class BinValidator:
    """Validator for 15-minute bins."""

    def __init__(self, data_dir: Path, day_str: str):
        self.data_dir = data_dir
        self.day_str = day_str
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []

        # Load data
        self.raw_records = load_raw_logs_for_day(data_dir, day_str)
        self.bins = self._load_bins()

    def _load_bins(self) -> list[dict[str, Any]]:
        """Load aggregated bins for the day."""
        bin_path = (
            self.data_dir
            / "aggregates"
            / "daily"
            / f"day={self.day_str}"
            / "agg_15m.jsonl"
        )
        if not bin_path.exists():
            return []

        bins = []
        with open(bin_path) as f:
            for line in f:
                try:
                    bins.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return bins

    def validate_all(self) -> bool:
        """Run all validation checks.

        Returns:
            True if all checks passed
        """
        if not self.raw_records and not self.bins:
            self.warnings.append(
                ValidationError(
                    "data_present",
                    f"No data found for {self.day_str} (neither raw nor bins)",
                    severity="warning",
                )
            )
            return True  # Not an error if no data

        if not self.bins and self.raw_records:
            self.errors.append(
                ValidationError(
                    "bins_missing",
                    f"Raw records exist ({len(self.raw_records)}) but bins not found",
                )
            )
            return False

        if not self.raw_records and self.bins:
            self.errors.append(
                ValidationError(
                    "raw_missing",
                    f"Bins exist ({len(self.bins)}) but raw records not found",
                )
            )
            return False

        # Run all checks
        self.check_schema()
        self.check_dedupe_keys()
        self.check_bin_alignment()
        self.check_totals_match()
        self.check_work_seconds()
        self.check_timestamp_ranges()

        return len(self.errors) == 0

    def check_schema(self):
        """Validate bin schema: all required fields present and correct types."""
        required_fields = {
            "bin_ts_utc": str,
            "bin_version": int,
            "project_id": str,
            "script_id": str,
            "operation": str,
            "dest_category": str,
            "file_count": int,
            "file_count_total": int,
            "event_count": int,
            "work_seconds": (int, float),
            "first_event_ts": str,
            "last_event_ts": str,
            "dedupe_key": str,
            "tz_source": str,
            "created_at": str,
        }

        for idx, bin_record in enumerate(self.bins):
            for field, expected_type in required_fields.items():
                if field not in bin_record:
                    self.errors.append(
                        ValidationError(
                            "schema", f'Bin {idx}: Missing required field "{field}"'
                        )
                    )
                    continue

                value = bin_record[field]
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        self.errors.append(
                            ValidationError(
                                "schema",
                                f'Bin {idx}: Field "{field}" has wrong type (expected {expected_type}, got {type(value).__name__})',
                            )
                        )
                elif not isinstance(value, expected_type):
                    self.errors.append(
                        ValidationError(
                            "schema",
                            f'Bin {idx}: Field "{field}" has wrong type (expected {expected_type.__name__}, got {type(value).__name__})',
                        )
                    )

    def check_dedupe_keys(self):
        """Check for duplicate dedupe_keys."""
        seen_keys: set[str] = set()
        duplicates = []

        for bin_record in self.bins:
            key = bin_record.get("dedupe_key", "")
            if key in seen_keys:
                duplicates.append(key)
            seen_keys.add(key)

        if duplicates:
            self.errors.append(
                ValidationError(
                    "dedupe_keys",
                    f"Found {len(duplicates)} duplicate dedupe_keys: {duplicates[:5]}",
                )
            )

    def check_bin_alignment(self):
        """Check that all bin timestamps are 15-minute aligned."""
        for idx, bin_record in enumerate(self.bins):
            ts_str = bin_record.get("bin_ts_utc", "")
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if dt.minute not in [0, 15, 30, 45] or dt.second != 0:
                    self.errors.append(
                        ValidationError(
                            "bin_alignment",
                            f"Bin {idx}: Timestamp not 15-minute aligned: {ts_str}",
                        )
                    )
            except ValueError:
                self.errors.append(
                    ValidationError(
                        "bin_alignment",
                        f"Bin {idx}: Invalid timestamp format: {ts_str}",
                    )
                )

    def check_totals_match(self):
        """Check that total file counts and event counts match between raw and bins."""
        # Sum from raw records
        raw_png_total = 0
        raw_file_total = 0
        raw_event_total = len(self.raw_records)

        for record in self.raw_records:
            png_count, total_count = count_png_files(record)
            raw_png_total += png_count
            raw_file_total += total_count

        # Sum from bins
        bin_png_total = sum(b.get("file_count", 0) for b in self.bins)
        bin_file_total = sum(b.get("file_count_total", 0) for b in self.bins)
        bin_event_total = sum(b.get("event_count", 0) for b in self.bins)

        # Check event count (should match exactly)
        if bin_event_total != raw_event_total:
            self.errors.append(
                ValidationError(
                    "event_count",
                    f"Event count mismatch: raw={raw_event_total}, bins={bin_event_total}",
                )
            )

        # Check PNG file count (±1% tolerance)
        if raw_png_total > 0:
            png_diff_pct = abs(bin_png_total - raw_png_total) / raw_png_total * 100
            if png_diff_pct > 1.0:
                self.errors.append(
                    ValidationError(
                        "file_count",
                        f"PNG file count diff > 1%: raw={raw_png_total}, bins={bin_png_total} ({png_diff_pct:.2f}%)",
                    )
                )
        elif bin_png_total != raw_png_total:
            self.errors.append(
                ValidationError(
                    "file_count",
                    f"PNG file count mismatch: raw={raw_png_total}, bins={bin_png_total}",
                )
            )

        # Check total file count (±1% tolerance)
        if raw_file_total > 0:
            file_diff_pct = abs(bin_file_total - raw_file_total) / raw_file_total * 100
            if file_diff_pct > 1.0:
                self.errors.append(
                    ValidationError(
                        "file_count_total",
                        f"Total file count diff > 1%: raw={raw_file_total}, bins={bin_file_total} ({file_diff_pct:.2f}%)",
                    )
                )
        elif bin_file_total != raw_file_total:
            self.errors.append(
                ValidationError(
                    "file_count_total",
                    f"Total file count mismatch: raw={raw_file_total}, bins={bin_file_total}",
                )
            )

    def check_work_seconds(self):
        """Check that work_seconds calculation matches (±1% tolerance)."""
        # Calculate work_seconds from raw records
        try:
            events_for_metrics = []
            for event in self.raw_records:
                event_copy = dict(event)
                ts = event_copy.get("timestamp")
                if isinstance(ts, datetime):
                    event_copy["timestamp"] = ts.isoformat()
                elif not isinstance(ts, str):
                    ts_str = event_copy.get("timestamp_str")
                    if isinstance(ts_str, str):
                        event_copy["timestamp"] = ts_str
                events_for_metrics.append(event_copy)

            metrics = get_file_operation_metrics(events_for_metrics)
            raw_work_seconds = float(metrics.get("work_time_minutes", 0) or 0) * 60.0
        except Exception as e:
            self.warnings.append(
                ValidationError(
                    "work_seconds",
                    f"Could not calculate raw work_seconds: {e}",
                    severity="warning",
                )
            )
            return

        # Sum work_seconds from bins
        bin_work_seconds = sum(b.get("work_seconds", 0) for b in self.bins)

        # Check with ±1% tolerance
        if raw_work_seconds > 0:
            work_diff_pct = (
                abs(bin_work_seconds - raw_work_seconds) / raw_work_seconds * 100
            )
            if work_diff_pct > 1.0:
                self.errors.append(
                    ValidationError(
                        "work_seconds",
                        f"Work seconds diff > 1%: raw={raw_work_seconds:.2f}, bins={bin_work_seconds:.2f} ({work_diff_pct:.2f}%)",
                    )
                )
        elif bin_work_seconds != raw_work_seconds:
            # Allow small absolute differences for zero/near-zero values
            if abs(bin_work_seconds - raw_work_seconds) > 1.0:
                self.errors.append(
                    ValidationError(
                        "work_seconds",
                        f"Work seconds mismatch: raw={raw_work_seconds:.2f}, bins={bin_work_seconds:.2f}",
                    )
                )

    def check_timestamp_ranges(self):
        """Check that first/last event timestamps are within bin boundaries."""
        for idx, bin_record in enumerate(self.bins):
            bin_ts_str = bin_record.get("bin_ts_utc", "")
            first_ts_str = bin_record.get("first_event_ts", "")
            last_ts_str = bin_record.get("last_event_ts", "")

            try:
                bin_ts = datetime.fromisoformat(bin_ts_str.replace("Z", "+00:00"))
                first_ts = parse_timestamp_to_utc(first_ts_str)
                last_ts = parse_timestamp_to_utc(last_ts_str)

                if not first_ts or not last_ts:
                    self.warnings.append(
                        ValidationError(
                            "timestamp_range",
                            f"Bin {idx}: Could not parse event timestamps",
                            severity="warning",
                        )
                    )
                    continue

                # Events should be within 15 minutes of bin start
                from datetime import timedelta

                bin_end = bin_ts + timedelta(minutes=15)

                if first_ts < bin_ts or first_ts >= bin_end:
                    self.warnings.append(
                        ValidationError(
                            "timestamp_range",
                            f"Bin {idx}: first_event_ts outside bin range",
                            severity="warning",
                        )
                    )

                if last_ts < bin_ts or last_ts >= bin_end:
                    self.warnings.append(
                        ValidationError(
                            "timestamp_range",
                            f"Bin {idx}: last_event_ts outside bin range",
                            severity="warning",
                        )
                    )

                if first_ts > last_ts:
                    self.errors.append(
                        ValidationError(
                            "timestamp_range",
                            f"Bin {idx}: first_event_ts > last_event_ts",
                        )
                    )
            except Exception as e:
                self.warnings.append(
                    ValidationError(
                        "timestamp_range",
                        f"Bin {idx}: Error checking timestamps: {e}",
                        severity="warning",
                    )
                )

    def print_report(self):
        """Print validation report."""
        print(f"\n{'='*70}")
        print(f"Validation Report: {self.day_str}")
        print(f"{'='*70}")

        print("\nData Summary:")
        print(f"  Raw records:  {len(self.raw_records)}")
        print(f"  Bins:         {len(self.bins)}")

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(error)

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(warning)

        if not self.errors and not self.warnings:
            print("\n✅ All validation checks passed!")
        elif not self.errors:
            print("\n✅ All critical checks passed (warnings only)")
        else:
            print(f"\n❌ Validation failed with {len(self.errors)} error(s)")

        print(f"\n{'='*70}\n")


def validate_day(data_dir: Path, day_str: str, verbose: bool = False) -> bool:
    """Validate bins for a single day.

    Returns:
        True if validation passed
    """
    validator = BinValidator(data_dir, day_str)
    passed = validator.validate_all()

    if verbose or not passed or validator.warnings:
        validator.print_report()
    elif passed:
        print(f"✓ {day_str}: Validation passed")

    return passed


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate 15-minute bins against raw logs"
    )
    parser.add_argument(
        "--day",
        type=str,
        help="Day to validate (YYYYMMDD format). If not specified, validates last 7 days.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of recent days to validate (default: 7)",
    )
    parser.add_argument(
        "--data-dir", type=str, help="Data directory path (default: project_root/data)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation reports even when passing",
    )

    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "data"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(2)

    print(f"Data directory: {data_dir}\n")

    if args.day:
        # Validate single day
        days = [args.day]
    else:
        # Validate last N days
        from datetime import timedelta

        today = datetime.now().date()
        days = []
        for i in range(args.days):
            day = today - timedelta(days=i)
            days.append(day.strftime("%Y%m%d"))

    # Validate each day
    all_passed = True
    failed_days = []

    for day_str in days:
        try:
            passed = validate_day(data_dir, day_str, verbose=args.verbose)
            if not passed:
                all_passed = False
                failed_days.append(day_str)
        except Exception as e:
            print(f"✗ {day_str}: Error during validation: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False
            failed_days.append(day_str)

    # Summary
    print(f"\n{'='*70}")
    if all_passed:
        print(f"✅ All {len(days)} day(s) passed validation")
        sys.exit(0)
    else:
        print(f"❌ {len(failed_days)} of {len(days)} day(s) failed validation:")
        for day in failed_days:
            print(f"  - {day}")
        sys.exit(1)


if __name__ == "__main__":
    main()
