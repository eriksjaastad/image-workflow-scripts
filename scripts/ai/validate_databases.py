#!/usr/bin/env python3
"""
Database Validation Script - AI Training Decisions
===================================================
Validates schema integrity, data quality, and constraints for all
ai_training_decisions databases.

Checks:
- Schema correctness (all required columns exist)
- NOT NULL constraints
- Data type validation
- JSON format validation (images, crop_coords)
- Coordinate ranges [0-1]
- Valid user_action values
- CHECK constraints (user_selected_index 0-3, dimensions > 0)
- Statistics and summary

Usage:
    python scripts/ai/validate_databases.py
    python scripts/ai/validate_databases.py --verbose  # Show detailed issues

Exit codes:
    0 = All checks passed
    1 = Warnings found (non-critical)
    2 = Errors found (data integrity issues)
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
DB_DIR = WORKSPACE / "data" / "training" / "ai_training_decisions"

# Expected schema
REQUIRED_COLUMNS = {
    "group_id": "TEXT",
    "timestamp": "TEXT",
    "project_id": "TEXT",
    "images": "TEXT",
    "image_width": "INTEGER",
    "image_height": "INTEGER",
}

OPTIONAL_COLUMNS = {
    "directory": "TEXT",
    "batch_number": "INTEGER",
    "ai_selected_index": "INTEGER",
    "ai_crop_coords": "TEXT",
    "ai_confidence": "REAL",
    "user_selected_index": "INTEGER",
    "user_action": "TEXT",
    "final_crop_coords": "TEXT",
    "crop_timestamp": "TEXT",
    "selection_match": "BOOLEAN",
    "crop_match": "BOOLEAN",
    "ai_crop_accepted": "BOOLEAN",
}

VALID_USER_ACTIONS = {"approve", "crop", "reject"}


class ValidationReport:
    """Collects validation results."""

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.errors = []
        self.warnings = []
        self.stats = {}

    def error(self, msg: str):
        self.errors.append(msg)

    def warning(self, msg: str):
        self.warnings.append(msg)

    def add_stat(self, key: str, value: Any):
        self.stats[key] = value

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


def get_schema(db_path: Path) -> dict[str, str]:
    """Get actual schema from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(ai_decisions)")
    columns = {}
    for row in cursor.fetchall():
        col_name = row[1]
        col_type = row[2]
        columns[col_name] = col_type

    conn.close()
    return columns


def validate_schema(db_path: Path, report: ValidationReport) -> bool:
    """Check if schema has all required columns."""
    try:
        schema = get_schema(db_path)

        # Check required columns
        for col_name, _expected_type in REQUIRED_COLUMNS.items():
            if col_name not in schema:
                report.error(f"Missing required column: {col_name}")
                return False
            # Note: SQLite type affinity is flexible, so we just check existence

        report.add_stat("columns_found", len(schema))
        return True

    except Exception as e:
        report.error(f"Schema validation failed: {e}")
        return False


def validate_records(db_path: Path, report: ValidationReport, verbose: bool = False):
    """Validate all records in database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM ai_decisions")
    records = cursor.fetchall()

    total_records = len(records)
    report.add_stat("total_records", total_records)

    if total_records == 0:
        report.warning("Database is empty (no records)")
        conn.close()
        return

    # Track issues
    issues = {
        "null_required_fields": 0,
        "invalid_json": 0,
        "invalid_coords": 0,
        "invalid_user_action": 0,
        "invalid_dimensions": 0,
        "invalid_selected_index": 0,
    }

    has_ai_predictions = 0
    has_user_data = 0

    for idx, record in enumerate(records):
        record_dict = dict(record)
        group_id = record_dict.get("group_id", f"row_{idx}")

        # Check required NOT NULL fields
        for field in [
            "group_id",
            "timestamp",
            "project_id",
            "images",
            "image_width",
            "image_height",
        ]:
            if record_dict.get(field) is None:
                issues["null_required_fields"] += 1
                if verbose:
                    report.error(f"{group_id}: NULL value in required field '{field}'")

        # Validate JSON fields
        for json_field in ["images", "ai_crop_coords", "final_crop_coords"]:
            value = record_dict.get(json_field)
            if value is not None:
                try:
                    parsed = json.loads(value)

                    # Validate coordinate format
                    if "crop_coords" in json_field and parsed is not None:
                        if not isinstance(parsed, list) or len(parsed) != 4:
                            issues["invalid_coords"] += 1
                            if verbose:
                                report.error(
                                    f"{group_id}: Invalid {json_field} format (expected [x1,y1,x2,y2])"
                                )
                        else:
                            # Check range [0, 1]
                            for coord in parsed:
                                if not isinstance(coord, (int, float)) or not (
                                    0 <= coord <= 1
                                ):
                                    issues["invalid_coords"] += 1
                                    if verbose:
                                        report.error(
                                            f"{group_id}: Coordinate out of range [0,1] in {json_field}: {coord}"
                                        )
                                    break

                except (json.JSONDecodeError, TypeError):
                    issues["invalid_json"] += 1
                    if verbose:
                        report.error(
                            f"{group_id}: Invalid JSON in field '{json_field}': {value[:50]}..."
                        )

        # Validate user_action
        user_action = record_dict.get("user_action")
        if user_action is not None and user_action not in VALID_USER_ACTIONS:
            issues["invalid_user_action"] += 1
            if verbose:
                report.error(
                    f"{group_id}: Invalid user_action '{user_action}' (expected: {VALID_USER_ACTIONS})"
                )

        # Validate dimensions
        width = record_dict.get("image_width")
        height = record_dict.get("image_height")
        if width is not None and width <= 0:
            issues["invalid_dimensions"] += 1
            if verbose:
                report.error(f"{group_id}: Invalid image_width: {width}")
        if height is not None and height <= 0:
            issues["invalid_dimensions"] += 1
            if verbose:
                report.error(f"{group_id}: Invalid image_height: {height}")

        # Validate user_selected_index range (0-3)
        user_idx = record_dict.get("user_selected_index")
        if user_idx is not None and not (0 <= user_idx <= 3):
            issues["invalid_selected_index"] += 1
            if verbose:
                report.error(
                    f"{group_id}: Invalid user_selected_index: {user_idx} (expected 0-3)"
                )

        # Track AI predictions presence
        if record_dict.get("ai_selected_index") is not None:
            has_ai_predictions += 1

        # Track user data presence
        if record_dict.get("user_action") is not None:
            has_user_data += 1

    conn.close()

    # Report issues
    total_issues = sum(issues.values())

    if issues["null_required_fields"] > 0:
        report.error(
            f"Found {issues['null_required_fields']} NULL values in required fields"
        )
    if issues["invalid_json"] > 0:
        report.error(f"Found {issues['invalid_json']} invalid JSON values")
    if issues["invalid_coords"] > 0:
        report.error(f"Found {issues['invalid_coords']} invalid coordinate values")
    if issues["invalid_user_action"] > 0:
        report.error(
            f"Found {issues['invalid_user_action']} invalid user_action values"
        )
    if issues["invalid_dimensions"] > 0:
        report.error(f"Found {issues['invalid_dimensions']} invalid dimension values")
    if issues["invalid_selected_index"] > 0:
        report.error(
            f"Found {issues['invalid_selected_index']} invalid selected_index values"
        )

    # Stats
    report.add_stat("records_with_ai_predictions", has_ai_predictions)
    report.add_stat("records_with_user_data", has_user_data)
    report.add_stat(
        "ai_prediction_coverage", f"{has_ai_predictions/total_records*100:.1f}%"
    )
    report.add_stat("user_data_coverage", f"{has_user_data/total_records*100:.1f}%")

    if total_issues == 0:
        report.add_stat("data_quality", "✅ EXCELLENT")
    elif total_issues < total_records * 0.01:  # < 1% issues
        report.add_stat("data_quality", "⚠️  GOOD (minor issues)")
    else:
        report.add_stat("data_quality", "❌ POOR (many issues)")


def print_report(report: ValidationReport):
    """Print validation report for one database."""
    # Stats
    if report.stats:
        for _key, _value in report.stats.items():
            pass

    # Warnings
    if report.warnings:
        for _warning in report.warnings[:5]:  # Show first 5
            pass
        if len(report.warnings) > 5:
            pass

    # Errors
    if report.errors:
        for _error in report.errors[:10]:  # Show first 10
            pass
        if len(report.errors) > 10:
            pass

    # Status
    if (not report.has_errors() and not report.has_warnings()) or report.has_errors():
        pass
    else:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Validate all AI training decision databases"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed error messages for each issue",
    )
    args = parser.parse_args()

    # Find all databases (exclude backups and temps)
    db_files = sorted(
        [
            f
            for f in DB_DIR.glob("*.db")
            if not f.name.endswith("_temp.db") and "backup" not in f.name
        ]
    )

    if not db_files:
        return 2

    # Validate each database
    all_reports = []
    total_errors = 0
    total_warnings = 0

    for db_path in db_files:
        report = ValidationReport(db_path.name)

        # Validate schema
        if not validate_schema(db_path, report):
            print_report(report)
            total_errors += len(report.errors)
            total_warnings += len(report.warnings)
            all_reports.append(report)
            continue

        # Validate records
        try:
            validate_records(db_path, report, verbose=args.verbose)
        except Exception as e:
            report.error(f"Validation failed: {e}")

        print_report(report)

        total_errors += len(report.errors)
        total_warnings += len(report.warnings)
        all_reports.append(report)

    # Summary

    sum(1 for r in all_reports if not r.has_errors())

    if total_errors > 0:
        return 2
    if total_warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
