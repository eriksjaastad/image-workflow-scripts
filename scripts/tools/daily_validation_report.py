#!/usr/bin/env python3
"""
Daily Validation Report
=======================

Runs comprehensive validation checks on the image workflow system.
Generates reports and sends notifications for any issues found.

Designed to run daily via cron job to catch silent failures early.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.error_monitoring import get_error_monitor


class DailyValidationReport:
    """Comprehensive daily validation system."""

    def __init__(self):
        self.monitor = get_error_monitor("daily_validation")
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "issues": [],
            "summary": {},
        }

    def run_all_checks(self) -> Dict:
        """Run all validation checks."""

        print("🔍 Starting Daily Validation Report")
        print("=" * 60)

        # Core system checks
        self.check_filetracker_access()
        self.check_database_integrity()
        self.check_recent_errors()
        self.check_data_quality()
        self.check_disk_space()
        self.check_git_status()

        # Generate summary
        self.generate_summary()

        return self.report

    def check_filetracker_access(self):
        """Check if FileTracker can be initialized."""
        print("\n📊 Checking FileTracker access...")

        try:
            from file_tracker import FileTracker

            tracker = FileTracker("validation_test")
            tracker.close()  # Clean up

            self.report["checks"]["filetracker"] = "PASS"
            print("✅ FileTracker access OK")

        except Exception as e:
            self.report["checks"]["filetracker"] = "FAIL"
            self.report["issues"].append(
                {
                    "check": "filetracker",
                    "severity": "CRITICAL",
                    "message": f"FileTracker initialization failed: {e}",
                    "action": "Check file_tracker module and database access",
                }
            )
            print(f"❌ FileTracker access FAILED: {e}")

    def check_database_integrity(self):
        """Check AI training databases integrity."""
        print("\n🗄️ Checking database integrity...")

        db_dir = PROJECT_ROOT / "data" / "training" / "ai_training_decisions"
        if not db_dir.exists():
            self.report["checks"]["database"] = "SKIP"
            print("⚠️ No training databases found")
            return

        issues = 0
        total_dbs = 0

        for db_file in db_dir.glob("*.db"):
            total_dbs += 1
            try:
                import sqlite3

                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()

                # Check table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='decisions'"
                )
                if not cursor.fetchone():
                    issues += 1
                    self.report["issues"].append(
                        {
                            "check": "database",
                            "severity": "HIGH",
                            "message": f"Missing decisions table in {db_file.name}",
                            "action": "Reinitialize database",
                        }
                    )

                # Check for recent activity (last 7 days)
                seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute(
                    "SELECT COUNT(*) FROM decisions WHERE timestamp > ?",
                    (seven_days_ago,),
                )
                recent_count = cursor.fetchone()[0]

                conn.close()

            except Exception as e:
                issues += 1
                self.report["issues"].append(
                    {
                        "check": "database",
                        "severity": "HIGH",
                        "message": f"Database corruption in {db_file.name}: {e}",
                        "action": "Check database file integrity",
                    }
                )

        status = "PASS" if issues == 0 else f"FAIL ({issues}/{total_dbs} issues)"
        self.report["checks"]["database"] = status

        if issues == 0:
            print(f"✅ Database integrity OK ({total_dbs} databases checked)")
        else:
            print(f"❌ Database integrity ISSUES: {issues}/{total_dbs} databases")

    def check_recent_errors(self):
        """Check for recent error logs."""
        print("\n📋 Checking recent error logs...")

        error_dir = PROJECT_ROOT / "data" / "error_logs"
        if not error_dir.exists():
            self.report["checks"]["error_logs"] = "SKIP"
            print("⚠️ No error logs directory found")
            return

        recent_errors = []
        for error_file in error_dir.glob("*.log"):
            try:
                mtime = datetime.fromtimestamp(error_file.stat().st_mtime)
                if mtime > datetime.now() - timedelta(days=1):
                    with open(error_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            recent_errors.append(
                                {
                                    "file": error_file.name,
                                    "last_error": lines[-1].strip()
                                    if lines
                                    else "Empty file",
                                }
                            )
            except Exception as e:
                print(f"Error reading {error_file}: {e}")

        if recent_errors:
            self.report["checks"]["error_logs"] = (
                f"WARN ({len(recent_errors)} recent errors)"
            )
            for error in recent_errors:
                self.report["issues"].append(
                    {
                        "check": "error_logs",
                        "severity": "MEDIUM",
                        "message": f"Recent errors in {error['file']}: {error['last_error'][:100]}...",
                        "action": "Review error logs and fix underlying issues",
                    }
                )
            print(f"⚠️ Found {len(recent_errors)} recent error logs")
        else:
            self.report["checks"]["error_logs"] = "PASS"
            print("✅ No recent error logs")

    def check_data_quality(self):
        """Run data quality validation tests."""
        print("\n🔍 Running data quality checks...")

        try:
            # Run the inline validation tests
            result = subprocess.run(
                [sys.executable, "scripts/tests/test_inline_validation.py"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )

            if result.returncode == 0:
                self.report["checks"]["data_quality"] = "PASS"
                print("✅ Data quality validation passed")
            else:
                self.report["checks"]["data_quality"] = "FAIL"
                self.report["issues"].append(
                    {
                        "check": "data_quality",
                        "severity": "HIGH",
                        "message": f"Data quality validation failed: {result.stdout[-200:]}",
                        "action": "Run test_inline_validation.py manually to see details",
                    }
                )
                print("❌ Data quality validation FAILED")

        except Exception as e:
            self.report["checks"]["data_quality"] = "ERROR"
            self.report["issues"].append(
                {
                    "check": "data_quality",
                    "severity": "HIGH",
                    "message": f"Could not run data quality checks: {e}",
                    "action": "Fix test_inline_validation.py script",
                }
            )
            print(f"❌ Could not run data quality checks: {e}")

    def check_disk_space(self):
        """Check available disk space."""
        print("\n💾 Checking disk space...")

        try:
            result = subprocess.run(
                ["df", "-h", str(PROJECT_ROOT)], capture_output=True, text=True
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    # Parse the second line (data line)
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        available = parts[3]
                        use_percent = parts[4].rstrip("%")

                        try:
                            use_percent_int = int(use_percent)
                            if use_percent_int > 90:
                                self.report["checks"]["disk_space"] = (
                                    f"WARN ({use_percent}% used)"
                                )
                                self.report["issues"].append(
                                    {
                                        "check": "disk_space",
                                        "severity": "MEDIUM",
                                        "message": f"Disk space critically low: {use_percent}% used, {available} available",
                                        "action": "Clean up old files or expand storage",
                                    }
                                )
                                print(f"⚠️ Disk space LOW: {use_percent}% used")
                            else:
                                self.report["checks"]["disk_space"] = (
                                    f"PASS ({use_percent}% used)"
                                )
                                print(f"✅ Disk space OK: {use_percent}% used")
                        except ValueError:
                            self.report["checks"]["disk_space"] = "UNKNOWN"
                            print("⚠️ Could not parse disk usage percentage")
                    else:
                        self.report["checks"]["disk_space"] = "UNKNOWN"
                        print("⚠️ Could not parse df output")
                else:
                    self.report["checks"]["disk_space"] = "UNKNOWN"
                    print("⚠️ df command failed")
            else:
                self.report["checks"]["disk_space"] = "UNKNOWN"
                print("⚠️ df command failed")

        except Exception as e:
            self.report["checks"]["disk_space"] = "ERROR"
            print(f"❌ Could not check disk space: {e}")

    def check_git_status(self):
        """Check git repository status."""
        print("\n🔄 Checking git status...")

        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )

            if result.returncode == 0:
                if result.stdout.strip():
                    self.report["checks"]["git_status"] = "WARN (uncommitted changes)"
                    self.report["issues"].append(
                        {
                            "check": "git_status",
                            "severity": "LOW",
                            "message": "Uncommitted changes in repository",
                            "action": "Commit or stash changes",
                        }
                    )
                    print("⚠️ Uncommitted changes in repository")
                else:
                    self.report["checks"]["git_status"] = "PASS"
                    print("✅ Repository is clean")
            else:
                self.report["checks"]["git_status"] = "ERROR"
                print("❌ Git status check failed")

        except Exception as e:
            self.report["checks"]["git_status"] = "ERROR"
            print(f"❌ Could not check git status: {e}")

    def generate_summary(self):
        """Generate summary statistics."""
        total_checks = len(self.report["checks"])
        passed = sum(
            1 for status in self.report["checks"].values() if status.startswith("PASS")
        )
        failed = sum(
            1
            for status in self.report["checks"].values()
            if "FAIL" in status or "ERROR" in status
        )
        warnings = sum(
            1 for status in self.report["checks"].values() if "WARN" in status
        )

        self.report["summary"] = {
            "total_checks": total_checks,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "critical_issues": len(
                [i for i in self.report["issues"] if i["severity"] == "CRITICAL"]
            ),
            "high_issues": len(
                [i for i in self.report["issues"] if i["severity"] == "HIGH"]
            ),
        }

    def send_notifications(self):
        """Send notifications for issues found."""
        critical_issues = [
            i for i in self.report["issues"] if i["severity"] == "CRITICAL"
        ]
        high_issues = [i for i in self.report["issues"] if i["severity"] == "HIGH"]

        if critical_issues or high_issues:
            title = "🚨 Daily Validation Report: Issues Found"
            message = f"Found {len(critical_issues)} critical and {len(high_issues)} high-priority issues"

            self.monitor._send_macos_notification(title, message)

            # Also log as validation errors
            if critical_issues:
                for issue in critical_issues:
                    self.monitor.validation_error(f"CRITICAL: {issue['message']}")

    def save_report(self):
        """Save the report to file."""
        reports_dir = PROJECT_ROOT / "data" / "daily_summaries"
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"validation_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(self.report, f, indent=2)

        print(f"\n📄 Report saved to: {report_file}")
        return report_file


def main():
    """Run daily validation report."""
    print("🔍 Starting Daily Validation Report")
    print("=" * 60)

    validator = DailyValidationReport()

    try:
        report = validator.run_all_checks()

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        summary = report["summary"]
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Critical Issues: {summary['critical_issues']}")
        print(f"High Priority Issues: {summary['high_issues']}")

        if summary["critical_issues"] > 0 or summary["failed"] > 0:
            print("\n❌ VALIDATION FAILED - Requires attention!")
        else:
            print("\n✅ All validations passed!")

        # Send notifications for issues
        validator.send_notifications()

        # Save report
        validator.save_report()

        # Exit with appropriate code
        return 1 if (summary["critical_issues"] > 0 or summary["failed"] > 0) else 0

    except Exception as e:
        print(f"\n💥 VALIDATION SYSTEM ERROR: {e}")
        validator.monitor.critical_error("Daily validation system failed", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
