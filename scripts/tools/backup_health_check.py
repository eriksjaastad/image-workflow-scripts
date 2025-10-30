#!/usr/bin/env python3
"""
Backup Health Check Monitor
===========================

Monitors backup system health and alerts on failures.
Runs regularly to ensure backups are working correctly.

Usage:
  python scripts/tools/backup_health_check.py
  # Add to cron: */30 * * * * python scripts/tools/backup_health_check.py
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.error_monitoring import get_error_monitor


def check_backup_health():
    """Check backup system health and alert on issues."""
    monitor = get_error_monitor("backup_health")
    backup_root = Path("~/project-data-archives/image-workflow").expanduser()

    issues = []

    # 1. Check backup status file exists
    status_file = backup_root / "backup_status.json"
    if not status_file.exists():
        issues.append("Backup status file missing - backups may not be running")
    else:
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)

            # Check last backup age
            last_backup = status_data.get("last_backup")
            if last_backup:
                last_backup_date = datetime.strptime(last_backup, "%Y-%m-%d")
                days_since_backup = (datetime.now() - last_backup_date).days

                if days_since_backup > 2:
                    issues.append(f"Last backup is {days_since_backup} days old - backups overdue")
                elif days_since_backup > 1:
                    monitor.validation_error(f"Backup is {days_since_backup} day old")
            else:
                issues.append("No last backup date in status file")

            # Check backup status
            status = status_data.get("status", "unknown")
            if status == "failed":
                failures = status_data.get("failures", [])
                issues.append(f"Recent backup failed: {len(failures)} failures")
            elif status == "overdue":
                issues.append("Backups are overdue")

        except Exception as e:
            issues.append(f"Backup status file corrupted: {e}")

    # 2. Check recent backup directories exist
    recent_backups = 0
    for i in range(3):  # Check last 3 days
        check_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        backup_dir = backup_root / check_date
        if backup_dir.exists():
            recent_backups += 1

    if recent_backups == 0:
        issues.append("No backup directories found in last 3 days")

    # 3. Check backup log exists and is recent
    backup_log = backup_root / "backup.log"
    if backup_log.exists():
        log_mtime = datetime.fromtimestamp(backup_log.stat().st_mtime)
        hours_since_log = (datetime.now() - log_mtime).total_seconds() / 3600

        if hours_since_log > 48:
            issues.append(f"Backup log is {hours_since_log:.1f} hours old - backups may be failing silently")
    else:
        issues.append("Backup log file missing")

    # Report issues
    if issues:
        critical_issues = [i for i in issues if "missing" in i or "corrupted" in i or "failed" in i]
        warning_issues = [i for i in issues if i not in critical_issues]

        if critical_issues:
            for issue in critical_issues:
                monitor.critical_error(f"BACKUP SYSTEM CRITICAL: {issue}")

        if warning_issues:
            for issue in warning_issues:
                monitor.validation_error(f"BACKUP SYSTEM WARNING: {issue}")

        print(f"❌ Backup health check found {len(issues)} issues")
        for issue in issues:
            print(f"   • {issue}")

        return False
    else:
        print("✅ Backup system health check passed")
        return True


def main():
    """Run backup health check."""
    try:
        success = check_backup_health()

        # Write health check status
        status_file = Path("~/project-data-archives/image-workflow/health_check_status.json").expanduser()
        status_file.parent.mkdir(parents=True, exist_ok=True)

        status_data = {
            "last_check": datetime.now().isoformat(),
            "status": "healthy" if success else "unhealthy",
            "timestamp": datetime.now().timestamp()
        }

        with open(status_file, 'w') as f:
            json.dump(status_data, f)

        return 0 if success else 1

    except Exception as e:
        print(f"💥 Backup health check crashed: {e}")
        monitor = get_error_monitor("backup_health")
        monitor.critical_error(f"Backup health check system failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
