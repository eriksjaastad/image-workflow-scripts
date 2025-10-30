#!/usr/bin/env python3
"""
Simple, Reliable Daily Backup with Monitoring
==============================================

A simplified backup script that definitely works, logs everything,
and alerts on failures to prevent silent failures.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import error monitoring
try:
    from scripts.utils.error_monitoring import fatal_error, get_error_monitor
except ImportError:
    # Fallback if monitoring not available
    def get_error_monitor(script_name="backup"):
        class MockMonitor:
            def critical_error(self, msg, exc=None):
                print(f"CRITICAL BACKUP ERROR: {msg}", file=sys.stderr)
                if exc:
                    print(f"Exception: {exc}", file=sys.stderr)

        return MockMonitor()

    def fatal_error(msg, exc=None):
        import sys

        print(f"FATAL BACKUP ERROR: {msg}", file=sys.stderr)
        if exc:
            print(f"Exception: {exc}", file=sys.stderr)
        sys.exit(1)


def log(message, level="INFO"):
    """Log to both stdout and backup log file."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}", flush=True)

    # Also log to file
    log_file = Path("~/project-data-archives/image-workflow/backup.log").expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {level}: {message}\n")


# Removed alert_failure - it was a footgun that crashed scripts


def find_database_files(root_dir: Path) -> List[Path]:
    """Find all SQLite database files in the project."""
    db_files = []

    # Common database file patterns
    patterns = ["*.db", "*.sqlite", "*.sqlite3"]

    for pattern in patterns:
        db_files.extend(root_dir.rglob(pattern))

    # Filter out any files in backup directories to avoid infinite loops
    db_files = [f for f in db_files if "project-data-archives" not in str(f)]

    return sorted(db_files)


def verify_backup(src, dst, name):
    """Verify that backup was successful."""
    monitor = get_error_monitor("daily_backup")

    try:
        if not dst.exists():
            monitor.validation_error(f"Backup destination missing: {dst}")
            return False

        if src.is_file():
            src_size = src.stat().st_size
            dst_size = dst.stat().st_size
            if src_size != dst_size:
                monitor.validation_error(
                    f"File size mismatch for {name}: {src_size} vs {dst_size}"
                )
                return False
        else:
            # Count files
            src_files = len(list(src.rglob("*")))
            dst_files = len(list(dst.rglob("*")))
            if src_files != dst_files:
                monitor.validation_error(
                    f"File count mismatch for {name}: {src_files} vs {dst_files}"
                )
                return False

        log(f"✅ Verified backup integrity for {name}")
        return True

    except Exception as e:
        monitor.validation_error(
            f"Backup verification failed for {name}", {"error": str(e)}
        )
        return False


def backup_directory(src, dst, name):
    """Backup a directory with error handling."""
    try:
        if not src.exists():
            log(f"⚠️  Source {name} not found: {src}")
            return False

        log(f"📁 Backing up {name} from {src} to {dst}")

        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            size = dst.stat().st_size
            log(f"✅ Copied file {name}: {size:,} bytes")
        else:
            # Copy directory
            dst.mkdir(parents=True, exist_ok=True)

            files_copied = 0
            total_size = 0

            for item in src.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(src)
                    dest_file = dst / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        shutil.copy2(item, dest_file)
                        files_copied += 1
                        total_size += dest_file.stat().st_size
                    except Exception as e:
                        log(f"❌ Failed to copy {item}: {e}")

            log(
                f"✅ Copied directory {name}: {files_copied} files, {total_size:,} bytes"
            )

        return True

    except Exception as e:
        log(f"❌ Failed to backup {name}: {e}")
        return False


def main():
    """Run the daily backup."""
    log("🚀 Starting Daily Backup", "START")

    success = True
    backup_items = []

    try:
        # Setup paths
        data_dir = PROJECT_ROOT / "data"
        backup_root = Path("~/project-data-archives/image-workflow").expanduser()
        today = datetime.now().strftime("%Y-%m-%d")
        backup_dir = backup_root / today

        log(f"📂 Backup destination: {backup_dir}")

        # Create backup directory
        backup_dir.mkdir(parents=True, exist_ok=True)

        # What to backup
        backup_sources = [
            (data_dir / "file_operations_logs", "file operations logs"),
            (data_dir / "snapshot", "snapshot data"),
            (data_dir / "training", "training data"),
            (data_dir / "ai_data", "AI data"),
        ]

        # Separate backup for databases (SQLite files throughout the project)
        log("📊 Finding database files...")
        db_sources = find_database_files(PROJECT_ROOT)
        if db_sources:
            db_backup_dir = backup_dir / "databases"
            db_backup_dir.mkdir(parents=True, exist_ok=True)

            db_copied = 0
            for db_path in db_sources:
                try:
                    # Create relative path in backup
                    rel_path = db_path.relative_to(PROJECT_ROOT)
                    backup_path = db_backup_dir / str(rel_path)
                    backup_path.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(db_path, backup_path)
                    db_copied += 1

                    # Verify the copy
                    if backup_path.stat().st_size == db_path.stat().st_size:
                        log(f"✅ Database: {rel_path}")
                    else:
                        log(f"❌ Size mismatch for database: {rel_path}", "ERROR")

                except Exception as e:
                    log(f"❌ Failed to backup database {db_path}: {e}", "ERROR")

            log(f"📊 Backed up {db_copied} database files")
            total_db_files = db_copied
        else:
            log("⚠️ No database files found to backup")
            total_db_files = 0

        total_files = 0
        total_size = 0

        for src, name in backup_sources:
            dst = backup_dir / src.name
            if backup_directory(src, dst, name):
                # Verify the backup
                if not verify_backup(src, dst, name):
                    success = False
                    continue

                # Count files in destination
                if dst.exists():
                    if dst.is_file():
                        total_files += 1
                        total_size += dst.stat().st_size
                    else:
                        for f in dst.rglob("*"):
                            if f.is_file():
                                total_files += 1
                                total_size += f.stat().st_size

                backup_items.append(
                    {
                        "name": name,
                        "source": str(src),
                        "destination": str(dst),
                        "status": "success",
                    }
                )
            else:
                success = False
                backup_items.append(
                    {
                        "name": name,
                        "source": str(src),
                        "destination": str(dst),
                        "status": "failed",
                    }
                )

        # Add database backup info
        if total_db_files > 0:
            backup_items.append(
                {
                    "name": "databases",
                    "source": "project-wide SQLite files",
                    "destination": str(backup_dir / "databases"),
                    "status": "success" if total_db_files > 0 else "failed",
                }
            )

        total_files += total_db_files

        # Create manifest
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "backup_date": today,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "overall_status": "success" if success else "failed",
            "source": str(PROJECT_ROOT),
            "destination": str(backup_dir),
            "items": backup_items,
        }

        manifest_file = backup_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        # Create status file for dashboard monitoring
        status_file = backup_root / "backup_status.json"
        status_info = {
            "last_backup": today,
            "last_backup_timestamp": datetime.now().isoformat(),
            "status": "success" if success else "failed",
            "total_files": total_files,
            "total_size_mb": manifest["total_size_mb"],
            "failures": [item for item in backup_items if item["status"] == "failed"],
        }
        with open(status_file, "w") as f:
            json.dump(status_info, f, indent=2)

        if success:
            log("✅ Backup completed successfully!", "SUCCESS")
            log(f"📊 Summary: {total_files} files, {manifest['total_size_mb']} MB")
            log(f"📄 Manifest: {manifest_file}")

            # Success notification
            monitor = get_error_monitor("daily_backup")
            monitor._send_macos_notification(
                "Backup Success",
                f"Daily backup completed: {total_files} files, {manifest['total_size_mb']} MB",
            )
        else:
            log("❌ Backup completed with failures!", "ERROR")
            monitor = get_error_monitor("daily_backup")
            monitor.validation_error("Daily backup completed but some items failed")

    except Exception as e:
        success = False
        log(f"💥 Backup failed with exception: {e}", "CRITICAL")
        # This is a fatal error - the entire backup process failed
        fatal_error("Daily backup failed with unrecoverable exception", e)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
