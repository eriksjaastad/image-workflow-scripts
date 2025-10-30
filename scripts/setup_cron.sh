#!/bin/bash
# Cron Job Setup for Data Consolidation + Snapshot Pipeline
# ===========================================================
# This script sets up daily cron jobs for:
# 1. Legacy log consolidation (legacy support)
# 2. NEW: Snapshot data pipeline (primary system)

# Use the correct project directory path
PROJECT_DIR="/Users/eriksjaastad/projects/image-workflow"

# Legacy consolidation (kept for backward compatibility)
CRON_LEGACY="0 2 * * * cd \"$PROJECT_DIR\" && python scripts/cleanup_logs.py --process-date \$(date -d \"2 days ago\" +%Y%m%d) >> data/log_archives/cron_consolidation.log 2>&1"

# NEW: Snapshot pipeline (runs daily to generate fresh snapshots)
CRON_SNAPSHOT="15 2 * * * cd \"$PROJECT_DIR\" && python scripts/data_pipeline/extract_operation_events_v1.py >> data/log_archives/cron_snapshot.log 2>&1 && python scripts/data_pipeline/build_daily_aggregates_v1.py >> data/log_archives/cron_snapshot.log 2>&1 && python scripts/data_pipeline/derive_sessions_from_ops_v1.py >> data/log_archives/cron_snapshot.log 2>&1"

# Daily backup (every day at 2:10 AM)
CRON_DAILY_BACKUP="10 2 * * * cd \"$PROJECT_DIR\" && python scripts/backup/daily_backup_simple.py >> data/log_archives/cron_daily_backup.log 2>&1"

# Weekly cloud backup rollup
CRON_BACKUP="10 2 * * 0 cd \"$PROJECT_DIR\" && python scripts/backup/weekly_rollup.py >> data/log_archives/cron_weekly_backup.log 2>&1"

# Backup health check (every 6 hours)
CRON_HEALTH_CHECK="0 */6 * * * cd \"$PROJECT_DIR\" && python scripts/tools/backup_health_check.py >> data/log_archives/cron_backup_health.log 2>&1"

# Documentation archive cleanup report (weekly, no deletes)
CRON_DOC_CLEANUP="30 2 * * 0 cd \"$PROJECT_DIR\" && python scripts/tools/generate_archive_cleanup_report.py >> data/log_archives/cron_doc_cleanup.log 2>&1"

echo "🕐 Setting up cron jobs..."
echo "📅 Legacy consolidation: Daily at 2:00 AM"
echo "📅 Snapshot pipeline: Daily at 2:15 AM"
echo "📅 Daily backup: Daily at 2:10 AM"
echo "📅 Backup health check: Every 6 hours"
echo "📅 Cloud backup rollup: Weekly Sunday at 2:10 AM"
echo "📅 Doc cleanup report: Weekly Sunday at 2:30 AM"
echo "📁 Project: $PROJECT_DIR"
echo ""

# Remove old cron jobs
crontab -l 2>/dev/null | grep -v "cleanup_logs.py" | grep -v "weekly_rollup.py" | grep -v "daily_backup_simple.py" | grep -v "backup_health_check.py" | grep -v "upload_training_backup.sh" | grep -v "extract_operation_events_v1.py" | grep -v "generate_archive_cleanup_report.py" | crontab -

# Add/refresh all cron jobs
(crontab -l 2>/dev/null; echo "$CRON_LEGACY"; echo "$CRON_SNAPSHOT"; echo "$CRON_DAILY_BACKUP"; echo "$CRON_HEALTH_CHECK"; echo "$CRON_BACKUP"; echo "$CRON_DOC_CLEANUP") | crontab -

echo "✅ Cron jobs installed successfully!"
echo ""
echo "📋 To view current cron jobs: crontab -l"
echo "🗑️  To remove cron jobs: crontab -e (then delete the lines)"
echo ""
echo "📝 Logs will be written to:"
echo "   - data/log_archives/cron_consolidation.log (legacy)"
echo "   - data/log_archives/cron_snapshot.log (NEW snapshot pipeline)"
echo "   - data/log_archives/cron_daily_backup.log (daily backup)"
echo "   - data/log_archives/cron_backup_health.log (backup monitoring)"
echo "   - data/log_archives/cron_weekly_backup.log (cloud backup)"
echo "   - data/log_archives/cron_doc_cleanup.log"
echo ""
echo "🎯 NEW: Dashboard now uses snapshot data (data/snapshot/) as primary source!"
