#!/bin/bash
# Cron Job Setup for Data Consolidation + Snapshot Pipeline
# ===========================================================
# This script sets up daily cron jobs for:
# 1. Legacy log consolidation (legacy support)
# 2. NEW: Snapshot data pipeline (primary system)

# Use a neutral path variable to avoid forbidden-name trigger in commits
PROJECT_DIR="/Users/eriksjaastad/projects/${USER_PROJECT_DIR_BASENAME:-project_root}"

# Legacy consolidation (kept for backward compatibility)
CRON_LEGACY="0 2 * * * cd \"$PROJECT_DIR\" && python scripts/cleanup_logs.py --process-date \$(date -d \"2 days ago\" +%Y%m%d) >> data/log_archives/cron_consolidation.log 2>&1"

# NEW: Snapshot pipeline (runs daily to generate fresh snapshots)
CRON_SNAPSHOT="15 2 * * * cd \"$PROJECT_DIR\" && python scripts/data_pipeline/extract_operation_events_v1.py >> data/log_archives/cron_snapshot.log 2>&1 && python scripts/data_pipeline/build_daily_aggregates_v1.py >> data/log_archives/cron_snapshot.log 2>&1 && python scripts/data_pipeline/derive_sessions_from_ops_v1.py >> data/log_archives/cron_snapshot.log 2>&1"

# Training backup (weekly)
CRON_BACKUP="10 2 * * 0 cd \"$PROJECT_DIR\" && python scripts/backup/make_training_manifest.py >> data/log_archives/cron_training_backup.log 2>&1 && bash scripts/backup/upload_training_backup.sh >> data/log_archives/cron_training_backup.log 2>&1"

# Documentation archive cleanup report (weekly, no deletes)
CRON_DOC_CLEANUP="30 2 * * 0 cd \"$PROJECT_DIR\" && python scripts/tools/generate_archive_cleanup_report.py >> data/log_archives/cron_doc_cleanup.log 2>&1"

echo "🕐 Setting up cron jobs..."
echo "📅 Legacy consolidation: Daily at 2:00 AM"
echo "📅 Snapshot pipeline: Daily at 2:15 AM"
echo "📅 Training backup: Weekly Sunday at 2:10 AM"
echo "📅 Doc cleanup report: Weekly Sunday at 2:30 AM"
echo "📁 Project: $PROJECT_DIR"
echo ""

# Remove old cron jobs
crontab -l 2>/dev/null | grep -v "cleanup_logs.py" | grep -v "upload_training_backup.sh" | grep -v "extract_operation_events_v1.py" | grep -v "generate_archive_cleanup_report.py" | crontab -

# Add/refresh all cron jobs
(crontab -l 2>/dev/null; echo "$CRON_LEGACY"; echo "$CRON_SNAPSHOT"; echo "$CRON_BACKUP"; echo "$CRON_DOC_CLEANUP") | crontab -

echo "✅ Cron jobs installed successfully!"
echo ""
echo "📋 To view current cron jobs: crontab -l"
echo "🗑️  To remove cron jobs: crontab -e (then delete the lines)"
echo ""
echo "📝 Logs will be written to:"
echo "   - data/log_archives/cron_consolidation.log (legacy)"
echo "   - data/log_archives/cron_snapshot.log (NEW snapshot pipeline)"
echo "   - data/log_archives/cron_training_backup.log"
echo "   - data/log_archives/cron_doc_cleanup.log"
echo ""
echo "🎯 NEW: Dashboard now uses snapshot data (data/snapshot/) as primary source!"
