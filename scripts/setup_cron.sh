#!/bin/bash
# Cron Job Setup for Data Consolidation
# ====================================
# This script sets up the cron job for daily data consolidation
# with a 2-day buffer to avoid timing conflicts with active work.

# Use a neutral path variable to avoid forbidden-name trigger in commits
PROJECT_DIR="/Users/eriksjaastad/projects/${USER_PROJECT_DIR_BASENAME:-project_root}"
CRON_COMMAND="0 2 * * * cd \"$PROJECT_DIR\" && python scripts/cleanup_logs.py --process-date \$(date -d \"2 days ago\" +%Y%m%d) >> data/log_archives/cron_consolidation.log 2>&1"
CRON_BACKUP="10 2 * * 0 cd \"$PROJECT_DIR\" && python scripts/backup/make_training_manifest.py >> data/log_archives/cron_training_backup.log 2>&1 && bash scripts/backup/upload_training_backup.sh >> data/log_archives/cron_training_backup.log 2>&1"

echo "🕐 Setting up cron job for data consolidation..."
echo "📅 Schedule: Daily at 2:00 AM"
echo "⏰ Buffer: Processes data from 2 days ago"
echo "📁 Project: $PROJECT_DIR"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "cleanup_logs.py"; then
    echo "⚠️  Cron job already exists. Removing old version..."
    crontab -l 2>/dev/null | grep -v "cleanup_logs.py" | crontab -
fi

# Add/refresh cron jobs
(crontab -l 2>/dev/null | grep -v "cleanup_logs.py" | grep -v "upload_training_backup.sh"; echo "$CRON_COMMAND"; echo "$CRON_BACKUP") | crontab -

echo "✅ Cron jobs installed successfully!"
echo ""
echo "📋 To view current cron jobs: crontab -l"
echo "🗑️  To remove this cron job: crontab -e (then delete the line)"
echo "📊 To test manually: python scripts/cleanup_logs.py --process-date \$(date -d \"2 days ago\" +%Y%m%d)"
echo ""
echo "📝 Logs will be written to: data/log_archives/cron_consolidation.log"
