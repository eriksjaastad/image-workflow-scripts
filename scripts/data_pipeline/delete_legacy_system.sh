#!/bin/bash
# Delete Legacy System
# ====================
# Removes the old daily_summaries system after verifying snapshots work.
#
# ⚠️  ONLY RUN THIS AFTER:
# 1. Running test_snapshot_data_integrity.py (all tests pass)
# 2. Testing dashboard manually (verify all data shows up)
# 3. Confirming snapshots cover all historical data
#
# This script will:
# 1. Archive legacy daily_summaries (backup)
# 2. Remove legacy cron job
# 3. Update cron to only use snapshot pipeline
# 4. Clean up legacy consolidation script (optional)

set -e  # Exit on error

PROJECT_DIR="/Users/eriksjaastad/projects/Eros Mate"
cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║              DELETE LEGACY SYSTEM                              ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Safety check
read -p "⚠️  Have you tested the dashboard and confirmed snapshots work? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Aborted. Test the dashboard first!"
    exit 1
fi

echo ""
echo "=== Step 1: Archive legacy daily_summaries ==="
ARCHIVE_DIR="data/legacy_archive_$(date +%Y%m%d)"
mkdir -p "$ARCHIVE_DIR"
if [ -d "data/daily_summaries" ]; then
    echo "  📦 Moving data/daily_summaries → $ARCHIVE_DIR/"
    mv data/daily_summaries "$ARCHIVE_DIR/"
    echo "  ✅ Archived (can restore if needed)"
else
    echo "  ℹ️  data/daily_summaries not found (already deleted?)"
fi

echo ""
echo "=== Step 2: Remove legacy cron job ==="
if crontab -l 2>/dev/null | grep -q "cleanup_logs.py"; then
    echo "  🗑️  Removing cleanup_logs.py cron job"
    crontab -l 2>/dev/null | grep -v "cleanup_logs.py" | crontab -
    echo "  ✅ Legacy cron removed"
else
    echo "  ℹ️  Legacy cron job not found"
fi

echo ""
echo "=== Step 3: Install snapshot-only cron ==="
# Remove any old snapshot cron first
crontab -l 2>/dev/null | grep -v "extract_operation_events_v1.py" | crontab - 2>/dev/null || true

# Install new snapshot cron
CRON_SNAPSHOT="15 2 * * * cd \"$PROJECT_DIR\" && python scripts/data_pipeline/extract_operation_events_v1.py >> data/log_archives/cron_snapshot.log 2>&1 && python scripts/data_pipeline/build_daily_aggregates_v1.py >> data/log_archives/cron_snapshot.log 2>&1 && python scripts/data_pipeline/derive_sessions_from_ops_v1.py >> data/log_archives/cron_snapshot.log 2>&1"

(crontab -l 2>/dev/null; echo "$CRON_SNAPSHOT") | crontab -
echo "  ✅ Snapshot cron installed (runs daily at 2:15 AM)"

echo ""
echo "=== Step 4: Optional - Archive cleanup script ==="
if [ -f "scripts/cleanup_logs.py" ]; then
    read -p "Archive scripts/cleanup_logs.py? (yes/no): " ARCHIVE_SCRIPT
    if [ "$ARCHIVE_SCRIPT" == "yes" ]; then
        mv scripts/cleanup_logs.py "$ARCHIVE_DIR/"
        echo "  ✅ Archived cleanup_logs.py"
    else
        echo "  ℹ️  Keeping cleanup_logs.py"
    fi
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║                  ✅ LEGACY SYSTEM REMOVED                      ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 What was done:"
echo "  ✅ Archived data/daily_summaries → $ARCHIVE_DIR/"
echo "  ✅ Removed legacy cron job"
echo "  ✅ Installed snapshot-only cron (2:15 AM daily)"
echo ""
echo "📊 Active cron jobs:"
crontab -l 2>/dev/null | grep -v "^#" | grep -v "^$"
echo ""
echo "🗄️  Backup location: $ARCHIVE_DIR/"
echo "    (Can delete after 30 days if no issues)"
echo ""
echo "🎯 Dashboard now uses ONLY snapshot data!"

