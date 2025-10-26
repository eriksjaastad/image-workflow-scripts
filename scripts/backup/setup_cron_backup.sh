#!/bin/bash
# Setup a daily cron job to back up workflow data
# Usage: bash scripts/backup/setup_cron_backup.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEST_DIR="$HOME/project-data-archives/image-workflow"
PY_BIN="$PROJECT_DIR/.venv311/bin/python"

mkdir -p "$DEST_DIR"

CRON_LINE="10 2 * * * /bin/bash -lc 'cd \"$PROJECT_DIR\" && if [ -x \"$PY_BIN\" ]; then \"$PY_BIN\" scripts/backup/daily_backup.py --dest \"$DEST_DIR\"; else python scripts/backup/daily_backup.py --dest \"$DEST_DIR\"; fi >> \"$DEST_DIR/cron_backup.log\" 2>&1'"

# Install idempotently
TMP_CRON=$(mktemp)
crontab -l 2>/dev/null | grep -v "scripts/backup/daily_backup.py" > "$TMP_CRON" || true
echo "$CRON_LINE" >> "$TMP_CRON"
crontab "$TMP_CRON"
rm -f "$TMP_CRON"

echo "✅ Cron installed: $CRON_LINE"
echo "Backups will appear under $DEST_DIR/YYYY-MM-DD/"

