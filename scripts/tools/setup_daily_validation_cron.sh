#!/bin/bash
# Setup Daily Validation Cron Job
# ================================
#
# Sets up a cron job to run daily validation reports at noon Eastern time.
# This ensures we catch silent failures before they become big problems.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VALIDATION_SCRIPT="$PROJECT_ROOT/scripts/tools/daily_validation_report.py"

# Check if validation script exists
if [ ! -f "$VALIDATION_SCRIPT" ]; then
    echo "❌ Error: Validation script not found at $VALIDATION_SCRIPT"
    exit 1
fi

# Eastern timezone offset (ET is UTC-5, EDT is UTC-4)
# Cron uses local system time, so we need to calculate noon ET in local time
# For systems in ET: noon ET = noon local
# For systems in other timezones: adjust accordingly

# For now, assume system is in Eastern time - noon ET = noon local
CRON_TIME="0 12 * * *"

# Cron job command
CRON_COMMAND="$CRON_TIME /usr/bin/env python3 $VALIDATION_SCRIPT >> $PROJECT_ROOT/data/daily_summaries/validation_cron.log 2>&1"

echo "🔧 Setting up daily validation cron job..."
echo "Script: $VALIDATION_SCRIPT"
echo "Schedule: Noon Eastern Time (daily)"
echo "Log: $PROJECT_ROOT/data/daily_summaries/validation_cron.log"

# Create log directory
mkdir -p "$PROJECT_ROOT/data/daily_summaries"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "daily_validation_report.py"; then
    echo "⚠️ Cron job already exists. Removing old one first..."
    crontab -l 2>/dev/null | grep -v "daily_validation_report.py" | crontab -
fi

# Add the new cron job
(crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -

echo "✅ Daily validation cron job installed!"
echo ""
echo "To verify: crontab -l"
echo "To test: $VALIDATION_SCRIPT"
echo "To remove: crontab -l | grep -v daily_validation_report.py | crontab -"
