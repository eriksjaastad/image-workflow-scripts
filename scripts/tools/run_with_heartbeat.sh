#!/usr/bin/env bash
set -euo pipefail

# Usage: run_with_heartbeat.sh <heartbeat_file> <command...>
# Writes a timestamp to the heartbeat file every 2 seconds while the command runs.

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <heartbeat_file> <command...>" >&2
  exit 2
fi

HB="$1"; shift
mkdir -p "$(dirname "$HB")"
touch "$HB"

# Build the command string to execute via bash -lc for compatibility
CMD="$*"

# Start heartbeat writer in background
(
  while :; do
    date +%s >> "$HB" || true
    sleep 2
  done
) &
HB_PID=$!

# Run the actual command
bash -lc "$CMD"
RC=$?

# Cleanup heartbeat writer
kill "$HB_PID" 2>/dev/null || true
exit "$RC"


