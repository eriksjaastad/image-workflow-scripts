#!/bin/bash
# Create a timestamped archive of training logs and manifests.
# Placeholder uploader: stores archive locally under data/training_backups/.
# Later, replace the final copy step with an S3/B2/Drive uploader.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TRAINING_DIR="$PROJECT_DIR/data/training"
BACKUP_DIR="$PROJECT_DIR/data/training_backups"
LOG_DIR="$PROJECT_DIR/data/log_archives"

mkdir -p "$BACKUP_DIR" "$LOG_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE_PATH="$BACKUP_DIR/training_$STAMP.tar.gz"

echo "[*] Creating training data archive: $ARCHIVE_PATH"

if [ -d "$TRAINING_DIR" ]; then
  tar -czf "$ARCHIVE_PATH" \
    -C "$TRAINING_DIR" \
    $(ls "$TRAINING_DIR" | grep -E '\\.(csv)$' || true) \
    manifests || true
  echo "[*] Archive created: $(basename "$ARCHIVE_PATH")"
else
  echo "[!] Training directory not found: $TRAINING_DIR" >&2
  exit 1
fi

# TODO: Implement cloud upload here (S3/B2/Drive) using env vars.
# Example placeholder:
# aws s3 cp "$ARCHIVE_PATH" "s3://YOUR_BUCKET/training_backups/" --only-show-errors

exit 0


