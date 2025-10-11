#!/usr/bin/env bash
set -euo pipefail

# Source and destination can be overridden via env; defaults are safe and generic
SRC="${SRC:-/Volumes/T7Shield/mojo1-4-new/mojo1-4/mojo2}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DEST="${DEST:-$PROJECT_ROOT/sandbox}"

if [[ ! -d "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST/mojo2"

# Use rsync for robust, resumable copy with progress; preserves timestamps/permissions
rsync -avh --progress \
  --exclude='*/.DS_Store' \
  --exclude='*/Thumbs.db' \
  "$SRC"/ "$DEST/mojo2/"

echo "Copy complete: $DEST/mojo2"

