#!/usr/bin/env bash
set -euo pipefail


# USAGE:
# bash scripts/tools/count_images_zip.sh /path/to/file.zip

if [ $# -eq 0 ]; then
  echo "Usage: $0 <zipfile.zip>"
  exit 1
fi

ZIP_FILE="$1"

if [ ! -f "$ZIP_FILE" ]; then
  echo "Error: File not found: $ZIP_FILE"
  exit 1
fi

# Image extensions to match
ex=(
  "*.png" "*.jpg" "*.jpeg" "*.webp" "*.gif"
  "*.tif" "*.tiff" "*.bmp" "*.heic" "*.heif"
)

echo "Zip file: $(basename "$ZIP_FILE")"
echo "Full path: $(cd "$(dirname "$ZIP_FILE")" && pwd)/$(basename "$ZIP_FILE")"

# Get list of all files in zip
all_files=$(unzip -l "$ZIP_FILE" | awk 'NR>3 {print $NF}' | grep -v '^$')
total_files=$(echo "$all_files" | wc -l | tr -d ' ')
echo "Total files: $total_files"

# Count images (case-insensitive)
total=0
for e in "${ex[@]}"; do
  # Convert glob pattern to grep pattern (*.png -> \.png$)
  pattern="${e#\*}"  # Remove leading *
  c=$(echo "$all_files" | grep -iE "${pattern}\$" | wc -l | tr -d ' ')
  total=$((total + c))
done

echo "Images (total): $total"
echo ""

# Per-extension breakdown
echo "Breakdown by extension:"
for e in "${ex[@]}"; do
  pattern="${e#\*}"
  c=$(echo "$all_files" | grep -iE "${pattern}\$" | wc -l | tr -d ' ')
  if [ "$c" -gt 0 ]; then
    printf "  %-6s %8d\n" "${e#*.}" "$c"
  fi
done

