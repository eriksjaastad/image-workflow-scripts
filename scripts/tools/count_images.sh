#!/usr/bin/env bash
set -euo pipefail


# USAGE:
# bash scripts/tools/count_images.sh /path/to/directory

DIR="${1:-.}"
ex=(
  "*.png" "*.jpg" "*.jpeg" "*.webp" "*.gif"
  "*.tif" "*.tiff" "*.bmp" "*.heic" "*.heif"
)

echo "Directory: $(cd "$DIR" && pwd)"
subdirs=$(find "$DIR" -type d ! -path "$DIR" | wc -l | tr -d ' ')
echo "Subdirectories: $subdirs"

# total images
find_expr=()
for e in "${ex[@]}"; do find_expr+=(-iname "$e" -o); done
# drop trailing -o
unset 'find_expr[${#find_expr[@]}-1]'

total=$(find "$DIR" -type f \( "${find_expr[@]}" \) | wc -l | tr -d ' ')
echo "Images (total): $total"

# per-extension breakdown
for e in "${ex[@]}"; do
  c=$(find "$DIR" -type f -iname "$e" | wc -l | tr -d ' ')
  printf "  %-6s %8d\n" "${e#*.}" "$c"
done
