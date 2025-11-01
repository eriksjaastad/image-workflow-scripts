#!/usr/bin/env python3
"""
Training Data Validation Script
================================
Validates integrity of AI training data (selections, crops, embeddings).

Run this script daily or after any data collection to catch problems early!

Usage:
    python scripts/ai/validate_training_data.py
    python scripts/ai/validate_training_data.py --fix  # Auto-fix some issues
    
Exit codes:
    0 = All checks passed
    1 = Warnings found
    2 = Errors found (data is unusable)
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SELECTION_LOG = PROJECT_ROOT / "data/training/selection_only_log.csv"
CROP_LOG = PROJECT_ROOT / "data/training/select_crop_log.csv"
EMBEDDINGS_CACHE = PROJECT_ROOT / "data/ai_data/cache/processed_images.jsonl"
EMBEDDINGS_DIR = PROJECT_ROOT / "data/ai_data/cache/embeddings"


class ValidationReport:
    """Collects validation results."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.stats = {}
    
    def error(self, msg: str):
        self.errors.append(msg)
    
    def warning(self, msg: str):
        self.warnings.append(msg)
    
    def info_msg(self, msg: str):
        self.info.append(msg)
    
    def add_stat(self, key: str, value):
        self.stats[key] = value
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def print_report(self):
        """Print formatted validation report."""
        # Stats
        if self.stats:
            for _key, _value in self.stats.items():
                pass
        
        # Info
        if self.info:
            for _msg in self.info:
                pass
        
        # Warnings
        if self.warnings:
            for _msg in self.warnings:
                pass
        
        # Errors
        if self.errors:
            for _msg in self.errors:
                pass
        
        # Summary
        if self.has_errors() or self.has_warnings():
            pass
        else:
            pass


def validate_selection_data(report: ValidationReport) -> dict[str, bool]:
    """Validate selection log integrity."""
    if not SELECTION_LOG.exists():
        report.error(f"Selection log missing: {SELECTION_LOG}")
        return {}
    
    # Load and parse
    selections = []
    filename_to_path = {}
    
    with open(SELECTION_LOG) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            try:
                chosen = row['chosen_path']
                filename = Path(chosen).name
                filename_to_path[filename] = chosen
                selections.append(row)
            except Exception as e:
                report.warning(f"Line {i}: Failed to parse row - {e}")
    
    report.add_stat("Selection entries", len(selections))
    
    # Recent activity
    if selections:
        try:
            recent = [s for s in selections if s.get('timestamp')]
            if recent:
                last_entry = recent[-1]
                last_time = datetime.fromisoformat(last_entry['timestamp'].replace('Z', '+00:00'))
                days_ago = (datetime.now().astimezone() - last_time).days
                
                if days_ago > 7:
                    report.warning(f"Last selection was {days_ago} days ago - no recent activity")
                else:
                    report.info_msg(f"Last selection: {days_ago} days ago")
        except Exception as e:
            report.warning(f"Could not parse selection timestamps: {e}")
    
    return filename_to_path


def validate_crop_data(report: ValidationReport, selection_filenames: dict[str, str]):
    """Validate crop log integrity."""
    if not CROP_LOG.exists():
        report.error(f"Crop log missing: {CROP_LOG}")
        return {}
    
    # Load and analyze
    crops = []
    invalid_dimensions = []
    invalid_coords = []
    filename_to_crop = {}
    
    with open(CROP_LOG) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            try:
                chosen_idx = int(row['chosen_index'])
                width_key = f'width_{chosen_idx}'
                height_key = f'height_{chosen_idx}'
                
                width = float(row.get(width_key, 0))
                height = float(row.get(height_key, 0))
                
                x1 = float(row['crop_x1'])
                y1 = float(row['crop_y1'])
                x2 = float(row['crop_x2'])
                y2 = float(row['crop_y2'])
                
                filename = Path(row['chosen_path']).name
                
                # Check for invalid dimensions
                if width <= 0 or height <= 0:
                    invalid_dimensions.append((i, filename, width, height))
                
                # Check for invalid crop coords (in normalized space)
                # Normalize if in pixels
                if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                    if width > 0 and height > 0:
                        x1_n, y1_n = x1/width, y1/height
                        x2_n, y2_n = x2/width, y2/height
                    else:
                        x1_n, y1_n, x2_n, y2_n = x1, y1, x2, y2
                else:
                    x1_n, y1_n, x2_n, y2_n = x1, y1, x2, y2
                
                if x2_n <= x1_n or y2_n <= y1_n or x1_n < 0 or y1_n < 0 or x2_n > 1 or y2_n > 1:
                    invalid_coords.append((i, filename, (x1, y1, x2, y2)))
                else:
                    crops.append(row)
                    filename_to_crop[filename] = row
                    
            except Exception as e:
                report.warning(f"Crop line {i}: Failed to parse - {e}")
    
    report.add_stat("Crop entries", len(crops))
    
    # Report dimension issues
    if invalid_dimensions:
        count = len(invalid_dimensions)
        report.error(f"{count} crop entries have invalid dimensions (0x0)")
        if count <= 5:
            for line, fname, w, h in invalid_dimensions:
                report.error(f"   Line {line}: {fname} → {w}x{h}")
        else:
            report.error("   Showing first 5:")
            for line, fname, w, h in invalid_dimensions[:5]:
                report.error(f"   Line {line}: {fname} → {w}x{h}")
    
    # Report coordinate issues
    if invalid_coords:
        count = len(invalid_coords)
        report.error(f"{count} crop entries have invalid coordinates")
        if count <= 5:
            for line, fname, coords in invalid_coords:
                report.error(f"   Line {line}: {fname} → {coords}")
        else:
            report.error("   Showing first 5:")
            for line, fname, coords in invalid_coords[:5]:
                report.error(f"   Line {line}: {fname} → {coords}")
    
    return filename_to_crop


def validate_embeddings(report: ValidationReport, selection_files: dict[str, str], crop_files: dict[str, str]):
    """Validate embeddings cache and files."""
    if not EMBEDDINGS_CACHE.exists():
        report.error(f"Embeddings cache missing: {EMBEDDINGS_CACHE}")
        return set(), set()
    
    if not EMBEDDINGS_DIR.exists():
        report.error(f"Embeddings directory missing: {EMBEDDINGS_DIR}")
        return set(), set()
    
    # Load cache
    cache_entries = []
    filename_cache = set()
    hash_to_file = {}
    
    with open(EMBEDDINGS_CACHE) as f:
        for line in f:
            try:
                entry = json.loads(line)
                cache_entries.append(entry)
                filename = Path(entry['image_path']).name
                filename_cache.add(filename)
                hash_to_file[entry['hash']] = entry['image_path']
            except Exception as e:
                report.warning(f"Failed to parse cache entry: {e}")
    
    report.add_stat("Embeddings cache entries", len(cache_entries))
    
    # Check actual files
    embedding_files = list(EMBEDDINGS_DIR.glob('*.npy'))
    report.add_stat("Embedding files on disk", len(embedding_files))
    
    # Find orphaned cache entries (cache but no file)
    missing_files = []
    for hash_val, path in hash_to_file.items():
        emb_file = EMBEDDINGS_DIR / f"{hash_val}.npy"
        if not emb_file.exists():
            missing_files.append(path)
    
    if missing_files:
        count = len(missing_files)
        report.error(f"{count} embeddings in cache but files missing on disk!")
        if count <= 10:
            for path in missing_files:
                report.error(f"   Missing: {path}")
        else:
            report.error("   Showing first 10:")
            for path in missing_files[:10]:
                report.error(f"   Missing: {path}")
        report.error("   → Run: python scripts/ai/compute_embeddings.py")
    
    # Check selection coverage
    all_training_files = set(selection_files.keys()) | set(crop_files.keys())
    missing_embeddings = all_training_files - filename_cache
    
    if missing_embeddings:
        count = len(missing_embeddings)
        report.warning(f"{count} training images missing embeddings")
        if count <= 10:
            for fname in sorted(missing_embeddings):
                report.warning(f"   No embedding: {fname}")
        else:
            report.warning("   Showing first 10:")
            for fname in sorted(list(missing_embeddings)[:10]):
                report.warning(f"   No embedding: {fname}")
        report.warning("   → Run: python scripts/ai/compute_embeddings.py")
    
    # Report coverage
    if all_training_files:
        coverage = (len(all_training_files) - len(missing_embeddings)) / len(all_training_files) * 100
        report.add_stat("Embedding coverage", f"{coverage:.1f}%")
        
        if coverage < 95:
            report.error(f"Low embedding coverage ({coverage:.1f}%) - many training images missing embeddings!")
        elif coverage < 99:
            report.warning(f"Embedding coverage is {coverage:.1f}% - some images missing embeddings")
    
    return filename_cache, missing_files


def validate_recent_activity(report: ValidationReport):
    """Check for recent data collection activity."""
    # Check last modified times
    logs_to_check = [
        ("Selection log", SELECTION_LOG),
        ("Crop log", CROP_LOG),
        ("Embeddings cache", EMBEDDINGS_CACHE)
    ]
    
    for name, path in logs_to_check:
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            days_ago = (datetime.now() - mtime).days
            
            if days_ago > 7:
                report.warning(f"{name} not updated in {days_ago} days")
            else:
                report.info_msg(f"{name}: Updated {days_ago} days ago")


def main():
    parser = argparse.ArgumentParser(description='Validate AI training data integrity')
    parser.add_argument('--fix', action='store_true',
                       help='Attempt to auto-fix some issues (e.g., recompute missing embeddings)')
    args = parser.parse_args()
    
    report = ValidationReport()
    
    
    # Run all validation checks
    selection_files = validate_selection_data(report)
    crop_files = validate_crop_data(report, selection_files)
    _filename_cache, missing_emb_files = validate_embeddings(report, selection_files, crop_files)
    validate_recent_activity(report)
    
    # Print report
    report.print_report()
    
    # Auto-fix if requested
    if args.fix and (missing_emb_files or report.has_errors()):
        if missing_emb_files:
            pass
    
    # Exit with appropriate code
    if report.has_errors():
        return 2
    if report.has_warnings():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

