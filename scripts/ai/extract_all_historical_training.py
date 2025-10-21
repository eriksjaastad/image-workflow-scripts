#!/usr/bin/env python3
"""
Extract training data from ALL historical projects.

This script processes multiple completed projects to generate training data
for the AI ranking model. Each project should have:
- A raw/source directory with all original images
- A final directory with Erik's selected images

The script:
1. Groups raw images by timestamp
2. Finds which images were selected (present in final directory)
3. Creates selection training data (chosen vs rejected images)
4. Identifies crops (files with different modification times)

Usage:
    python scripts/ai/extract_all_historical_training.py
    
Output:
    - data/training/historical_selection_log.csv
    - data/training/historical_crop_log.csv
    - data/training/historical_extraction_report.json
"""

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Base directory
BASE_DIR = Path("/Users/eriksjaastad/projects/Eros Mate")
TRAINING_DATA_DIR = BASE_DIR / "training data"
OUTPUT_DIR = BASE_DIR / "data/training"

# Project configurations from timesheet.csv
# Format: (project_name, raw_dir, final_dir)
PROJECTS = [
    # Already processed:
    # ("mojo-1", TRAINING_DATA_DIR / "mojo1", TRAINING_DATA_DIR / "mojo1_final"),
    
    # To process (adjust these paths based on your actual directory structure):
    # Note: Some of these might need path adjustments - the script will report which ones are found
]

# Since we don't know the exact directory structure, let's scan for likely candidates
def discover_projects() -> List[Tuple[str, Path, Path]]:
    """
    Discover project directories by looking for patterns.
    Returns list of (project_name, raw_dir, final_dir) tuples.
    """
    print("🔍 Discovering projects in training data directory...")
    projects = []
    
    # Look for directories that might be raw vs final pairs
    if TRAINING_DATA_DIR.exists():
        all_dirs = [d for d in TRAINING_DATA_DIR.iterdir() if d.is_dir()]
        
        # Look for patterns like "name" and "name_final"
        dir_names = [d.name for d in all_dirs]
        
        for dir_name in dir_names:
            if not dir_name.endswith('_final'):
                # Check if there's a corresponding _final directory
                final_name = f"{dir_name}_final"
                if final_name in dir_names:
                    raw_dir = TRAINING_DATA_DIR / dir_name
                    final_dir = TRAINING_DATA_DIR / final_name
                    
                    # Verify they contain PNG files
                    raw_pngs = list(raw_dir.rglob("*.png"))
                    final_pngs = list(final_dir.rglob("*.png"))
                    
                    if raw_pngs and final_pngs:
                        projects.append((dir_name, raw_dir, final_dir))
                        print(f"   Found: {dir_name} ({len(raw_pngs)} raw → {len(final_pngs)} final)")
    
    return projects


def parse_filename(filename: str) -> Optional[Dict]:
    """Extract timestamp and stage from filename."""
    pattern = r'(\d{8}_\d{6})_stage(\d+(?:\.\d+)?)'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'timestamp': match.group(1),
            'stage': float(match.group(2))
        }
    return None


def process_project(project_name: str, raw_dir: Path, final_dir: Path) -> Dict:
    """
    Process a single project and extract training data.
    
    Returns:
        Dict with selection_entries, crop_entries, and statistics
    """
    print(f"\n{'='*80}")
    print(f"Processing: {project_name}")
    print(f"{'='*80}")
    print(f"Raw dir: {raw_dir}")
    print(f"Final dir: {final_dir}")
    
    # Group raw images by timestamp
    raw_groups = defaultdict(list)
    raw_files = {}  # filename -> (path, mtime)
    
    for img_path in raw_dir.rglob("*.png"):
        parsed = parse_filename(img_path.name)
        if parsed:
            raw_groups[parsed['timestamp']].append({
                'path': img_path,
                'filename': img_path.name,
                'stage': parsed['stage']
            })
        raw_files[img_path.name] = (img_path, img_path.stat().st_mtime)
    
    print(f"   Raw: {len(raw_groups)} groups, {len(raw_files)} total images")
    
    # Find selections and crops
    selection_entries = []
    crop_entries = []
    
    for final_path in final_dir.rglob("*.png"):
        parsed = parse_filename(final_path.name)
        if not parsed:
            continue
        
        timestamp = parsed['timestamp']
        final_mtime = final_path.stat().st_mtime
        
        # Check if this timestamp has a group in raw
        if timestamp in raw_groups:
            raw_group = raw_groups[timestamp]
            
            # Create selection entry if there were alternatives
            if len(raw_group) > 1:
                # Find the chosen image and the rejected ones
                chosen_stage = parsed['stage']
                losers = []
                loser_stages = []
                
                for img in raw_group:
                    if img['filename'] != final_path.name:
                        losers.append(str(img['path']))
                        loser_stages.append(img['stage'])
                
                if losers:
                    all_stages = [chosen_stage] + loser_stages
                    is_anomaly = chosen_stage < max(all_stages)
                    
                    selection_entries.append({
                        'session_id': f'{project_name}_historical',
                        'set_id': timestamp,
                        'chosen_path': str(final_path),
                        'chosen_stage': chosen_stage,
                        'neg_paths': json.dumps(losers),
                        'neg_stages': loser_stages,
                        'timestamp': timestamp,
                        'is_anomaly': is_anomaly
                    })
            
            # Check if image was cropped (mtime differs)
            if final_path.name in raw_files:
                raw_path, raw_mtime = raw_files[final_path.name]
                time_diff = abs(final_mtime - raw_mtime)
                
                if time_diff > 1:  # More than 1 second difference = cropped
                    crop_entries.append({
                        'session_id': f'{project_name}_historical',
                        'set_id': timestamp,
                        'directory': project_name,
                        'chosen_path': str(final_path),
                        'chosen_stage': parsed['stage'],
                        'timestamp': timestamp,
                        'time_diff_days': time_diff / 86400,
                        'raw_mtime': datetime.fromtimestamp(raw_mtime, tz=timezone.utc).isoformat(),
                        'final_mtime': datetime.fromtimestamp(final_mtime, tz=timezone.utc).isoformat()
                    })
    
    anomaly_count = sum(1 for e in selection_entries if e['is_anomaly'])
    
    print(f"   ✅ Selections: {len(selection_entries)} ({anomaly_count} anomalies)")
    print(f"   ✅ Crops: {len(crop_entries)}")
    
    return {
        'project': project_name,
        'selection_entries': selection_entries,
        'crop_entries': crop_entries,
        'stats': {
            'raw_images': len(raw_files),
            'raw_groups': len(raw_groups),
            'selections': len(selection_entries),
            'crops': len(crop_entries),
            'anomalies': anomaly_count,
            'anomaly_rate': anomaly_count / len(selection_entries) if selection_entries else 0
        }
    }


def main():
    print("=" * 80)
    print("HISTORICAL PROJECT TRAINING DATA EXTRACTION")
    print("=" * 80)
    print()
    
    # Discover projects
    projects = discover_projects()
    
    if not projects:
        print("⚠️  No projects found!")
        print()
        print("Expected directory structure:")
        print("  training data/")
        print("    project_name/          (raw images)")
        print("    project_name_final/    (selected images)")
        print()
        print("Please verify your directory structure and try again.")
        return
    
    print(f"\n📊 Found {len(projects)} projects to process")
    print()
    
    # Process all projects
    all_results = []
    all_selections = []
    all_crops = []
    
    for project_name, raw_dir, final_dir in projects:
        result = process_project(project_name, raw_dir, final_dir)
        all_results.append(result)
        all_selections.extend(result['selection_entries'])
        all_crops.extend(result['crop_entries'])
    
    # Write combined output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Selection log
    selection_path = OUTPUT_DIR / "historical_selection_log.csv"
    print(f"\n📝 Writing selection log: {selection_path}")
    
    with selection_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'session_id', 'set_id', 'chosen_path', 'neg_paths', 'timestamp'
        ])
        writer.writeheader()
        
        for entry in all_selections:
            writer.writerow({
                'session_id': entry['session_id'],
                'set_id': entry['set_id'],
                'chosen_path': entry['chosen_path'],
                'neg_paths': entry['neg_paths'],
                'timestamp': entry['timestamp']
            })
    
    print(f"   Wrote {len(all_selections)} selection decisions")
    
    # Crop log
    crop_path = OUTPUT_DIR / "historical_crop_log.csv"
    print(f"\n📝 Writing crop log: {crop_path}")
    
    with crop_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'session_id', 'set_id', 'directory', 'chosen_path',
            'chosen_stage', 'timestamp', 'time_diff_days', 'raw_mtime', 'final_mtime'
        ])
        writer.writeheader()
        writer.writerows(all_crops)
    
    print(f"   Wrote {len(all_crops)} crop decisions")
    
    # Generate report
    report_path = OUTPUT_DIR / "historical_extraction_report.json"
    print(f"\n📝 Writing report: {report_path}")
    
    total_anomalies = sum(r['stats']['anomalies'] for r in all_results)
    
    report = {
        'extraction_date': datetime.now(timezone.utc).isoformat(),
        'projects_processed': len(all_results),
        'total_selections': len(all_selections),
        'total_crops': len(all_crops),
        'total_anomalies': total_anomalies,
        'overall_anomaly_rate': total_anomalies / len(all_selections) if all_selections else 0,
        'projects': [r['stats'] for r in all_results]
    }
    
    with report_path.open('w') as f:
        json.dump(report, f, indent=2)
    
    # Summary
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Projects processed: {len(all_results)}")
    print(f"✅ Selection decisions: {len(all_selections)}")
    print(f"✅ Crop decisions: {len(all_crops)}")
    print(f"⚠️  Anomaly cases: {total_anomalies} ({total_anomalies/len(all_selections)*100:.1f}%)")
    print()
    print("📁 Files created:")
    print(f"   - {selection_path}")
    print(f"   - {crop_path}")
    print(f"   - {report_path}")
    print()
    print("🎯 Next step: Merge with existing selection_only_log.csv and retrain!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
        print("\n" + "="*60)
        print("✅ SCRIPT COMPLETED SUCCESSFULLY")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print("❌ SCRIPT FAILED")
        print(f"Error: {e}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*60)
        raise

