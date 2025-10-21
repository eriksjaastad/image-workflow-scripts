#!/usr/bin/env python3
"""
Extract training data from Mojo 1 project (historical data extraction)

This script:
1. Groups all raw images in mojo1/ by timestamp (finds image groups)
2. Identifies which images Erik selected (they're in mojo1_final/)
3. Creates selection training data: (chosen_image, rejected_images)
4. Identifies which selected images Erik cropped (modified in Oct 2025)
5. Outputs training data compatible with existing logs

Usage:
    python scripts/ai/extract_mojo1_training.py
    
Output:
    - data/training/mojo1_selection_log.csv (selection decisions)
    - data/training/mojo1_crop_log.csv (crop decisions)
    - data/training/mojo1_extraction_report.json (stats and metadata)
"""

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Project configuration
RAW_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/training data/mojo1")
FINAL_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/training data/mojo1_final")
OUTPUT_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/data/training")

# Project dates (from timesheet.csv)
PROJECT_START = datetime(2025, 10, 1, tzinfo=timezone.utc)
PROJECT_END = datetime(2025, 10, 11, 23, 59, 59, tzinfo=timezone.utc)


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Extract timestamp and stage from filename.
    
    Patterns:
        20250708_150059_stage2_upscaled.png
        20250725_183405_stage1.5_face_swapped.png
        20250820_063708_stage1_generated.png
        20250706_023542_stage3_enhanced.png
    """
    # Match: YYYYMMDD_HHMMSS_stageX.X_suffix.png
    pattern = r'(\d{8}_\d{6})_stage(\d+(?:\.\d+)?)'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'timestamp': match.group(1),
            'stage': float(match.group(2))
        }
    return None


def group_raw_images() -> Dict[str, List[Dict]]:
    """
    Group all raw images by timestamp.
    
    Returns:
        Dict mapping timestamp -> list of image info dicts
    """
    print(f"📂 Scanning raw directory: {RAW_DIR}")
    groups = defaultdict(list)
    
    raw_files = list(RAW_DIR.rglob("*.png"))
    print(f"   Found {len(raw_files)} raw PNG files")
    
    for img_path in raw_files:
        parsed = parse_filename(img_path.name)
        if parsed:
            groups[parsed['timestamp']].append({
                'path': img_path,
                'filename': img_path.name,
                'stage': parsed['stage']
            })
    
    print(f"   Grouped into {len(groups)} unique timestamp groups")
    return groups


def find_winners(raw_groups: Dict) -> Dict[str, Dict]:
    """
    For each group, find which image (if any) Erik selected.
    Selected images are in mojo1_final/.
    
    Returns:
        Dict mapping timestamp -> winner info
    """
    print(f"\n📂 Scanning final directory: {FINAL_DIR}")
    final_files = list(FINAL_DIR.rglob("*.png"))
    print(f"   Found {len(final_files)} final PNG files")
    
    winners = {}
    matched_count = 0
    
    for final_img in final_files:
        parsed = parse_filename(final_img.name)
        if parsed:
            timestamp = parsed['timestamp']
            
            if timestamp in raw_groups:
                # Found a match!
                matched_count += 1
                
                # Get file modification time (when Erik cropped it)
                mtime = datetime.fromtimestamp(final_img.stat().st_mtime, tz=timezone.utc)
                
                # Extract original date from filename
                # Format: YYYYMMDD_HHMMSS
                filename_date = datetime.strptime(timestamp[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
                
                # If file was modified significantly after its filename date, it was cropped
                # Most raw files are from July/August, crops happened in October
                days_after = (mtime - filename_date).days
                was_cropped = days_after > 7  # Modified more than a week after original date
                
                winners[timestamp] = {
                    'winner_filename': final_img.name,
                    'winner_path': final_img,
                    'winner_stage': parsed['stage'],
                    'raw_group': raw_groups[timestamp],
                    'mtime': mtime,
                    'filename_date': filename_date,
                    'days_after': days_after,
                    'was_cropped': was_cropped
                }
    
    print(f"   Matched {matched_count} selections")
    print(f"   Cropped (modified >7 days after creation): {sum(1 for w in winners.values() if w['was_cropped'])}")
    
    return winners


def create_selection_training_data(winners: Dict) -> List[Dict]:
    """
    Create selection training entries: (winner, losers) pairs.
    
    Format matches existing selection_only_log.csv:
        session_id, set_id, chosen_path, neg_paths, timestamp
    """
    entries = []
    session_id = "mojo1_historical_extraction"
    
    for timestamp, data in winners.items():
        winner_filename = data['winner_filename']
        winner_stage = data['winner_stage']
        
        # Find losers (images in same group that weren't selected)
        losers = []
        loser_stages = []
        
        for img_info in data['raw_group']:
            if img_info['filename'] != winner_filename:
                losers.append(str(img_info['path']))
                loser_stages.append(img_info['stage'])
        
        # Only create entry if there were alternatives to choose from
        if losers:
            entries.append({
                'session_id': session_id,
                'set_id': timestamp,
                'chosen_path': str(data['winner_path']),
                'chosen_stage': winner_stage,
                'neg_paths': json.dumps(losers),  # JSON array as string
                'neg_stages': loser_stages,
                'timestamp': timestamp,
                'is_anomaly': winner_stage < max([img['stage'] for img in data['raw_group']]),
            })
    
    return entries


def create_crop_training_data(winners: Dict) -> List[Dict]:
    """
    Create crop training entries for images Erik cropped.
    
    We can't extract crop coordinates (they're baked into the cropped image),
    but we can flag which images were cropped for future analysis.
    """
    entries = []
    session_id = "mojo1_historical_extraction"
    
    for timestamp, data in winners.items():
        if data['was_cropped']:
            entries.append({
                'session_id': session_id,
                'set_id': timestamp,
                'directory': 'mojo1',
                'chosen_path': str(data['winner_path']),
                'chosen_stage': data['winner_stage'],
                'timestamp': timestamp,
                'note': 'Crop coordinates not available (image already cropped)',
                'mtime': data['mtime'].isoformat()
            })
    
    return entries


def write_selection_log(entries: List[Dict], output_path: Path):
    """Write selection training data to CSV."""
    print(f"\n📝 Writing selection log: {output_path}")
    
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'session_id', 'set_id', 'chosen_path', 'neg_paths', 'timestamp'
        ])
        writer.writeheader()
        
        for entry in entries:
            writer.writerow({
                'session_id': entry['session_id'],
                'set_id': entry['set_id'],
                'chosen_path': entry['chosen_path'],
                'neg_paths': entry['neg_paths'],
                'timestamp': entry['timestamp']
            })
    
    print(f"   Wrote {len(entries)} selection decisions")


def write_crop_log(entries: List[Dict], output_path: Path):
    """Write crop training data to CSV."""
    print(f"\n📝 Writing crop log: {output_path}")
    
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'session_id', 'set_id', 'directory', 'chosen_path', 
            'chosen_stage', 'timestamp', 'note', 'mtime'
        ])
        writer.writeheader()
        
        for entry in entries:
            writer.writerow(entry)
    
    print(f"   Wrote {len(entries)} crop decisions (metadata only)")


def generate_report(raw_groups: Dict, winners: Dict, 
                   selection_entries: List, crop_entries: List) -> Dict:
    """Generate extraction statistics report."""
    
    # Analyze anomalies (when Erik chose lower stage)
    anomalies = [e for e in selection_entries if e['is_anomaly']]
    
    # Count stage preferences
    stage_counts = defaultdict(int)
    for entry in selection_entries:
        stage_counts[entry['chosen_stage']] += 1
    
    report = {
        'extraction_date': datetime.now(timezone.utc).isoformat(),
        'project': 'mojo-1',
        'project_dates': {
            'start': PROJECT_START.isoformat(),
            'end': PROJECT_END.isoformat()
        },
        'raw_data': {
            'total_images': sum(len(group) for group in raw_groups.values()),
            'total_groups': len(raw_groups),
            'directory': str(RAW_DIR)
        },
        'final_data': {
            'total_selections': len(winners),
            'total_cropped': sum(1 for w in winners.values() if w['was_cropped']),
            'directory': str(FINAL_DIR)
        },
        'training_data': {
            'selection_entries': len(selection_entries),
            'crop_entries': len(crop_entries),
            'anomaly_cases': len(anomalies),
            'anomaly_rate': len(anomalies) / len(selection_entries) if selection_entries else 0
        },
        'stage_distribution': dict(stage_counts),
        'sample_anomalies': [
            {
                'timestamp': a['set_id'],
                'chosen_stage': a['chosen_stage'],
                'available_stages': [img['stage'] for img in winners[a['set_id']]['raw_group']],
                'max_stage': max(img['stage'] for img in winners[a['set_id']]['raw_group'])
            }
            for a in anomalies[:10]  # First 10 examples
        ]
    }
    
    return report


def main():
    print("=" * 70)
    print("MOJO 1 TRAINING DATA EXTRACTION")
    print("=" * 70)
    print(f"Project: Mojo 1 (Oct 1-11, 2025)")
    print(f"Raw dir: {RAW_DIR}")
    print(f"Final dir: {FINAL_DIR}")
    print()
    
    # Step 1: Group raw images by timestamp
    raw_groups = group_raw_images()
    
    # Step 2: Find which images Erik selected
    winners = find_winners(raw_groups)
    
    # Step 3: Create selection training data
    print(f"\n🧠 Creating selection training data...")
    selection_entries = create_selection_training_data(winners)
    print(f"   Created {len(selection_entries)} selection entries")
    print(f"   Anomalies (chose lower stage): {sum(1 for e in selection_entries if e['is_anomaly'])}")
    
    # Step 4: Create crop training data
    print(f"\n✂️  Creating crop training data...")
    crop_entries = create_crop_training_data(winners)
    print(f"   Created {len(crop_entries)} crop entries")
    
    # Step 5: Write output files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    selection_path = OUTPUT_DIR / "mojo1_selection_log.csv"
    crop_path = OUTPUT_DIR / "mojo1_crop_log.csv"
    report_path = OUTPUT_DIR / "mojo1_extraction_report.json"
    
    write_selection_log(selection_entries, selection_path)
    write_crop_log(crop_entries, crop_path)
    
    # Step 6: Generate report
    print(f"\n📊 Generating extraction report...")
    report = generate_report(raw_groups, winners, selection_entries, crop_entries)
    
    with report_path.open('w') as f:
        json.dump(report, f, indent=2)
    print(f"   Report saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✅ Selection decisions: {len(selection_entries)}")
    print(f"✅ Crop decisions: {len(crop_entries)}")
    print(f"⚠️  Anomaly cases: {sum(1 for e in selection_entries if e['is_anomaly'])}")
    print(f"   ({sum(1 for e in selection_entries if e['is_anomaly'])/len(selection_entries)*100:.1f}% of selections)")
    print()
    print(f"📁 Files created:")
    print(f"   - {selection_path}")
    print(f"   - {crop_path}")
    print(f"   - {report_path}")
    print()
    print("🎯 Next steps:")
    print("   1. Merge with existing selection_only_log.csv")
    print("   2. Retrain ranking model with combined data")
    print("   3. Test if anomaly detection improves")
    print("=" * 70)


if __name__ == "__main__":
    main()

