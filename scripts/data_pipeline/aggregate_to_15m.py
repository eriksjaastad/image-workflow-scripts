#!/usr/bin/env python3
"""
15-Minute Bin Aggregator
========================
Pre-aggregates raw file operation logs into 15-minute UTC-aligned bins.

Input:
- data/file_operations_logs/file_operations_YYYYMMDD.log (JSONL)
- data/log_archives/file_operations_YYYYMMDD.log.gz (JSONL)

Output:
- data/aggregates/daily/day=YYYYMMDD/agg_15m.jsonl
  Each line is a 15-minute bin with schema:
  {
    "bin_ts_utc": "2025-10-17T16:00:00Z",  # UTC bin start
    "bin_version": 1,                       # For corrections/reprocessing
    "project_id": "mojo1",                  # From path hints
    "script_id": "01_web_image_selector",   # Script name
    "operation": "move",                    # Operation type
    "dest_category": "crop",                # Destination category (for moves)
    "file_count": 15,                       # PNG files only
    "file_count_total": 30,                 # All files (PNG + YAML)
    "event_count": 15,                      # Number of operations
    "work_seconds": 875.3,                  # Derived work time (break-aware)
    "first_event_ts": "2025-10-17T16:00:05Z",
    "last_event_ts": "2025-10-17T16:14:32Z",
    "dedupe_key": "20251017T160000Z_mojo1_01_web_image_selector_move_crop_v1",
    "tz_source": "local",                   # Timestamp source timezone
    "created_at": "2025-10-17T18:30:00Z"    # When this bin was created
  }

Features:
- Idempotent: Re-running on same day produces same output (atomic write)
- UTC-aligned: All bins start at :00, :15, :30, :45
- Break-aware: Uses same work_seconds derivation as dashboard
- Stable keys: dedupe_key ensures no duplicates across runs
- Correction support: bin_version allows reprocessing
"""

import gzip
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.utils.companion_file_utils import get_file_operation_metrics


def parse_timestamp_to_utc(ts_str: str) -> Optional[datetime]:
    """Parse timestamp string to UTC datetime."""
    if not ts_str:
        return None
    try:
        # Handle 'Z' suffix
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        dt = datetime.fromisoformat(ts_str)
        # If naive, assume local time (will need tz_source tracking)
        if dt.tzinfo is None:
            # For now, treat as UTC (this matches current system behavior)
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception as e:
        print(f"Warning: Failed to parse timestamp '{ts_str}': {e}")
        return None


def floor_to_15min(dt: datetime) -> datetime:
    """Floor datetime to nearest 15-minute boundary."""
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def extract_project_id(record: Dict[str, Any]) -> str:
    """Extract project ID from file operation record paths."""
    # Try to infer from directory paths
    for path_key in ['source_dir', 'dest_dir', 'working_dir']:
        path = record.get(path_key, '')
        if path and isinstance(path, str):
            # Common patterns: "mojo1/", "character_group_1/", etc.
            # Extract first directory component
            parts = path.strip('/').split('/')
            if parts and parts[0]:
                return parts[0]
    return 'unknown'


def extract_dest_category(record: Dict[str, Any]) -> str:
    """Extract destination category for move operations."""
    if record.get('operation') != 'move':
        return ''
    
    dest = record.get('dest_dir', '').lower()
    if 'crop' in dest:
        return 'crop'
    elif 'selected' in dest:
        return 'selected'
    elif 'delete' in dest:
        return 'delete'
    elif 'reviewed' in dest:
        return 'reviewed'
    else:
        return 'other'


def count_png_files(record: Dict[str, Any]) -> tuple[int, int]:
    """Count PNG files and total files in a record.
    
    Returns:
        (png_count, total_count)
    """
    files_list = record.get('files', [])
    if files_list:
        png_count = sum(1 for f in files_list if isinstance(f, str) and f.lower().endswith('.png'))
        return png_count, len(files_list)
    
    # Fallback to file_count (assume all PNG for aggregated records)
    file_count = int(record.get('file_count', 0) or 0)
    return file_count, file_count


def load_raw_logs_for_day(data_dir: Path, day_str: str) -> List[Dict[str, Any]]:
    """Load all raw file operation logs for a specific day (YYYYMMDD format).
    
    Args:
        data_dir: Root data directory
        day_str: Day in YYYYMMDD format
        
    Returns:
        List of file operation records
    """
    records = []
    
    # Try current logs
    log_path = data_dir / 'file_operations_logs' / f'file_operations_{day_str}.log'
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('type') == 'file_operation':
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Failed to read {log_path}: {e}")
    
    # Try archived logs
    archive_path = data_dir / 'log_archives' / f'file_operations_{day_str}_archived.log.gz'
    if not archive_path.exists():
        # Try without _archived suffix
        archive_path = data_dir / 'log_archives' / f'file_operations_{day_str}.log.gz'
    
    if archive_path.exists():
        try:
            with gzip.open(archive_path, 'rt') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('type') == 'file_operation':
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Failed to read {archive_path}: {e}")
    
    return records


def aggregate_to_bins(records: List[Dict[str, Any]], bin_version: int = 1) -> List[Dict[str, Any]]:
    """Aggregate file operation records into 15-minute bins.
    
    Args:
        records: Raw file operation records
        bin_version: Version number for this aggregation run
        
    Returns:
        List of aggregated bin records
    """
    # Group by: (bin_ts_utc, project_id, script_id, operation, dest_category)
    bins: Dict[tuple, Dict[str, Any]] = defaultdict(lambda: {
        'events': [],
        'file_count': 0,
        'file_count_total': 0,
        'event_count': 0
    })
    
    for record in records:
        # Parse timestamp
        ts_str = record.get('timestamp') or record.get('timestamp_str', '')
        dt = parse_timestamp_to_utc(ts_str)
        if not dt:
            continue
        
        # Floor to 15-minute bin
        bin_ts = floor_to_15min(dt)
        
        # Extract grouping keys
        project_id = extract_project_id(record)
        script_id = record.get('script', 'unknown')
        operation = record.get('operation', 'unknown')
        dest_category = extract_dest_category(record)
        
        # Create bin key
        key = (bin_ts.isoformat(), project_id, script_id, operation, dest_category)
        
        # Accumulate
        png_count, total_count = count_png_files(record)
        bins[key]['events'].append(record)
        bins[key]['file_count'] += png_count
        bins[key]['file_count_total'] += total_count
        bins[key]['event_count'] += 1
    
    # Convert to output format
    now = datetime.now(timezone.utc).isoformat()
    output_bins = []
    
    for key, data in bins.items():
        bin_ts_str, project_id, script_id, operation, dest_category = key
        events = data['events']
        
        # Calculate work_seconds from events (break-aware)
        try:
            # Prepare events for metrics calculation
            events_for_metrics = []
            for event in events:
                event_copy = dict(event)
                ts = event_copy.get('timestamp')
                if isinstance(ts, datetime):
                    event_copy['timestamp'] = ts.isoformat()
                elif not isinstance(ts, str):
                    ts_str = event_copy.get('timestamp_str')
                    if isinstance(ts_str, str):
                        event_copy['timestamp'] = ts_str
                events_for_metrics.append(event_copy)
            
            metrics = get_file_operation_metrics(events_for_metrics)
            work_seconds = float(metrics.get('work_time_minutes', 0) or 0) * 60.0
        except Exception as e:
            print(f"Warning: Failed to calculate work_seconds for bin {bin_ts_str}: {e}")
            work_seconds = 0.0
        
        # Get first and last event timestamps
        event_timestamps = []
        for event in events:
            ts_str = event.get('timestamp') or event.get('timestamp_str', '')
            dt = parse_timestamp_to_utc(ts_str)
            if dt:
                event_timestamps.append(dt)
        
        first_event_ts = min(event_timestamps).isoformat() if event_timestamps else bin_ts_str
        last_event_ts = max(event_timestamps).isoformat() if event_timestamps else bin_ts_str
        
        # Create dedupe key
        dest_suffix = f"_{dest_category}" if dest_category else ""
        dedupe_key = f"{bin_ts_str.replace(':', '').replace('-', '')}_{project_id}_{script_id}_{operation}{dest_suffix}_v{bin_version}"
        
        output_bins.append({
            'bin_ts_utc': bin_ts_str,
            'bin_version': bin_version,
            'project_id': project_id,
            'script_id': script_id,
            'operation': operation,
            'dest_category': dest_category,
            'file_count': data['file_count'],
            'file_count_total': data['file_count_total'],
            'event_count': data['event_count'],
            'work_seconds': round(work_seconds, 2),
            'first_event_ts': first_event_ts,
            'last_event_ts': last_event_ts,
            'dedupe_key': dedupe_key,
            'tz_source': 'local',  # Current system uses local time
            'created_at': now
        })
    
    # Sort by bin timestamp
    output_bins.sort(key=lambda x: x['bin_ts_utc'])
    
    return output_bins


def write_bins_atomic(output_path: Path, bins: List[Dict[str, Any]]) -> None:
    """Write bins to file atomically (temp file + rename).
    
    Args:
        output_path: Target output file
        bins: List of bin records to write
    """
    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        for bin_record in bins:
            json.dump(bin_record, tmp_file)
            tmp_file.write('\n')
    
    # Atomic rename
    shutil.move(str(tmp_path), str(output_path))
    print(f"✓ Wrote {len(bins)} bins to {output_path}")


def aggregate_day(data_dir: Path, day_str: str, bin_version: int = 1, dry_run: bool = False) -> int:
    """Aggregate a single day's raw logs into 15-minute bins.
    
    Args:
        data_dir: Root data directory
        day_str: Day in YYYYMMDD format
        bin_version: Version number for corrections
        dry_run: If True, don't write output
        
    Returns:
        Number of bins created
    """
    print(f"\nProcessing day: {day_str}")
    
    # Load raw logs
    records = load_raw_logs_for_day(data_dir, day_str)
    if not records:
        print(f"  No records found for {day_str}")
        return 0
    
    print(f"  Loaded {len(records)} records")
    
    # Aggregate to bins
    bins = aggregate_to_bins(records, bin_version=bin_version)
    print(f"  Created {len(bins)} bins")
    
    if dry_run:
        print(f"  [DRY RUN] Would write to: data/aggregates/daily/day={day_str}/agg_15m.jsonl")
        # Show sample
        if bins:
            print("\n  Sample bin:")
            print(f"    {json.dumps(bins[0], indent=4)}")
        return len(bins)
    
    # Write output
    output_path = data_dir / 'aggregates' / 'daily' / f'day={day_str}' / 'agg_15m.jsonl'
    write_bins_atomic(output_path, bins)
    
    return len(bins)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Aggregate raw file operation logs into 15-minute bins'
    )
    parser.add_argument(
        '--day',
        type=str,
        help='Day to process (YYYYMMDD format). If not specified, processes last 7 days.'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of recent days to process (default: 7)'
    )
    parser.add_argument(
        '--bin-version',
        type=int,
        default=1,
        help='Bin version number for corrections (default: 1)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without writing files'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Data directory path (default: project_root/data)'
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / 'data'
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Data directory: {data_dir}")
    print(f"Bin version: {args.bin_version}")
    
    if args.day:
        # Process single day
        days = [args.day]
    else:
        # Process last N days
        from datetime import timedelta
        today = datetime.now().date()
        days = []
        for i in range(args.days):
            day = today - timedelta(days=i)
            days.append(day.strftime('%Y%m%d'))
    
    # Process each day
    total_bins = 0
    for day_str in days:
        try:
            bin_count = aggregate_day(data_dir, day_str, bin_version=args.bin_version, dry_run=args.dry_run)
            total_bins += bin_count
        except Exception as e:
            print(f"Error processing {day_str}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Total bins created: {total_bins}")
    print("✓ Aggregation complete")


if __name__ == '__main__':
    main()

