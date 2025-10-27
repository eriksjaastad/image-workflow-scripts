#!/usr/bin/env python3
"""
Build Daily Aggregates v1
==========================
Materializes daily aggregates from file operation events.

Output: snapshot/daily_aggregates_v1/day=YYYYMMDD.json
Contents per day:
- Total operations by script and by operation type
- First/last operation timestamps
- Total files processed
- Distinct projects touched
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Load config
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "metrics_config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshot" / "operation_events_v1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "snapshot" / "daily_aggregates_v1"


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp to UTC datetime."""
    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    return dt.astimezone(timezone.utc)


def build_aggregate_for_day(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build aggregate statistics for a single day."""
    by_script = defaultdict(lambda: {
        "operations": defaultdict(int),
        "files_processed": 0,
        "event_count": 0,
        "timestamps": []
    })
    
    by_operation = defaultdict(int)
    projects = set()
    all_timestamps = []
    
    for event in events:
        # Support both v1 snapshot format and legacy format
        event_type = event.get("event_type") or event.get("type")
        
        if event_type == "file_operation":
            script = event.get("script_id") or event.get("script", "unknown")
            operation = event.get("operation", "unknown")
            file_count = event.get("file_count") or 1
            timestamp = event.get("ts_utc") or event.get("timestamp")
            
            # Per-script stats
            by_script[script]["operations"][operation] += 1
            by_script[script]["files_processed"] += file_count
            by_script[script]["event_count"] += 1
            if timestamp:
                ts = parse_timestamp(timestamp)
                by_script[script]["timestamps"].append(ts)
                all_timestamps.append(ts)
            
            # Global stats
            by_operation[operation] += 1
            
            if "source_dir" in event:
                projects.add(event["source_dir"])
    
    # Calculate time ranges per script
    script_stats = {}
    for script, data in by_script.items():
        if data["timestamps"]:
            script_stats[script] = {
                "operations": dict(data["operations"]),
                "files_processed": data["files_processed"],
                "event_count": data["event_count"],
                "first_op_ts": min(data["timestamps"]).isoformat(),
                "last_op_ts": max(data["timestamps"]).isoformat()
            }
    
    return {
        "by_script": script_stats,
        "by_operation": dict(by_operation),
        "projects_touched": sorted(list(projects)),
        "total_files_processed": sum(s["files_processed"] for s in script_stats.values()),
        "total_events": sum(s["event_count"] for s in script_stats.values()),
        "first_op_ts": min(all_timestamps).isoformat() if all_timestamps else None,
        "last_op_ts": max(all_timestamps).isoformat() if all_timestamps else None
    }


def main():
    """Main entry point."""
    print("Building daily aggregates from operation_events_v1 snapshot...")
    
    # Read from snapshot (which contains all historical data)
    if not SNAPSHOT_DIR.exists():
        print(f"ERROR: Snapshot not found: {SNAPSHOT_DIR}")
        print("   Run extract_operation_events_v1.py first!")
        return
    
    # Load events from all day partitions
    by_day = defaultdict(list)
    day_dirs = sorted(SNAPSHOT_DIR.glob("day=*"))
    
    for day_dir in day_dirs:
        day_str = day_dir.name.split("=")[1]
        events_file = day_dir / "events.jsonl"
        
        if not events_file.exists():
            continue
        
        with open(events_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    by_day[day_str].append(event)
                except json.JSONDecodeError:
                    continue
    
    total_events = sum(len(evts) for evts in by_day.values())
    print(f"Loaded {total_events} events from {len(by_day)} days")
    print(f"Processing {len(by_day)} days...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for day_str in sorted(by_day.keys()):
        day_events = by_day[day_str]
        aggregate = build_aggregate_for_day(day_events)
        
        # Write output
        day_dir = OUTPUT_DIR / f"day={day_str}"
        day_dir.mkdir(parents=True, exist_ok=True)
        output_file = day_dir / "aggregate.json"
        
        with open(output_file, 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        print(f"  {day_str}: {aggregate['total_events']} events, "
              f"{aggregate['total_files_processed']} files, "
              f"{len(aggregate['by_script'])} scripts")
    
    print(f"\nDone! Aggregates written to {OUTPUT_DIR}")
    
    # Show sample
    if by_day:
        print("\nSample aggregate (most recent day):")
        last_day = sorted(by_day.keys())[-1]
        sample_file = OUTPUT_DIR / f"day={last_day}" / "aggregate.json"
        with open(sample_file) as f:
            sample = json.load(f)
        print(f"  Day: {last_day}")
        print(f"  Total events: {sample['total_events']}")
        print(f"  Total files: {sample['total_files_processed']}")
        print(f"  Scripts: {list(sample['by_script'].keys())}")


if __name__ == "__main__":
    main()

