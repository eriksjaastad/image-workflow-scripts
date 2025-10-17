#!/usr/bin/env python3
"""
Derive Sessions from Operation Events v1
=========================================
Derives session records directly from file operation events.

Session derivation rules:
- Session starts at first event after a gap ≥ GAP_MIN (default 5 min)
- Session ends at last event before a gap ≥ GAP_MIN
- Active time = sum of bounded inter-event gaps (max MAX_GAP_CONTRIB per gap)
- Files processed = sum of file_count for file_operation events
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import List, Dict, Any

# Load config
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "metrics_config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

GAP_MIN_MINUTES = CONFIG["metrics"]["gap_min_minutes"]
MAX_GAP_CONTRIB_SECONDS = CONFIG["metrics"]["max_gap_contrib_seconds"]
LOOKBACK_DAYS = CONFIG["lookbackDays"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "data" / "file_operations_logs" / "file_operations.log"
OUTPUT_DIR = PROJECT_ROOT / "data" / "snapshot" / "derived_sessions_v1"


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp to UTC datetime."""
    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    return dt.astimezone(timezone.utc)


def generate_session_id(script: str, start_ts: datetime, end_ts: datetime) -> str:
    """Generate stable session ID from canonical data."""
    canonical = f"{script}:{start_ts.isoformat()}:{end_ts.isoformat()}"
    return f"derived:{hashlib.md5(canonical.encode()).hexdigest()[:12]}"


def derive_sessions_for_day(events: List[Dict[str, Any]], day_str: str) -> List[Dict[str, Any]]:
    """Derive sessions from events for a single day."""
    # Group events by script
    by_script = defaultdict(list)
    for event in events:
        if event.get("type") == "file_operation":
            script = event.get("script", "unknown")
            by_script[script].append(event)
    
    sessions = []
    gap_threshold = timedelta(minutes=GAP_MIN_MINUTES)
    
    for script, script_events in by_script.items():
        # Sort by timestamp
        script_events.sort(key=lambda e: parse_timestamp(e["timestamp"]))
        
        if not script_events:
            continue
        
        # Build sessions
        current_session_events = [script_events[0]]
        
        for i in range(1, len(script_events)):
            prev_ts = parse_timestamp(script_events[i-1]["timestamp"])
            curr_ts = parse_timestamp(script_events[i]["timestamp"])
            gap = curr_ts - prev_ts
            
            if gap >= gap_threshold:
                # Finalize current session
                sessions.append(_build_session(script, current_session_events))
                current_session_events = [script_events[i]]
            else:
                current_session_events.append(script_events[i])
        
        # Finalize last session
        if current_session_events:
            sessions.append(_build_session(script, current_session_events))
    
    return sessions


def _build_session(script: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a single session record from events."""
    timestamps = [parse_timestamp(e["timestamp"]) for e in events]
    start_ts = min(timestamps)
    end_ts = max(timestamps)
    
    # Calculate active time with bounded gaps
    active_seconds = 0.0
    sorted_ts = sorted(timestamps)
    for i in range(len(sorted_ts) - 1):
        gap_seconds = (sorted_ts[i+1] - sorted_ts[i]).total_seconds()
        active_seconds += min(gap_seconds, MAX_GAP_CONTRIB_SECONDS)
    
    # Count files processed
    files_processed = sum(e.get("file_count", 1) for e in events if e.get("type") == "file_operation")
    
    # Count operations by type
    ops_by_type = defaultdict(int)
    projects_touched = set()
    for e in events:
        op_type = e.get("operation", "unknown")
        ops_by_type[op_type] += 1
        if "source_dir" in e:
            projects_touched.add(e["source_dir"])
    
    session_id = generate_session_id(script, start_ts, end_ts)
    
    return {
        "source": "derived_from_operation_events_v1",
        "script_id": script,
        "session_id": session_id,
        "start_ts_utc": start_ts.isoformat(),
        "end_ts_utc": end_ts.isoformat(),
        "active_seconds": round(active_seconds, 1),
        "files_processed": files_processed,
        "ops_by_type": dict(ops_by_type),
        "projects_touched": sorted(list(projects_touched)),
        "event_count": len(events),
        "params": {
            "gap_min": GAP_MIN_MINUTES,
            "max_gap_contrib": MAX_GAP_CONTRIB_SECONDS
        }
    }


def main():
    """Main entry point."""
    print(f"Deriving sessions from operation events (last {LOOKBACK_DAYS} days)...")
    print(f"Config: gap_min={GAP_MIN_MINUTES}min, max_gap_contrib={MAX_GAP_CONTRIB_SECONDS}s")
    
    # Read all events
    if not LOG_PATH.exists():
        print(f"ERROR: Log file not found: {LOG_PATH}")
        return
    
    events = []
    with open(LOG_PATH) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    
    print(f"Loaded {len(events)} events from log")
    
    # Filter to last N days and group by day
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    by_day = defaultdict(list)
    
    for event in events:
        if "timestamp" not in event:
            continue
        
        ts = parse_timestamp(event["timestamp"])
        if ts < cutoff:
            continue
        
        day_str = ts.strftime("%Y%m%d")
        by_day[day_str].append(event)
    
    print(f"Processing {len(by_day)} days...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_sessions = 0
    for day_str in sorted(by_day.keys()):
        day_events = by_day[day_str]
        sessions = derive_sessions_for_day(day_events, day_str)
        
        # Write output
        day_dir = OUTPUT_DIR / f"day={day_str}"
        day_dir.mkdir(parents=True, exist_ok=True)
        output_file = day_dir / "sessions.jsonl"
        
        with open(output_file, 'w') as f:
            for session in sessions:
                f.write(json.dumps(session) + '\n')
        
        total_sessions += len(sessions)
        print(f"  {day_str}: {len(sessions)} sessions from {len(day_events)} events")
    
    print(f"\nDone! {total_sessions} sessions written to {OUTPUT_DIR}")
    
    # Show sample
    if total_sessions > 0:
        print("\nSample sessions (first 3):")
        first_day = sorted(by_day.keys())[0]
        sample_file = OUTPUT_DIR / f"day={first_day}" / "sessions.jsonl"
        with open(sample_file) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                session = json.loads(line)
                print(f"  {session['session_id']}: {session['script_id']}, "
                      f"{session['active_seconds']}s, {session['files_processed']} files")


if __name__ == "__main__":
    main()

