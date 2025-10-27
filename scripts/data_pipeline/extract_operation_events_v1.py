#!/usr/bin/env python3
"""
Extract Operation Events v1
============================
Extracts and normalizes file operation events from raw logs to snapshot format.

Reads from:
- data/file_operations_logs/*.log (current + dated)
- data/log_archives/*.gz (compressed archives)

Outputs to:
- snapshot/operation_events_v1/day=YYYYMMDD/events.jsonl

Features:
- UTC timestamp normalization with tz_source tracking
- Stable event_id generation for deduplication
- Partition by day
- Handles gzip archives
"""

import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load config
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "metrics_config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

LOOKBACK_DAYS = CONFIG.get("lookbackDays", 14)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "data" / "file_operations_logs"
ARCHIVES_DIR = PROJECT_ROOT / "data" / "log_archives"
OUTPUT_DIR = PROJECT_ROOT / "data" / "snapshot" / "operation_events_v1"


def parse_timestamp(ts_str: str, tz_source: str = "local_guess") -> tuple[datetime, str]:
    """
    Parse timestamp to UTC datetime.
    
    Returns:
        (datetime in UTC, tz_source indicator)
    """
    # Try ISO with Z
    if ts_str.endswith('Z'):
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc), "iso_z"
    
    # Try ISO with timezone
    if '+' in ts_str or ts_str.count('-') > 2:
        try:
            dt = datetime.fromisoformat(ts_str)
            return dt.astimezone(timezone.utc), "iso_tz"
        except ValueError:
            pass
    
    # Try ISO naive (assume local)
    try:
        dt = datetime.fromisoformat(ts_str)
        # Assume local timezone, convert to UTC
        dt = dt.replace(tzinfo=timezone.utc)
        return dt, tz_source
    except ValueError:
        pass
    
    raise ValueError(f"Could not parse timestamp: {ts_str}")


def generate_event_id(event: Dict[str, Any]) -> str:
    """
    Generate stable event ID from canonical fields.
    
    Uses md5 hash of: script_id + session_id + ts_utc + event_type +
                      operation + source_dir + dest_dir + file_count + files_md5
    """
    # Build canonical string
    canonical_parts = [
        event.get("script", ""),
        event.get("session_id", ""),
        event.get("ts_utc", ""),
        event.get("type", ""),
        event.get("operation", ""),
        event.get("source_dir", ""),
        event.get("dest_dir", ""),
        str(event.get("file_count", 0)),
    ]
    
    # Add files hash if present
    files = event.get("files", []) or event.get("files_sample", [])
    if files:
        files_str = "|".join(sorted(files[:10]))  # Use first 10 for stability
        files_md5 = hashlib.md5(files_str.encode()).hexdigest()[:8]
        canonical_parts.append(files_md5)
    
    canonical = "|".join(canonical_parts)
    return f"evt:{hashlib.md5(canonical.encode()).hexdigest()[:12]}"


def normalize_event(raw_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize a raw event to canonical schema.
    
    Returns None if event should be skipped.
    """
    try:
        # Parse timestamp
        ts_str = raw_event.get("timestamp")
        if not ts_str:
            return None
        
        ts_utc, tz_source = parse_timestamp(ts_str)
        day_str = ts_utc.strftime("%Y%m%d")
        
        # Build normalized event
        normalized = {
            "script_id": raw_event.get("script", "unknown"),
            "session_id": raw_event.get("session_id"),
            "ts_utc": ts_utc.isoformat(),
            "tz_source": tz_source,
            "event_type": raw_event.get("type", "unknown"),
            "operation": raw_event.get("operation"),
            "source_dir": raw_event.get("source_dir"),
            "dest_dir": raw_event.get("dest_dir"),
            "file_count": raw_event.get("file_count"),
            "notes": raw_event.get("notes"),
            "day": day_str,
            "extra": {}
        }
        
        # Handle files - sample up to 50 for space efficiency
        files = raw_event.get("files", []) or raw_event.get("files_sample", [])
        if files:
            normalized["files_sample"] = files[:50] if len(files) > 50 else files
        
        # Capture extra fields (include artifact signals when present)
        extra_fields = ["batch_id", "metric_mode", "details", "directory", "files_sample", "artifact", "artifact_reasons"]
        for field in extra_fields:
            if field in raw_event and field not in ["files", "files_sample"]:
                normalized["extra"][field] = raw_event[field]
        
        # Generate stable event ID
        normalized["event_id"] = generate_event_id(normalized)
        
        return normalized
        
    except Exception:
        # Skip malformed events
        return None


def extract_from_log_file(log_path: Path) -> List[Dict[str, Any]]:
    """Extract events from a single log file (plain or gzip)."""
    events = []
    
    try:
        if log_path.suffix == '.gz':
            with gzip.open(log_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            raw_event = json.loads(line)
                            normalized = normalize_event(raw_event)
                            if normalized:
                                events.append(normalized)
                        except json.JSONDecodeError:
                            continue
        else:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            raw_event = json.loads(line)
                            normalized = normalize_event(raw_event)
                            if normalized:
                                events.append(normalized)
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"  ⚠️  Error reading {log_path}: {e}")
    
    return events


def main():
    """Main entry point."""
    print("Extracting operation events...")
    print(f"Lookback: {LOOKBACK_DAYS} days")
    
    # Collect all log files
    log_files = []
    
    if LOGS_DIR.exists():
        log_files.extend(LOGS_DIR.glob("*.log"))
        log_files.extend(LOGS_DIR.glob("file_operations_*.log"))
    
    if ARCHIVES_DIR.exists():
        log_files.extend(ARCHIVES_DIR.glob("*.gz"))
    
    print(f"Found {len(log_files)} log files")
    
    # Extract and group by day
    by_day = defaultdict(list)
    seen_event_ids = set()
    duplicate_count = 0
    
    for log_file in sorted(log_files):
        print(f"  Processing {log_file.name}...")
        events = extract_from_log_file(log_file)
        
        for event in events:
            event_id = event["event_id"]
            
            # Dedupe by event_id
            if event_id in seen_event_ids:
                duplicate_count += 1
                continue
            
            seen_event_ids.add(event_id)
            day = event["day"]
            by_day[day].append(event)
    
    print(f"\nExtracted {len(seen_event_ids)} unique events ({duplicate_count} duplicates skipped)")
    print(f"Days: {len(by_day)}")
    
    # Write partitioned output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_written = 0
    for day_str in sorted(by_day.keys()):
        day_events = by_day[day_str]
        
        # Create partition directory
        day_dir = OUTPUT_DIR / f"day={day_str}"
        day_dir.mkdir(parents=True, exist_ok=True)
        output_file = day_dir / "events.jsonl"
        
        # Write events
        with open(output_file, 'w') as f:
            for event in sorted(day_events, key=lambda e: e["ts_utc"]):
                f.write(json.dumps(event) + '\n')
        
        total_written += len(day_events)
        print(f"  {day_str}: {len(day_events)} events")
    
    print(f"\n✅ Done! {total_written} events written to {OUTPUT_DIR}")
    
    # Show sample
    if by_day:
        first_day = sorted(by_day.keys())[0]
        print(f"\nSample events from {first_day} (first 3):")
        for event in by_day[first_day][:3]:
            print(f"  {event['event_id']}: {event['event_type']} by {event['script_id']}")


if __name__ == "__main__":
    main()

