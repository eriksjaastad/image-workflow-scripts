#!/usr/bin/env python3
"""Analyze crop timing patterns from historical data to understand human work patterns."""

import csv
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

def parse_timestamp(ts_str):
    """Parse ISO 8601 timestamp."""
    return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

def analyze_crop_timing(csv_path):
    """Analyze timing patterns from crop log."""

    crops = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['timestamp']:
                crops.append({
                    'timestamp': parse_timestamp(row['timestamp']),
                    'session_id': row['session_id'],
                    'directory': row['directory']
                })

    # Sort by timestamp
    crops.sort(key=lambda x: x['timestamp'])

    print(f"\n{'='*80}")
    print(f"CROP TIMING ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal crops: {len(crops)}")
    print(f"Date range: {crops[0]['timestamp'].date()} to {crops[-1]['timestamp'].date()}")
    print(f"Total days: {(crops[-1]['timestamp'] - crops[0]['timestamp']).days}")

    # Calculate time between crops
    time_diffs = []
    for i in range(1, len(crops)):
        diff = (crops[i]['timestamp'] - crops[i-1]['timestamp']).total_seconds()
        time_diffs.append(diff)

    # Identify sessions (gaps > 10 minutes = new session)
    sessions = []
    current_session = [crops[0]]

    for i in range(1, len(crops)):
        time_diff = (crops[i]['timestamp'] - crops[i-1]['timestamp']).total_seconds()

        if time_diff < 600:  # Less than 10 minutes = same session
            current_session.append(crops[i])
        else:  # New session
            sessions.append(current_session)
            current_session = [crops[i]]

    sessions.append(current_session)  # Add last session

    print(f"\n{'='*80}")
    print(f"SESSION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal sessions: {len(sessions)}")

    session_lengths = []
    session_crop_counts = []
    session_speeds = []  # crops per hour

    for i, session in enumerate(sessions):
        if len(session) < 2:
            continue

        start = session[0]['timestamp']
        end = session[-1]['timestamp']
        duration = (end - start).total_seconds() / 60  # minutes
        crop_count = len(session)

        if duration > 0:
            crops_per_hour = (crop_count / duration) * 60
            session_lengths.append(duration)
            session_crop_counts.append(crop_count)
            session_speeds.append(crops_per_hour)

    if session_lengths:
        print(f"\nSession lengths (minutes):")
        print(f"  Min: {min(session_lengths):.1f}")
        print(f"  Max: {max(session_lengths):.1f}")
        print(f"  Average: {statistics.mean(session_lengths):.1f}")
        print(f"  Median: {statistics.median(session_lengths):.1f}")

        print(f"\nCrops per session:")
        print(f"  Min: {min(session_crop_counts)}")
        print(f"  Max: {max(session_crop_counts)}")
        print(f"  Average: {statistics.mean(session_crop_counts):.1f}")
        print(f"  Median: {statistics.median(session_crop_counts):.1f}")

        print(f"\nCrops per hour:")
        print(f"  Min: {min(session_speeds):.1f}")
        print(f"  Max: {max(session_speeds):.1f}")
        print(f"  Average: {statistics.mean(session_speeds):.1f}")
        print(f"  Median: {statistics.median(session_speeds):.1f}")

    # Analyze time between crops within sessions
    within_session_diffs = []
    for session in sessions:
        for i in range(1, len(session)):
            diff = (session[i]['timestamp'] - session[i-1]['timestamp']).total_seconds()
            if diff < 60:  # Only count diffs under 1 minute (within active cropping)
                within_session_diffs.append(diff)

    if within_session_diffs:
        print(f"\n{'='*80}")
        print(f"TIME BETWEEN CROPS (within sessions, < 60s)")
        print(f"{'='*80}")
        print(f"\nSeconds between crops:")
        print(f"  Min: {min(within_session_diffs):.2f}")
        print(f"  Max: {max(within_session_diffs):.2f}")
        print(f"  Average: {statistics.mean(within_session_diffs):.2f}")
        print(f"  Median: {statistics.median(within_session_diffs):.2f}")
        print(f"  Std Dev: {statistics.stdev(within_session_diffs):.2f}")

    # Analyze breaks between sessions
    if len(sessions) > 1:
        break_times = []
        for i in range(1, len(sessions)):
            prev_end = sessions[i-1][-1]['timestamp']
            next_start = sessions[i][0]['timestamp']
            break_duration = (next_start - prev_end).total_seconds() / 60  # minutes
            break_times.append(break_duration)

        print(f"\n{'='*80}")
        print(f"BREAK ANALYSIS")
        print(f"{'='*80}")
        print(f"\nBreak durations (minutes):")
        print(f"  Min: {min(break_times):.1f}")
        print(f"  Max: {max(break_times):.1f}")
        print(f"  Average: {statistics.mean(break_times):.1f}")
        print(f"  Median: {statistics.median(break_times):.1f}")

        # Categorize breaks
        short_breaks = [b for b in break_times if b < 30]
        medium_breaks = [b for b in break_times if 30 <= b < 120]
        long_breaks = [b for b in break_times if b >= 120]

        print(f"\nBreak categories:")
        print(f"  Short (< 30 min): {len(short_breaks)}")
        print(f"  Medium (30-120 min): {len(medium_breaks)}")
        print(f"  Long (> 120 min): {len(long_breaks)}")

    # Show top 10 longest sessions
    print(f"\n{'='*80}")
    print(f"TOP 10 LONGEST SESSIONS")
    print(f"{'='*80}\n")

    session_details = []
    for session in sessions:
        if len(session) < 2:
            continue
        start = session[0]['timestamp']
        end = session[-1]['timestamp']
        duration = (end - start).total_seconds() / 60
        crop_count = len(session)
        crops_per_hour = (crop_count / duration) * 60 if duration > 0 else 0

        session_details.append({
            'start': start,
            'duration': duration,
            'crops': crop_count,
            'speed': crops_per_hour
        })

    session_details.sort(key=lambda x: x['duration'], reverse=True)

    for i, s in enumerate(session_details[:10], 1):
        print(f"{i:2d}. {s['start'].strftime('%Y-%m-%d %H:%M')} | "
              f"{s['duration']:6.1f} min | {s['crops']:4d} crops | "
              f"{s['speed']:6.1f} crops/hr")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    csv_path = '/home/user/image-workflow-scripts/data/training/select_crop_log.csv'
    analyze_crop_timing(csv_path)
