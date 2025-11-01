#!/usr/bin/env python3
"""Deep dive into human cropping patterns for realistic timer simulation.

Usage:
  python scripts/tools/analyze_human_patterns.py \
    --csv data/training/select_crop_log.csv \
    --out data/ai_data/crop_queue/timing_patterns.json

If omitted, defaults are resolved relative to the repo root.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_timestamp(ts_str):
    """Parse ISO 8601 timestamp."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def analyze_human_patterns(csv_path):
    """Extract detailed timing patterns for realistic simulation.

    Returns:
        dict: timing_data structure with percentiles/means, etc.
    """
    crops = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["timestamp"]:
                crops.append(
                    {
                        "timestamp": parse_timestamp(row["timestamp"]),
                        "session_id": row["session_id"],
                        "directory": row["directory"],
                        "image_count": int(row["image_count"])
                        if row["image_count"]
                        else 1,
                    }
                )

    crops.sort(key=lambda x: x["timestamp"])

    # Group into sessions (gap > 10 min = new session)
    sessions = []
    current_session = [crops[0]]

    for i in range(1, len(crops)):
        time_diff = (crops[i]["timestamp"] - crops[i - 1]["timestamp"]).total_seconds()
        if time_diff < 600:
            current_session.append(crops[i])
        else:
            sessions.append(current_session)
            current_session = [crops[i]]
    sessions.append(current_session)

    print(f"\n{'='*80}")
    print("HUMAN CROPPING PATTERN ANALYSIS")
    print(f"{'='*80}\n")

    # Analyze time between crops within sessions
    within_session_times = []
    batch_times = defaultdict(list)  # Time between batches by time of day

    for session in sessions:
        session_times = []
        for i in range(1, len(session)):
            time_diff = (
                session[i]["timestamp"] - session[i - 1]["timestamp"]
            ).total_seconds()
            if time_diff < 60:  # Within active cropping
                session_times.append(time_diff)
                within_session_times.append(time_diff)

                # Group by hour of day
                hour = session[i]["timestamp"].hour
                batch_times[hour].append(time_diff)

    # Calculate percentiles for realistic distribution
    if within_session_times:
        within_session_times.sort()
        percentiles = [10, 25, 50, 75, 90, 95, 99]

        print("TIME BETWEEN BATCH SUBMISSIONS (seconds)")
        print("-" * 80)
        for p in percentiles:
            idx = int(len(within_session_times) * p / 100)
            print(f"  {p:2d}th percentile: {within_session_times[idx]:6.2f}s")

        print(f"\n  Mean: {statistics.mean(within_session_times):6.2f}s")
        print(f"  Std Dev: {statistics.stdev(within_session_times):6.2f}s")

    # Analyze speed changes throughout a session
    print(f"\n{'='*80}")
    print("FATIGUE/WARMUP PATTERNS")
    print("=" * 80)

    session_speed_changes = []
    for session in sessions:
        if len(session) < 20:  # Need reasonable session length
            continue

        # Split into quarters
        quarter_size = len(session) // 4
        quarters = [
            session[:quarter_size],
            session[quarter_size : quarter_size * 2],
            session[quarter_size * 2 : quarter_size * 3],
            session[quarter_size * 3 :],
        ]

        quarter_speeds = []
        for quarter in quarters:
            if len(quarter) < 2:
                continue
            duration = (
                quarter[-1]["timestamp"] - quarter[0]["timestamp"]
            ).total_seconds() / 60
            if duration > 0:
                crops_per_hour = (len(quarter) / duration) * 60
                quarter_speeds.append(crops_per_hour)

        if len(quarter_speeds) == 4:
            session_speed_changes.append(quarter_speeds)

    if session_speed_changes:
        avg_quarters = [
            statistics.mean([s[i] for s in session_speed_changes]) for i in range(4)
        ]
        print("\nAverage crops/hour by session quarter:")
        for i, speed in enumerate(avg_quarters, 1):
            pct_change = (
                ((speed - avg_quarters[0]) / avg_quarters[0] * 100)
                if avg_quarters[0] > 0
                else 0
            )
            print(f"  Quarter {i}: {speed:6.1f} crops/hr ({pct_change:+5.1f}%)")

    # Analyze work patterns by hour of day
    print(f"\n{'='*80}")
    print("PRODUCTIVITY BY HOUR OF DAY")
    print("=" * 80)

    hourly_crops = defaultdict(list)
    for crop in crops:
        hour = crop["timestamp"].hour
        hourly_crops[hour].append(crop)

    hourly_stats = []
    for hour in sorted(hourly_crops.keys()):
        crop_list = hourly_crops[hour]
        if len(crop_list) < 5:  # Skip hours with too few samples
            continue

        # Calculate average speed for this hour
        session_speeds = []
        i = 0
        while i < len(crop_list):
            # Find continuous sequence
            seq = [crop_list[i]]
            j = i + 1
            while j < len(crop_list):
                if (
                    crop_list[j]["timestamp"] - crop_list[j - 1]["timestamp"]
                ).total_seconds() < 600:
                    seq.append(crop_list[j])
                    j += 1
                else:
                    break

            if len(seq) >= 2:
                duration = (
                    seq[-1]["timestamp"] - seq[0]["timestamp"]
                ).total_seconds() / 60
                if duration > 0:
                    session_speeds.append((len(seq) / duration) * 60)

            i = j if j > i else i + 1

        if session_speeds:
            avg_speed = statistics.mean(session_speeds)
            hourly_stats.append((hour, avg_speed, len(crop_list)))

    if hourly_stats:
        print("\n Hour | Crops/hr | Total crops")
        print("------+----------+------------")
        for hour, speed, count in hourly_stats:
            time_str = f"{hour:02d}:00"
            print(f" {time_str} | {speed:8.1f} | {count:11d}")

    # Generate recommended timing parameters
    print(f"\n{'='*80}")
    print("RECOMMENDED TIMER PARAMETERS")
    print("=" * 80)

    if within_session_times:
        # Use realistic distribution
        p50 = within_session_times[int(len(within_session_times) * 0.50)]
        p25 = within_session_times[int(len(within_session_times) * 0.25)]
        p75 = within_session_times[int(len(within_session_times) * 0.75)]

        print("\nBase timing (seconds between batches):")
        print(f"  Fast (25th percentile): {p25:.2f}s")
        print(f"  Normal (median): {p50:.2f}s")
        print(f"  Slow (75th percentile): {p75:.2f}s")

        print("\nSession parameters:")
        avg_session_min = statistics.mean(
            [
                (s[-1]["timestamp"] - s[0]["timestamp"]).total_seconds() / 60
                for s in sessions
                if len(s) > 1
            ]
        )
        avg_session_crops = statistics.mean([len(s) for s in sessions])
        print(f"  Average session length: {avg_session_min:.1f} minutes")
        print(f"  Average crops per session: {avg_session_crops:.1f}")
        print(
            f"  Sessions per day (6hr workday): {(6*60) / (avg_session_min + 20.5):.1f}"
        )

        print("\nVariability parameters:")
        print(
            f"  Std dev: {statistics.stdev(within_session_times):.2f}s (use for random variation)"
        )
        print(f"  Add random jitter: ±{statistics.stdev(within_session_times)/2:.2f}s")

    # Prepare timing distribution for processor
    timing_data = {
        "percentiles": {
            f"p{p}": float(
                within_session_times[int(len(within_session_times) * p / 100)]
            )
            for p in [10, 25, 50, 75, 90, 95, 99]
        }
        if within_session_times
        else {},
        "mean": float(statistics.mean(within_session_times))
        if within_session_times
        else 0,
        "stddev": float(statistics.stdev(within_session_times))
        if within_session_times
        else 0,
        "session_avg_minutes": float(avg_session_min) if sessions else 0,
        "session_avg_crops": float(avg_session_crops) if sessions else 0,
        "break_median_minutes": 20.5,
        "hourly_productivity": {
            str(hour): float(speed) for hour, speed, _ in hourly_stats
        }
        if hourly_stats
        else {},
    }
    return timing_data


def _get_repo_root() -> Path:
    # scripts/tools/analyze_human_patterns.py → repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract timing patterns and write JSON for queue processor"
    )
    default_csv = _get_repo_root() / "data" / "training" / "select_crop_log.csv"
    default_out = (
        _get_repo_root() / "data" / "ai_data" / "crop_queue" / "timing_patterns.json"
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=str(default_csv),
        help="Path to select_crop_log.csv",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=str(default_out),
        help="Output JSON path for timing patterns",
    )
    args = parser.parse_args()

    timing_data = analyze_human_patterns(args.csv_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(timing_data, f, indent=2)
    print(f"\n✓ Timing patterns saved to: {out_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
