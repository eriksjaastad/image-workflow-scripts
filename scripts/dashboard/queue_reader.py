#!/usr/bin/env python3
"""
Queue Data Reader - Dashboard Integration
==========================================
Reads crop queue statistics for dashboard display.

Data sources:
- Queue file: data/ai_data/crop_queue/crop_queue.jsonl
- Timing log: data/ai_data/crop_queue/timing_log.csv

Provides:
- Queue status counts (pending, processing, completed, failed)
- Processing time statistics
- Batches per session
- Throughput trends over time
"""

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


class QueueDataReader:
    """Reads and analyzes crop queue data for dashboard."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.queue_file = self.data_dir / "data" / "ai_data" / "crop_queue" / "crop_queue.jsonl"
        self.timing_log = self.data_dir / "data" / "ai_data" / "crop_queue" / "timing_log.csv"

    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get current queue status counts.

        Returns:
            Dict with keys: pending, processing, completed, failed, total_crops
        """
        stats = {
            'pending': 0,
            'processing': 0,
            'completed': 0,
            'failed': 0,
            'total_crops': 0
        }

        if not self.queue_file.exists():
            return stats

        try:
            with open(self.queue_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        batch = json.loads(line)
                        status = batch.get('status', 'pending')
                        num_crops = len(batch.get('crops', []))

                        if status in stats:
                            stats[status] += 1
                            stats['total_crops'] += num_crops

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"[QueueReader] Error reading queue file: {e}")

        return stats

    def get_processing_time_trends(self, lookback_days: int = 30) -> List[Dict]:
        """
        Get processing time trends over time.

        Returns:
            List of dicts with: timestamp, avg_time_ms, crops_processed
        """
        if not self.timing_log.exists():
            return []

        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        trends = []

        try:
            with open(self.timing_log, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp_started'])
                        if timestamp < cutoff_date:
                            continue

                        trends.append({
                            'timestamp': row['timestamp_started'],
                            'batch_id': row['batch_id'],
                            'crops_in_batch': int(row['crops_in_batch']),
                            'processing_time_ms': int(row['processing_time_ms']),
                            'avg_time_per_crop_ms': int(row['processing_time_ms']) / max(1, int(row['crops_in_batch']))
                        })

                    except (ValueError, KeyError):
                        continue

        except Exception as e:
            print(f"[QueueReader] Error reading timing log: {e}")

        return trends

    def get_batches_per_session(self, lookback_days: int = 30) -> List[Dict]:
        """
        Group batches into sessions and count batches per session.

        Sessions are groups of batches with <5min gaps between them.

        Returns:
            List of dicts with: session_start, batch_count, total_crops
        """
        trends = self.get_processing_time_trends(lookback_days)
        if not trends:
            return []

        # Sort by timestamp
        trends.sort(key=lambda x: x['timestamp'])

        sessions = []
        current_session = {
            'session_start': trends[0]['timestamp'],
            'batch_count': 0,
            'total_crops': 0,
            'batches': []
        }

        session_gap_minutes = 5

        for i, batch in enumerate(trends):
            if i == 0:
                current_session['batches'].append(batch)
                current_session['batch_count'] += 1
                current_session['total_crops'] += batch['crops_in_batch']
                continue

            # Check gap from previous batch
            prev_time = datetime.fromisoformat(trends[i-1]['timestamp'])
            curr_time = datetime.fromisoformat(batch['timestamp'])
            gap = (curr_time - prev_time).total_seconds() / 60  # minutes

            if gap > session_gap_minutes:
                # Start new session
                sessions.append({
                    'session_start': current_session['session_start'],
                    'batch_count': current_session['batch_count'],
                    'total_crops': current_session['total_crops']
                })

                current_session = {
                    'session_start': batch['timestamp'],
                    'batch_count': 1,
                    'total_crops': batch['crops_in_batch'],
                    'batches': [batch]
                }
            else:
                # Add to current session
                current_session['batches'].append(batch)
                current_session['batch_count'] += 1
                current_session['total_crops'] += batch['crops_in_batch']

        # Don't forget last session
        if current_session['batch_count'] > 0:
            sessions.append({
                'session_start': current_session['session_start'],
                'batch_count': current_session['batch_count'],
                'total_crops': current_session['total_crops']
            })

        return sessions

    def get_throughput_by_day(self, lookback_days: int = 30) -> Dict[str, Dict]:
        """
        Get crops processed per day.

        Returns:
            Dict of {date_str: {crops: int, batches: int, avg_time_ms: float}}
        """
        trends = self.get_processing_time_trends(lookback_days)
        if not trends:
            return {}

        by_day = defaultdict(lambda: {'crops': 0, 'batches': 0, 'total_time_ms': 0})

        for batch in trends:
            date_str = batch['timestamp'][:10]  # YYYY-MM-DD
            by_day[date_str]['crops'] += batch['crops_in_batch']
            by_day[date_str]['batches'] += 1
            by_day[date_str]['total_time_ms'] += batch['processing_time_ms']

        # Calculate averages
        for date_str in by_day:
            day_data = by_day[date_str]
            day_data['avg_time_ms'] = day_data['total_time_ms'] / max(1, day_data['batches'])

        return dict(by_day)

    def get_summary_stats(self, lookback_days: int = 30) -> Dict:
        """
        Get summary statistics for queue system.

        Returns:
            Dict with queue status, processing stats, and trends
        """
        queue_stats = self.get_queue_stats()
        sessions = self.get_batches_per_session(lookback_days)
        by_day = self.get_throughput_by_day(lookback_days)

        # Calculate session averages
        avg_batches_per_session = 0
        avg_crops_per_session = 0
        if sessions:
            avg_batches_per_session = sum(s['batch_count'] for s in sessions) / len(sessions)
            avg_crops_per_session = sum(s['total_crops'] for s in sessions) / len(sessions)

        # Calculate daily averages
        total_crops_processed = sum(d['crops'] for d in by_day.values())
        total_batches_processed = sum(d['batches'] for d in by_day.values())
        days_with_activity = len(by_day)

        return {
            'queue_status': queue_stats,
            'session_stats': {
                'total_sessions': len(sessions),
                'avg_batches_per_session': round(avg_batches_per_session, 1),
                'avg_crops_per_session': round(avg_crops_per_session, 1)
            },
            'throughput': {
                'total_crops_processed': total_crops_processed,
                'total_batches_processed': total_batches_processed,
                'days_with_activity': days_with_activity,
                'avg_crops_per_day': round(total_crops_processed / max(1, days_with_activity), 1),
                'avg_batches_per_day': round(total_batches_processed / max(1, days_with_activity), 1)
            },
            'lookback_days': lookback_days
        }
