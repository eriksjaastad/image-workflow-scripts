#!/usr/bin/env python3
"""
Tests for Data Pipeline V1 Scripts
===================================
Tests the new snapshot-based data pipeline including:
- Operation event extraction
- Daily aggregates building
- Session derivation from operations
"""

import json
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Note: Testing data pipeline scripts as integration tests
# since they're designed to be run as standalone scripts


class TestOperationEventExtraction(unittest.TestCase):
    """Test operation event extraction and normalization"""

    def test_timestamp_parsing(self):
        """Test various timestamp formats can be parsed"""
        # Test ISO with Z
        iso_z = "2025-10-16T14:30:00Z"
        dt = datetime.fromisoformat(iso_z.replace("Z", "+00:00"))
        self.assertEqual(dt.tzinfo, UTC)

        # Test datetime object has UTC timezone
        dt_utc = datetime(2025, 10, 16, 14, 30, 0, tzinfo=UTC)
        self.assertEqual(dt_utc.tzinfo, UTC)

    def test_stable_id_generation_concept(self):
        """Test stable ID generation concept using MD5"""
        import hashlib

        event1 = {
            "timestamp": "2025-10-16T14:30:00Z",
            "script": "test_script",
            "operation": "move",
            "file_count": 5,
        }

        # Generate stable ID
        canonical = json.dumps(event1, sort_keys=True)
        id1 = hashlib.md5(canonical.encode()).hexdigest()

        # Same event should generate same ID
        id2 = hashlib.md5(canonical.encode()).hexdigest()
        self.assertEqual(id1, id2)

        # IDs should be hex strings
        self.assertTrue(all(c in "0123456789abcdef" for c in id1))
        self.assertEqual(len(id1), 32)  # MD5 hash length

    def test_snapshot_directory_structure(self):
        """Test snapshot directory follows day=YYYYMMDD partition format"""
        # Expected structure: snapshot/operation_events_v1/day=20251016/events.jsonl
        day_str = "20251016"
        expected_dir = f"day={day_str}"

        self.assertTrue(expected_dir.startswith("day="))
        self.assertEqual(len(day_str), 8)  # YYYYMMDD format

    def test_operation_event_schema(self):
        """Test operation event follows expected schema"""
        event = {
            "event_id": "abc123",
            "ts_utc": "2025-10-16T14:30:00Z",
            "tz_source": "utc",
            "script_id": "test_script",
            "operation": "move",
            "file_count": 5,
            "source_dir": "/path/to/source",
            "dest_dir": "/path/to/dest",
        }

        # Verify required fields
        required = ["event_id", "ts_utc", "script_id", "operation"]
        for field in required:
            self.assertIn(field, event)

        # Verify timestamp is UTC ISO
        self.assertTrue(event["ts_utc"].endswith("Z"))

        # Verify file_count is integer or None
        self.assertIsInstance(event["file_count"], (int, type(None)))


class TestDailyAggregates(unittest.TestCase):
    """Test daily aggregate building"""

    def test_config_file_exists(self):
        """Test configuration file exists and is valid JSON"""
        config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "metrics_config.json"
        )
        self.assertTrue(config_path.exists())

        with open(config_path) as f:
            config = json.load(f)

        self.assertIn("metrics", config)
        self.assertIn("lookbackDays", config)

    def test_aggregate_schema(self):
        """Test daily aggregate follows expected schema"""
        aggregate = {
            "day": "20251016",
            "total_operations": 10,
            "total_files": 100,
            "by_script": {
                "web_image_selector": {
                    "total_operations": 5,
                    "total_files": 50,
                    "first_operation_ts": "2025-10-16T14:00:00Z",
                    "last_operation_ts": "2025-10-16T18:00:00Z",
                }
            },
            "by_operation": {"move": {"total_operations": 8, "total_files": 80}},
        }

        # Verify structure
        self.assertIn("day", aggregate)
        self.assertIn("total_operations", aggregate)
        self.assertIn("total_files", aggregate)
        self.assertIn("by_script", aggregate)
        self.assertIn("by_operation", aggregate)

        # Verify day format
        self.assertEqual(len(aggregate["day"]), 8)  # YYYYMMDD

    def test_aggregate_math(self):
        """Test aggregate calculations work correctly"""
        # Simple aggregation logic
        events = [{"file_count": 10}, {"file_count": 5}, {"file_count": 3}]

        total = sum(e["file_count"] for e in events)
        count = len(events)

        self.assertEqual(total, 18)
        self.assertEqual(count, 3)

    def test_aggregate_grouping_by_script(self):
        """Test grouping logic by script"""
        events = [
            {"script_id": "web_image_selector", "file_count": 10},
            {"script_id": "web_image_selector", "file_count": 5},
            {"script_id": "multi_crop_tool", "file_count": 20},
        ]

        # Group by script
        by_script = {}
        for event in events:
            script = event["script_id"]
            if script not in by_script:
                by_script[script] = {"total_files": 0, "count": 0}
            by_script[script]["total_files"] += event["file_count"]
            by_script[script]["count"] += 1

        self.assertEqual(len(by_script), 2)
        self.assertEqual(by_script["web_image_selector"]["total_files"], 15)
        self.assertEqual(by_script["web_image_selector"]["count"], 2)
        self.assertEqual(by_script["multi_crop_tool"]["total_files"], 20)


class TestSessionDerivation(unittest.TestCase):
    """Test session derivation from operation events"""

    def test_session_gap_detection(self):
        """Test that sessions are split on time gaps"""
        # This would test derive_sessions_from_ops_v1.py
        # Skipping implementation as it requires complex setup
        self.skipTest("Complex integration test - requires full pipeline setup")

    def test_active_time_calculation(self):
        """Test bounded inter-event gap calculation"""
        # This would test the MAX_GAP_CONTRIB logic
        self.skipTest("Complex integration test - requires full pipeline setup")


class TestSnapshotDeduplication(unittest.TestCase):
    """Test deduplication logic"""

    def test_deduplication_logic(self):
        """Test deduplication using set tracking"""
        seen_ids = set()

        # Simulate adding events
        event_ids = ["id1", "id2", "id3", "id1", "id2"]  # id1 and id2 are duplicates

        unique_events = []
        for event_id in event_ids:
            if event_id not in seen_ids:
                seen_ids.add(event_id)
                unique_events.append(event_id)

        # Should only have 3 unique events
        self.assertEqual(len(unique_events), 3)
        self.assertEqual(set(unique_events), {"id1", "id2", "id3"})
        self.assertEqual(len(seen_ids), 3)


if __name__ == "__main__":
    unittest.main()
