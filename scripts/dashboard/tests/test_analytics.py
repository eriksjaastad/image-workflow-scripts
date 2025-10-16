#!/usr/bin/env python3
"""
Tests for Dashboard Analytics
==============================
Tests bucketing, alignment, edge cases, and contract compliance.

Run with: pytest test_analytics.py -v
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.dashboard.analytics import DashboardAnalytics
from scripts.dashboard.data_engine import DashboardDataEngine


class TestLabelGeneration:
    """Test baseline label generation for all time slices."""
    
    def test_daily_labels_lookback_7(self):
        """Test daily labels for 7-day lookback."""
        engine = DashboardDataEngine(str(project_root))
        labels = engine.build_time_labels("D", 7)
        
        assert len(labels) == 8  # Today + 7 days back = 8 days
        assert all(isinstance(label, str) for label in labels)
        
        # Check format: YYYY-MM-DD
        for label in labels:
            assert len(label) == 10
            datetime.fromisoformat(label)  # Should not raise
    
    def test_hourly_labels_lookback_1(self):
        """Test hourly labels for 1-day lookback."""
        engine = DashboardDataEngine(str(project_root))
        labels = engine.build_time_labels("1H", 1)
        
        # Should have ~25 labels (24 hours + partial hours)
        assert 24 <= len(labels) <= 26
        
        # Check format: ISO timestamp
        for label in labels:
            dt = datetime.fromisoformat(label)
            assert dt.minute == 0  # Hourly alignment
            assert dt.second == 0
    
    def test_15min_labels_lookback_1(self):
        """Test 15-minute labels for 1-day lookback."""
        engine = DashboardDataEngine(str(project_root))
        labels = engine.build_time_labels("15min", 1)
        
        # Should have ~96 labels (4 per hour * 24 hours)
        assert 90 <= len(labels) <= 100
        
        # Check 15-minute alignment
        for label in labels:
            dt = datetime.fromisoformat(label)
            assert dt.minute % 15 == 0
            assert dt.second == 0
    
    def test_weekly_labels_lookback_30(self):
        """Test weekly labels for 30-day lookback."""
        engine = DashboardDataEngine(str(project_root))
        labels = engine.build_time_labels("W", 30)
        
        # Should have ~4-5 weeks
        assert 4 <= len(labels) <= 6
        
        # Check format and Monday alignment
        for label in labels:
            dt = datetime.fromisoformat(label)
            assert dt.weekday() == 0  # Monday = 0
    
    def test_monthly_labels_lookback_90(self):
        """Test monthly labels for 90-day lookback."""
        engine = DashboardDataEngine(str(project_root))
        labels = engine.build_time_labels("M", 90)
        
        # Should have 3-4 months
        assert 3 <= len(labels) <= 5
        
        # Check format: YYYY-MM-01
        for label in labels:
            assert label.endswith("-01")
            dt = datetime.fromisoformat(label)
            assert dt.day == 1
    
    def test_labels_sorted_chronologically(self):
        """Test that all label types are sorted ascending."""
        engine = DashboardDataEngine(str(project_root))
        
        for slice_type in ["15min", "1H", "D", "W", "M"]:
            labels = engine.build_time_labels(slice_type, 30)
            
            # Convert to timestamps for comparison
            timestamps = [datetime.fromisoformat(label) for label in labels]
            
            # Check sorted
            assert timestamps == sorted(timestamps)


class TestAlignment:
    """Test series alignment to baseline labels."""
    
    def test_align_empty_records(self):
        """Test alignment with empty records."""
        analytics = DashboardAnalytics(project_root)
        engine = analytics.engine
        
        baseline = engine.build_time_labels("D", 3)
        result = analytics._aggregate_to_baseline(
            records=[],
            baseline_labels=baseline,
            group_field="script",
            value_field="file_count"
        )
        
        assert result == {}
    
    def test_align_single_record(self):
        """Test alignment with single record."""
        analytics = DashboardAnalytics(project_root)
        engine = analytics.engine
        
        baseline = ["2025-10-01", "2025-10-02", "2025-10-03"]
        records = [
            {
                "time_slice": "2025-10-02",
                "script": "test_tool",
                "file_count": 100
            }
        ]
        
        result = analytics._aggregate_to_baseline(
            records=records,
            baseline_labels=baseline,
            group_field="script",
            value_field="file_count"
        )
        
        assert "test_tool" in result
        assert result["test_tool"]["dates"] == baseline
        assert result["test_tool"]["counts"] == [0, 100, 0]
    
    def test_align_gaps_filled_with_zeros(self):
        """Test that gaps are filled with zeros, not nulls."""
        analytics = DashboardAnalytics(project_root)
        
        baseline = ["2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04"]
        records = [
            {"time_slice": "2025-10-01", "script": "tool1", "file_count": 50},
            {"time_slice": "2025-10-04", "script": "tool1", "file_count": 75}
        ]
        
        result = analytics._aggregate_to_baseline(
            records=records,
            baseline_labels=baseline,
            group_field="script",
            value_field="file_count"
        )
        
        assert result["tool1"]["counts"] == [50, 0, 0, 75]
        assert None not in result["tool1"]["counts"]
    
    def test_align_multiple_tools(self):
        """Test alignment with multiple tools."""
        analytics = DashboardAnalytics(project_root)
        
        baseline = ["2025-10-01", "2025-10-02"]
        records = [
            {"time_slice": "2025-10-01", "script": "tool1", "file_count": 100},
            {"time_slice": "2025-10-01", "script": "tool2", "file_count": 50},
            {"time_slice": "2025-10-02", "script": "tool1", "file_count": 80}
        ]
        
        result = analytics._aggregate_to_baseline(
            records=records,
            baseline_labels=baseline,
            group_field="script",
            value_field="file_count"
        )
        
        assert len(result) == 2
        assert result["tool1"]["counts"] == [100, 80]
        assert result["tool2"]["counts"] == [50, 0]
    
    def test_align_duplicate_timestamps_summed(self):
        """Test that multiple records for same timestamp are summed."""
        analytics = DashboardAnalytics(project_root)
        
        baseline = ["2025-10-01"]
        records = [
            {"time_slice": "2025-10-01", "script": "tool1", "file_count": 50},
            {"time_slice": "2025-10-01", "script": "tool1", "file_count": 30},
            {"time_slice": "2025-10-01", "script": "tool1", "file_count": 20}
        ]
        
        result = analytics._aggregate_to_baseline(
            records=records,
            baseline_labels=baseline,
            group_field="script",
            value_field="file_count"
        )
        
        assert result["tool1"]["counts"] == [100]  # Sum of 50+30+20


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_lookback_1_day_minimum(self):
        """Test minimum lookback of 1 day."""
        engine = DashboardDataEngine(str(project_root))
        labels = engine.build_time_labels("D", 1)
        
        assert len(labels) >= 2  # Today + 1 day back
    
    def test_lookback_365_days_maximum(self):
        """Test large lookback (365 days)."""
        engine = DashboardDataEngine(str(project_root))
        labels = engine.build_time_labels("D", 365)
        
        # Should handle large ranges without error
        assert len(labels) == 366  # 365 + today
    
    def test_dst_boundaries(self):
        """Test that DST transitions don't break label generation."""
        engine = DashboardDataEngine(str(project_root))
        
        # Generate labels spanning typical DST transition dates
        labels = engine.build_time_labels("D", 180)
        
        # Should have consistent 24-hour gaps (no duplicates or skips)
        timestamps = [datetime.fromisoformat(label) for label in labels]
        
        # Check all consecutive pairs
        for i in range(len(timestamps) - 1):
            delta = (timestamps[i + 1] - timestamps[i]).days
            assert delta == 1  # Exactly 1 day apart
    
    def test_sparse_data(self):
        """Test handling of very sparse data (mostly zeros)."""
        analytics = DashboardAnalytics(project_root)
        
        # 30 days of labels, only 1 record
        baseline = [f"2025-10-{d:02d}" for d in range(1, 31)]
        records = [
            {"time_slice": "2025-10-15", "script": "rare_tool", "file_count": 5}
        ]
        
        result = analytics._aggregate_to_baseline(
            records=records,
            baseline_labels=baseline,
            group_field="script",
            value_field="file_count"
        )
        
        counts = result["rare_tool"]["counts"]
        assert sum(counts) == 5
        assert counts.count(0) == 29  # All but one day is zero


class TestContractCompliance:
    """Test that response matches dashboard template contract."""
    
    def test_response_has_required_keys(self):
        """Test that response has all required top-level keys."""
        analytics = DashboardAnalytics(project_root)
        
        response = analytics.generate_dashboard_response(
            time_slice="D",
            lookback_days=7,
            project_id=None
        )
        
        required_keys = [
            "metadata",
            "projects",
            "charts",
            "timing_data",
            "project_comparisons",
            "project_kpi",
            "project_metrics",
            "project_markers"
        ]
        
        for key in required_keys:
            assert key in response, f"Missing required key: {key}"
    
    def test_metadata_structure(self):
        """Test metadata structure matches contract."""
        analytics = DashboardAnalytics(project_root)
        
        response = analytics.generate_dashboard_response(
            time_slice="D",
            lookback_days=7,
            project_id=None
        )
        
        metadata = response["metadata"]
        
        assert "generated_at" in metadata
        assert "time_slice" in metadata
        assert "lookback_days" in metadata
        assert "baseline_labels" in metadata
        assert "active_project" in metadata
        
        # Check baseline_labels has all slices
        baseline = metadata["baseline_labels"]
        assert all(slice in baseline for slice in ["15min", "1H", "D", "W", "M"])
    
    def test_charts_structure(self):
        """Test charts structure matches contract."""
        analytics = DashboardAnalytics(project_root)
        
        response = analytics.generate_dashboard_response(
            time_slice="D",
            lookback_days=7,
            project_id=None
        )
        
        charts = response["charts"]
        
        assert "by_script" in charts
        assert "by_operation" in charts
        
        # Each tool should have dates and counts
        for tool, data in charts["by_script"].items():
            assert "dates" in data
            assert "counts" in data
            assert len(data["dates"]) == len(data["counts"])
        
        for op, data in charts["by_operation"].items():
            assert "dates" in data
            assert "counts" in data
            assert len(data["dates"]) == len(data["counts"])
    
    def test_timing_data_structure(self):
        """Test timing_data structure matches contract."""
        analytics = DashboardAnalytics(project_root)
        
        response = analytics.generate_dashboard_response(
            time_slice="D",
            lookback_days=7,
            project_id=None
        )
        
        timing_data = response["timing_data"]
        
        # Each tool should have work_time_minutes and timing_method
        for tool, stats in timing_data.items():
            assert "work_time_minutes" in stats
            assert "timing_method" in stats
            assert stats["timing_method"] in ["file_operations", "activity_timer", "both", "unknown"]
    
    def test_project_kpi_structure(self):
        """Test project_kpi structure matches contract."""
        analytics = DashboardAnalytics(project_root)
        
        response = analytics.generate_dashboard_response(
            time_slice="D",
            lookback_days=7,
            project_id=None
        )
        
        kpi = response["project_kpi"]
        
        assert "images_per_hour" in kpi
        assert "images_processed" in kpi
        assert isinstance(kpi["images_per_hour"], (int, float))
        assert isinstance(kpi["images_processed"], int)
    
    def test_project_markers_structure(self):
        """Test project_markers structure matches contract."""
        analytics = DashboardAnalytics(project_root)
        
        response = analytics.generate_dashboard_response(
            time_slice="D",
            lookback_days=7,
            project_id=None
        )
        
        markers = response["project_markers"]
        
        assert "startedAt" in markers
        assert "finishedAt" in markers


class TestTimeSensitivity:
    """Test that current time doesn't break label generation."""
    
    def test_midnight_boundary(self):
        """Test label generation at midnight."""
        engine = DashboardDataEngine(str(project_root))
        
        # Should work at any time of day
        labels = engine.build_time_labels("D", 3)
        assert len(labels) == 4
    
    def test_month_boundary(self):
        """Test label generation at month boundaries."""
        engine = DashboardDataEngine(str(project_root))
        
        # Should handle month transitions
        labels = engine.build_time_labels("M", 60)
        assert len(labels) >= 2


def test_toy_example_daily():
    """
    Generate a tiny 3-label toy example for daily view.
    This demonstrates the exact JSON structure for documentation.
    """
    analytics = DashboardAnalytics(project_root)
    engine = analytics.engine
    
    # Mock baseline labels
    baseline = ["2025-10-13", "2025-10-14", "2025-10-15"]
    
    # Mock records
    records = [
        {"time_slice": "2025-10-13", "script": "01_web_image_selector", "file_count": 120},
        {"time_slice": "2025-10-14", "script": "01_web_image_selector", "file_count": 95},
        {"time_slice": "2025-10-15", "script": "01_web_image_selector", "file_count": 143},
        {"time_slice": "2025-10-13", "script": "04_multi_crop_tool", "file_count": 80},
        {"time_slice": "2025-10-15", "script": "04_multi_crop_tool", "file_count": 92}
    ]
    
    result = analytics._aggregate_to_baseline(
        records=records,
        baseline_labels=baseline,
        group_field="script",
        value_field="file_count"
    )
    
    print("\n=== TOY EXAMPLE (Daily, 3 labels) ===")
    print(f"Baseline labels: {baseline}")
    print(f"\nby_script chart data:")
    for tool, data in result.items():
        print(f"  {tool}:")
        print(f"    dates: {data['dates']}")
        print(f"    counts: {data['counts']}")
    
    # Assertions
    assert "Web Image Selector" in result
    assert "Multi Crop Tool" in result
    assert result["Web Image Selector"]["counts"] == [120, 95, 143]
    assert result["Multi Crop Tool"]["counts"] == [80, 0, 92]


def test_toy_example_hourly():
    """Generate a tiny 3-label toy example for hourly view."""
    analytics = DashboardAnalytics(project_root)
    
    # Mock 3 hourly labels
    baseline = [
        "2025-10-15T14:00:00",
        "2025-10-15T15:00:00",
        "2025-10-15T16:00:00"
    ]
    
    records = [
        {"time_slice": "2025-10-15T14:00:00", "operation": "crop", "file_count": 25},
        {"time_slice": "2025-10-15T15:00:00", "operation": "crop", "file_count": 30},
        {"time_slice": "2025-10-15T16:00:00", "operation": "crop", "file_count": 22},
        {"time_slice": "2025-10-15T14:00:00", "operation": "delete", "file_count": 10},
        {"time_slice": "2025-10-15T16:00:00", "operation": "delete", "file_count": 8}
    ]
    
    result = analytics._aggregate_to_baseline(
        records=records,
        baseline_labels=baseline,
        group_field="operation",
        value_field="file_count"
    )
    
    print("\n=== TOY EXAMPLE (Hourly, 3 labels) ===")
    print(f"Baseline labels: {baseline}")
    print(f"\nby_operation chart data:")
    for op, data in result.items():
        print(f"  {op}:")
        print(f"    dates: {data['dates']}")
        print(f"    counts: {data['counts']}")
    
    assert result["crop"]["counts"] == [25, 30, 22]
    assert result["delete"]["counts"] == [10, 0, 8]


if __name__ == "__main__":
    # Run toy examples to show output
    print("Running toy examples...")
    test_toy_example_daily()
    test_toy_example_hourly()
    
    print("\n✅ Toy examples complete!")
    print("\nRun full test suite with: pytest test_analytics.py -v")


