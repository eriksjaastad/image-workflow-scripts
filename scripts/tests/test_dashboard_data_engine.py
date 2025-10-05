#!/usr/bin/env python3
"""
Comprehensive tests for Dashboard Data Engine
============================================
Tests all critical data loading, processing, and transformation functionality
to prevent regression issues like the historical data loss we just fixed.
"""

import json
import tempfile
import unittest
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch, mock_open
import sys
import os

# Add the dashboard directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))

from data_engine import DashboardDataEngine


class TestDashboardDataEngine(unittest.TestCase):
    """Test the core data engine functionality"""
    
    def setUp(self):
        """Set up test environment with temporary data directories"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / "data"
        self.file_ops_dir = self.data_dir / "file_operations_logs"
        self.summaries_dir = self.data_dir / "daily_summaries"
        self.archives_dir = self.data_dir / "log_archives"
        
        # Create directories
        self.file_ops_dir.mkdir(parents=True)
        self.summaries_dir.mkdir(parents=True)
        self.archives_dir.mkdir(parents=True)
        
        # Initialize data engine
        self.engine = DashboardDataEngine(data_dir=str(self.temp_dir))
        
        # Sample data
        self.sample_file_ops = [
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T10:00:00Z",
                "script": "desktop_image_selector_crop",
                "session_id": "session_001",
                "operation": "crop",
                "file_count": 5,
                "files": ["image1.png", "image2.png"]
            },
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T11:00:00Z",
                "script": "character_sorter",
                "session_id": "session_002",
                "operation": "move",
                "file_count": 3,
                "files": ["char1.png", "char2.png"]
            }
        ]
        
        self.sample_daily_summary = {
            "date": "20251003",
            "processed_at": "2025-10-04T02:00:00Z",
            "total_operations": 2,
            "scripts": {
                "desktop_image_selector_crop": {
                    "total_files": 5,
                    "operations": {"crop": 5},
                    "session_count": 1,
                    "work_time_seconds": 3600,
                    "work_time_minutes": 60.0
                },
                "character_sorter": {
                    "total_files": 3,
                    "operations": {"move": 3},
                    "session_count": 1,
                    "work_time_seconds": 1800,
                    "work_time_minutes": 30.0
                }
            }
        }

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_from_detailed_logs(self):
        """Test loading from detailed log files"""
        # Create test log file
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, 'w') as f:
            for op in self.sample_file_ops:
                f.write(json.dumps(op) + '\n')
        
        # Test loading
        records = self.engine._load_from_detailed_logs('20251003', '20251003')
        
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['script'], 'desktop_image_selector_crop')
        self.assertEqual(records[1]['operation'], 'move')

    def test_load_from_daily_summaries(self):
        """Test loading from daily summary files"""
        # Create test summary file
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        with open(summary_file, 'w') as f:
            json.dump(self.sample_daily_summary, f)
        
        # Test loading
        records = self.engine._load_from_daily_summaries(self.summaries_dir, '20251003', '20251003')
        
        self.assertEqual(len(records), 2)  # 2 operation types
        self.assertEqual(records[0]['script'], 'desktop_image_selector_crop')
        self.assertEqual(records[0]['operation'], 'crop')
        self.assertEqual(records[0]['file_count'], 5)

    def test_load_from_archived_logs(self):
        """Test loading from compressed archived log files"""
        import gzip
        
        # Create test archived file
        archive_file = self.archives_dir / "file_operations_20250923.log.gz"
        with gzip.open(archive_file, 'wt') as f:
            for op in self.sample_file_ops:
                f.write(json.dumps(op) + '\n')
        
        # Test loading
        records = self.engine._load_from_detailed_logs('20250923', '20250923')
        
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['script'], 'desktop_image_selector_crop')

    def test_combined_data_loading(self):
        """Test loading from both daily summaries and detailed logs"""
        # Create both types of files
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        with open(summary_file, 'w') as f:
            json.dump(self.sample_daily_summary, f)
        
        # Create log file for a different date with correct timestamps
        log_file = self.file_ops_dir / "file_operations_20251002.log"
        with open(log_file, 'w') as f:
            for op in self.sample_file_ops:
                op_copy = op.copy()
                op_copy['timestamp'] = "2025-10-02T10:00:00Z"  # Correct date for file
                f.write(json.dumps(op_copy) + '\n')
        
        # Test combined loading
        records = self.engine.load_file_operations()
        
        # Should have records from both sources
        self.assertGreater(len(records), 2)
        
        # Check that we have data from both dates
        dates = set(r['date'] for r in records if r.get('date'))
        self.assertIn(date(2025, 10, 2), dates)
        self.assertIn(date(2025, 10, 3), dates)

    def test_aggregation_by_script(self):
        """Test data aggregation by script"""
        # Create test data
        records = [
            {
                'timestamp': datetime(2025, 10, 3, 10, 0),
                'date': date(2025, 10, 3),
                'script': 'desktop_image_selector_crop',
                'file_count': 5
            },
            {
                'timestamp': datetime(2025, 10, 3, 11, 0),
                'date': date(2025, 10, 3),
                'script': 'desktop_image_selector_crop',
                'file_count': 3
            }
        ]
        
        aggregated = self.engine.aggregate_by_time_slice(records, 'D', 'file_count', 'script')
        
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]['script'], 'Desktop Image Selector Crop')  # Display name
        self.assertEqual(aggregated[0]['file_count'], 8.0)  # Sum of file counts

    def test_aggregation_by_operation(self):
        """Test data aggregation by operation type"""
        records = [
            {
                'timestamp': datetime(2025, 10, 3, 10, 0),
                'date': date(2025, 10, 3),
                'operation': 'crop',
                'file_count': 5
            },
            {
                'timestamp': datetime(2025, 10, 3, 11, 0),
                'date': date(2025, 10, 3),
                'operation': 'move',
                'file_count': 3
            }
        ]
        
        aggregated = self.engine.aggregate_by_time_slice(records, 'D', 'file_count', 'operation')
        
        self.assertEqual(len(aggregated), 2)
        operations = [r['operation'] for r in aggregated]
        self.assertIn('crop', operations)
        self.assertIn('move', operations)

    def test_display_name_mapping(self):
        """Test script name to display name mapping"""
        display_name = self.engine.get_display_name('desktop_image_selector_crop')
        self.assertEqual(display_name, 'Desktop Image Selector Crop')
        
        display_name = self.engine.get_display_name('character_sorter')
        self.assertEqual(display_name, 'Character Sorter')
        
        display_name = self.engine.get_display_name('unknown_script')
        self.assertEqual(display_name, 'Unknown Script')

    def test_generate_dashboard_data(self):
        """Test complete dashboard data generation"""
        # Create test data
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        with open(summary_file, 'w') as f:
            json.dump(self.sample_daily_summary, f)
        
        # Generate dashboard data
        data = self.engine.generate_dashboard_data(time_slice='D', lookback_days=7)
        
        # Check structure
        self.assertIn('metadata', data)
        self.assertIn('file_operations_data', data)
        self.assertIn('by_script', data['file_operations_data'])
        self.assertIn('by_operation', data['file_operations_data'])
        
        # Check metadata
        metadata = data['metadata']
        self.assertEqual(metadata['time_slice'], 'D')
        self.assertEqual(metadata['lookback_days'], 7)
        self.assertIn('scripts_found', metadata)

    def test_date_filtering(self):
        """Test date range filtering"""
        # Create test data for multiple dates
        for date_str in ['20251001', '20251002', '20251003']:
            log_file = self.file_ops_dir / f"file_operations_{date_str}.log"
            with open(log_file, 'w') as f:
                op = self.sample_file_ops[0].copy()
                op['timestamp'] = f"2025-10-{date_str[-2:]}T10:00:00Z"
                f.write(json.dumps(op) + '\n')
        
        # Test date filtering
        records = self.engine.load_file_operations('20251002', '20251002')
        
        # Should only have records from Oct 2
        dates = set(r['date'] for r in records if r.get('date'))
        self.assertEqual(len(dates), 1)
        self.assertIn(date(2025, 10, 2), dates)

    def test_error_handling_malformed_json(self):
        """Test handling of malformed JSON in log files"""
        # Create log file with malformed JSON
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, 'w') as f:
            f.write('{"type": "file_operation", "timestamp": "2025-10-03T10:00:00Z", "script": "test", "operation": "test", "file_count": 1}\n')
            f.write('invalid json line\n')
            f.write('{"type": "file_operation", "timestamp": "2025-10-03T11:00:00Z", "script": "test", "operation": "test", "file_count": 1}\n')
        
        # Should not crash and should process valid records
        records = self.engine._load_from_detailed_logs('20251003', '20251003')
        self.assertEqual(len(records), 2)

    def test_error_handling_missing_files(self):
        """Test handling of missing data files"""
        # Should not crash when no files exist
        records = self.engine.load_file_operations()
        self.assertEqual(len(records), 0)

    def test_file_operation_timing_calculation(self):
        """Test file-operation-based timing calculation"""
        file_operations = [
            {
                'timestamp': '2025-10-03T10:00:00Z',
                'operation': 'move',
                'file_count': 5
            },
            {
                'timestamp': '2025-10-03T10:05:00Z',
                'operation': 'crop',
                'file_count': 3
            },
            {
                'timestamp': '2025-10-03T10:20:00Z',  # 15 min gap = break
                'operation': 'delete',
                'file_count': 2
            }
        ]
        
        metrics = self.engine.calculate_file_operation_work_time(file_operations, break_threshold_minutes=10)
        
        self.assertEqual(metrics['timing_method'], 'file_operations')
        self.assertEqual(metrics['total_operations'], 3)
        self.assertGreater(metrics['work_time_seconds'], 0)
        # Should only count time between first two operations (5 minutes)
        self.assertLess(metrics['work_time_seconds'], 600)  # Less than 10 minutes


class TestDashboardIntegration(unittest.TestCase):
    """Integration tests for the complete dashboard system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / "data"
        self.file_ops_dir = self.data_dir / "file_operations_logs"
        self.summaries_dir = self.data_dir / "daily_summaries"
        
        # Create directories
        self.file_ops_dir.mkdir(parents=True)
        self.summaries_dir.mkdir(parents=True)
        
        # Initialize dashboard
        sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))
        from productivity_dashboard import ProductivityDashboard
        self.dashboard = ProductivityDashboard(data_dir=str(self.temp_dir))

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_data_flow(self):
        """Test complete data flow from files to charts"""
        # Create test data
        sample_ops = [
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T10:00:00Z",
                "script": "desktop_image_selector_crop",
                "operation": "crop",
                "file_count": 10
            }
        ]
        
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, 'w') as f:
            for op in sample_ops:
                f.write(json.dumps(op) + '\n')
        
        # Test complete pipeline
        data = self.dashboard.data_engine.generate_dashboard_data()
        chart_data = self.dashboard.transform_for_charts(data)
        
        # Verify structure
        self.assertIn('charts', chart_data)
        self.assertIn('by_script', chart_data['charts'])
        self.assertIn('by_operation', chart_data['charts'])
        
        # Verify data
        by_script = chart_data['charts']['by_script']
        self.assertIn('Desktop Image Selector Crop', by_script)
        
        script_data = by_script['Desktop Image Selector Crop']
        self.assertIn('dates', script_data)
        self.assertIn('counts', script_data)
        self.assertEqual(script_data['counts'][0], 10)

    def test_chart_transformation_script_mapping(self):
        """Test script name mapping in chart transformation"""
        # Create test data with old script names
        data = {
            'metadata': {'generated_at': '2025-10-04T00:00:00Z'},
            'file_operations_data': {
                'by_script': [
                    {
                        'script': '01_web_image_selector',  # Using actual script name
                        'time_slice': '2025-10-03',
                        'file_count': 5
                    },
                    {
                        'script': '03_web_character_sorter',  # Using actual script name
                        'time_slice': '2025-10-03',
                        'file_count': 3
                    }
                ],
                'by_operation': []
            }
        }
        
        chart_data = self.dashboard.transform_for_charts(data)
        
        # Check that script names are present in charts
        by_script = chart_data['charts']['by_script']
        # Charts should contain script data (mapped from script names in dashboard)
        self.assertTrue(len(by_script) >= 2, f"Expected at least 2 scripts in chart data, got {len(by_script)}")
        # Verify data was transformed properly
        for script_name, script_data in by_script.items():
            self.assertIn('dates', script_data)
            self.assertIn('counts', script_data)

    def test_empty_data_handling(self):
        """Test handling of empty data gracefully"""
        # Test with no data
        data = {
            'metadata': {'generated_at': '2025-10-04T00:00:00Z'},
            'file_operations_data': {}
        }
        
        chart_data = self.dashboard.transform_for_charts(data)
        
        # Should not crash and should return empty charts
        self.assertIn('charts', chart_data)
        self.assertEqual(chart_data['charts'], {})

    def test_date_sorting_in_charts(self):
        """Test that chart data is properly sorted by date"""
        data = {
            'metadata': {'generated_at': '2025-10-04T00:00:00Z'},
            'file_operations_data': {
                'by_script': [
                    {
                        'script': 'Desktop Image Selector Crop',  # Use display name
                        'time_slice': '2025-10-03',
                        'file_count': 5
                    },
                    {
                        'script': 'Desktop Image Selector Crop',
                        'time_slice': '2025-10-01',
                        'file_count': 3
                    },
                    {
                        'script': 'Desktop Image Selector Crop',
                        'time_slice': '2025-10-02',
                        'file_count': 4
                    }
                ],
                'by_operation': []
            }
        }
        
        chart_data = self.dashboard.transform_for_charts(data)
        
        # Check that we have chart data
        self.assertIn('charts', chart_data)
        self.assertIn('by_script', chart_data['charts'])
        
        # Find the script data (may be under display name or original name)
        by_script = chart_data['charts']['by_script']
        self.assertTrue(len(by_script) > 0, "Expected at least one script in chart data")
        
        # Get the first script's data (should be our Desktop Image Selector Crop)
        script_data = list(by_script.values())[0]
        dates = script_data['dates']
        counts = script_data['counts']
        
        # Dates should be sorted
        self.assertEqual(len(dates), 3, f"Expected 3 dates, got {len(dates)}")
        self.assertEqual(dates, ['2025-10-01', '2025-10-02', '2025-10-03'])
        # Counts should match the sorted dates
        self.assertEqual(counts, [3, 4, 5])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
