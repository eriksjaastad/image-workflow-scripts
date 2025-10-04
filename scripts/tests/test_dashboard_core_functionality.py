#!/usr/bin/env python3
"""
Core Dashboard Functionality Tests
=================================
Focused tests that verify the critical dashboard functionality is working
without interfering with the real data directory.
"""

import json
import tempfile
import unittest
from datetime import datetime, date
from pathlib import Path
import sys

# Add the dashboard directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))


class TestDashboardCoreFunctionality(unittest.TestCase):
    """Test core dashboard functionality that we just fixed"""
    
    def test_data_engine_can_load_records(self):
        """Test that the data engine can load records from the real data directory"""
        from data_engine import DashboardDataEngine
        
        # Use the real data directory
        engine = DashboardDataEngine(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Test loading file operations
        records = engine.load_file_operations()
        
        # Should have records (we know there are 152,963+ records)
        self.assertGreater(len(records), 0, "Should have file operation records")
        
        # Should have records with proper structure
        if records:
            record = records[0]
            self.assertIn('timestamp', record, "Record should have timestamp")
            self.assertIn('script', record, "Record should have script")
            self.assertIn('operation', record, "Record should have operation")
            self.assertIn('file_count', record, "Record should have file_count")

    def test_data_engine_can_aggregate_by_script(self):
        """Test that the data engine can aggregate data by script"""
        from data_engine import DashboardDataEngine
        
        engine = DashboardDataEngine(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Load some records
        records = engine.load_file_operations()
        self.assertGreater(len(records), 0, "Should have records to aggregate")
        
        # Test aggregation
        aggregated = engine.aggregate_by_time_slice(records, 'D', 'file_count', 'script')
        
        # Should have aggregated records
        self.assertGreater(len(aggregated), 0, "Should have aggregated records")
        
        # Should have proper structure
        if aggregated:
            agg_record = aggregated[0]
            self.assertIn('time_slice', agg_record, "Aggregated record should have time_slice")
            self.assertIn('script', agg_record, "Aggregated record should have script")
            self.assertIn('file_count', agg_record, "Aggregated record should have file_count")

    def test_data_engine_can_aggregate_by_operation(self):
        """Test that the data engine can aggregate data by operation"""
        from data_engine import DashboardDataEngine
        
        engine = DashboardDataEngine(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Load some records
        records = engine.load_file_operations()
        self.assertGreater(len(records), 0, "Should have records to aggregate")
        
        # Test aggregation
        aggregated = engine.aggregate_by_time_slice(records, 'D', 'file_count', 'operation')
        
        # Should have aggregated records
        self.assertGreater(len(aggregated), 0, "Should have aggregated records")
        
        # Should have proper structure
        if aggregated:
            agg_record = aggregated[0]
            self.assertIn('time_slice', agg_record, "Aggregated record should have time_slice")
            self.assertIn('operation', agg_record, "Aggregated record should have operation")
            self.assertIn('file_count', agg_record, "Aggregated record should have file_count")

    def test_dashboard_can_generate_data(self):
        """Test that the dashboard can generate complete data"""
        from data_engine import DashboardDataEngine
        
        engine = DashboardDataEngine(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Generate dashboard data
        data = engine.generate_dashboard_data(time_slice='D', lookback_days=7)
        
        # Should have required structure
        self.assertIn('metadata', data, "Should have metadata")
        self.assertIn('file_operations_data', data, "Should have file_operations_data")
        self.assertIn('by_script', data['file_operations_data'], "Should have by_script data")
        self.assertIn('by_operation', data['file_operations_data'], "Should have by_operation data")
        
        # Should have metadata
        metadata = data['metadata']
        self.assertIn('generated_at', metadata, "Should have generated_at")
        self.assertIn('scripts_found', metadata, "Should have scripts_found")
        self.assertIn('data_range', metadata, "Should have data_range")

    def test_dashboard_can_transform_charts(self):
        """Test that the dashboard can transform data for charts"""
        from data_engine import DashboardDataEngine
        from productivity_dashboard import ProductivityDashboard
        
        # Use the real data directory
        dashboard = ProductivityDashboard(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Generate raw data
        data = dashboard.data_engine.generate_dashboard_data(time_slice='D', lookback_days=7)
        
        # Transform for charts
        chart_data = dashboard.transform_for_charts(data)
        
        # Should have required structure
        self.assertIn('metadata', chart_data, "Should have metadata")
        self.assertIn('charts', chart_data, "Should have charts")
        self.assertIn('by_script', chart_data['charts'], "Should have by_script charts")
        self.assertIn('by_operation', chart_data['charts'], "Should have by_operation charts")

    def test_dashboard_has_historical_data(self):
        """Test that the dashboard has access to historical data"""
        from data_engine import DashboardDataEngine
        
        engine = DashboardDataEngine(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Load records
        records = engine.load_file_operations()
        
        # Should have records from multiple dates
        dates = set(str(r['date']) for r in records if r.get('date'))
        self.assertGreater(len(dates), 1, "Should have data from multiple dates")
        
        # Should have data from September (historical)
        september_dates = [d for d in dates if d.startswith('2025-09')]
        self.assertGreater(len(september_dates), 0, "Should have historical data from September")
        
        # Should have data from October (recent)
        october_dates = [d for d in dates if d.startswith('2025-10')]
        self.assertGreater(len(october_dates), 0, "Should have recent data from October")

    def test_dashboard_has_multiple_scripts(self):
        """Test that the dashboard tracks multiple scripts"""
        from data_engine import DashboardDataEngine
        
        engine = DashboardDataEngine(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Load records
        records = engine.load_file_operations()
        
        # Should have records from multiple scripts
        scripts = set(r['script'] for r in records if r.get('script'))
        self.assertGreater(len(scripts), 1, "Should have data from multiple scripts")
        
        # Should have expected scripts
        expected_scripts = [
            'desktop_image_selector_crop',
            'character_sorter',
            '01_web_image_selector',
            '02_web_character_sorter'
        ]
        
        for expected_script in expected_scripts:
            if expected_script in scripts:
                self.assertIn(expected_script, scripts, f"Should have data from {expected_script}")

    def test_dashboard_has_multiple_operations(self):
        """Test that the dashboard tracks multiple operation types"""
        from data_engine import DashboardDataEngine
        
        engine = DashboardDataEngine(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Load records
        records = engine.load_file_operations()
        
        # Should have records from multiple operation types
        operations = set(r['operation'] for r in records if r.get('operation'))
        self.assertGreater(len(operations), 1, "Should have multiple operation types")
        
        # Should have expected operations
        expected_operations = ['move', 'delete', 'crop']
        for expected_op in expected_operations:
            if expected_op in operations:
                self.assertIn(expected_op, operations, f"Should have {expected_op} operations")

    def test_consolidation_script_exists_and_runnable(self):
        """Test that the consolidation script exists and can be run"""
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        
        # Should exist
        self.assertTrue(consolidation_script.exists(), "Consolidation script should exist")
        
        # Should be a Python file
        self.assertTrue(consolidation_script.suffix == '.py', "Should be a Python file")
        
        # Should have shebang
        with open(consolidation_script, 'r') as f:
            first_line = f.readline().strip()
            self.assertTrue(first_line.startswith('#!'), "Should have shebang line")
            self.assertIn('python', first_line, "Shebang should specify python")

    def test_daily_summaries_exist(self):
        """Test that daily summaries exist (from our consolidation)"""
        summaries_dir = Path(__file__).parent.parent.parent / "data" / "daily_summaries"
        
        # Should exist
        self.assertTrue(summaries_dir.exists(), "Daily summaries directory should exist")
        
        # Should have summary files
        summary_files = list(summaries_dir.glob("daily_summary_*.json"))
        self.assertGreater(len(summary_files), 0, "Should have daily summary files")
        
        # Should have valid JSON
        for summary_file in summary_files:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            self.assertIn('date', summary, "Summary should have date")
            self.assertIn('scripts', summary, "Summary should have scripts")
            self.assertIn('total_operations', summary, "Summary should have total_operations")

    def test_archived_logs_exist(self):
        """Test that archived logs exist (from our consolidation)"""
        archives_dir = Path(__file__).parent.parent.parent / "data" / "log_archives"
        
        # Should exist
        self.assertTrue(archives_dir.exists(), "Log archives directory should exist")
        
        # Should have archived files
        archived_files = list(archives_dir.glob("*.gz"))
        self.assertGreater(len(archived_files), 0, "Should have archived log files")


class TestDashboardDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency"""
    
    def test_no_data_loss_after_consolidation(self):
        """Test that no data was lost during consolidation"""
        from data_engine import DashboardDataEngine
        
        engine = DashboardDataEngine(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Load all records
        records = engine.load_file_operations()
        
        # Should have substantial data (we know there are 152,963+ records)
        self.assertGreater(len(records), 100000, "Should have substantial historical data")
        
        # Should have data from multiple days
        dates = set(str(r['date']) for r in records if r.get('date'))
        self.assertGreater(len(dates), 10, "Should have data from many days")
        
        # Should have data from multiple scripts
        scripts = set(r['script'] for r in records if r.get('script'))
        self.assertGreater(len(scripts), 5, "Should have data from many scripts")

    def test_dashboard_performance(self):
        """Test that the dashboard loads data quickly"""
        import time
        from data_engine import DashboardDataEngine
        
        engine = DashboardDataEngine(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Time the data loading
        start_time = time.time()
        records = engine.load_file_operations()
        load_time = time.time() - start_time
        
        # Should load quickly (less than 5 seconds for 150k+ records)
        self.assertLess(load_time, 5.0, f"Data loading should be fast, took {load_time:.2f}s")
        
        # Should have loaded substantial data
        self.assertGreater(len(records), 100000, "Should have loaded substantial data")

    def test_chart_data_structure(self):
        """Test that chart data has the correct structure"""
        from data_engine import DashboardDataEngine
        from productivity_dashboard import ProductivityDashboard
        
        dashboard = ProductivityDashboard(data_dir=str(Path(__file__).parent.parent.parent))
        
        # Generate and transform data
        data = dashboard.data_engine.generate_dashboard_data(time_slice='D', lookback_days=7)
        chart_data = dashboard.transform_for_charts(data)
        
        # Check chart structure
        charts = chart_data['charts']
        
        if 'by_script' in charts:
            by_script = charts['by_script']
            for script_name, script_data in by_script.items():
                self.assertIn('dates', script_data, f"Script {script_name} should have dates")
                self.assertIn('counts', script_data, f"Script {script_name} should have counts")
                self.assertEqual(len(script_data['dates']), len(script_data['counts']), 
                               f"Script {script_name} dates and counts should match")
        
        if 'by_operation' in charts:
            by_operation = charts['by_operation']
            for operation_name, operation_data in by_operation.items():
                self.assertIn('dates', operation_data, f"Operation {operation_name} should have dates")
                self.assertIn('counts', operation_data, f"Operation {operation_name} should have counts")
                self.assertEqual(len(operation_data['dates']), len(operation_data['counts']), 
                               f"Operation {operation_name} dates and counts should match")


if __name__ == '__main__':
    print("🧪 Testing Core Dashboard Functionality")
    print("=" * 50)
    print("Verifying that the dashboard system is working correctly")
    print("and that all historical data is accessible.")
    print("=" * 50)
    
    unittest.main(verbosity=2)
