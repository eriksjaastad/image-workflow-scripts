#!/usr/bin/env python3
"""
Productivity Dashboard - Core Data Engine (Simple Version)
==========================================================
Processes ActivityTimer and FileTracker logs into dashboard-ready data.
No external dependencies - uses only Python standard library.

Features:
- Modular script detection (handles script renames/additions)
- Flexible time slice aggregation (15min, 1hr, daily, weekly, monthly)
- Historical average calculations for "cloud" overlays
- Script update correlation tracking
- Standardized output format for dashboard consumption
"""

import json
import glob
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.companion_file_utils import calculate_work_time_from_file_operations, get_file_operation_metrics
from collections import defaultdict

class DashboardDataEngine:
    def __init__(self, data_dir: str = ".."):
        self.data_dir = Path(data_dir)
        self.timer_data_dir = self.data_dir / "data" / "timer_data"
        self.file_ops_dir = self.data_dir / "data" / "file_operations_logs"
        
        # Script update tracking (in dashboard directory)
        self.script_updates_file = Path(__file__).parent / "script_updates.csv"
        
        # Cache for processed data
        self._cache = {}
        
    def get_display_name(self, script_name: str) -> str:
        """Convert script filename to human-readable display name"""
        display_names = {
            # Current script names
            '01_web_image_selector': 'Web Image Selector',
            '01_desktop_image_selector_crop': 'Desktop Image Selector Crop',
            '02_web_character_sorter': 'Web Character Sorter',
            '03_web_character_sorter': 'Web Character Sorter',
            '04_multi_crop_tool': 'Multi Crop Tool',
            '04_batch_crop_tool': 'Multi Crop Tool',
            '05_web_multi_directory_viewer': 'Multi Directory Viewer',
            '06_web_duplicate_finder': 'Duplicate Finder',
            
            # Log script names (from actual data)
            'desktop_image_selector_crop': 'Desktop Image Selector Crop',
            'character_sorter': 'Character Sorter',
            'multi_directory_viewer': 'Multi Directory Viewer',
            'image_version_selector': 'Web Image Selector',
            'multi_batch_crop_tool': 'Multi Crop Tool',
            
            # Legacy names
            'batch_crop_tool': 'Multi Crop Tool'
        }
        
        return display_names.get(script_name, script_name.replace('_', ' ').title())
        
    def discover_scripts(self) -> List[str]:
        """Dynamically discover all scripts that have generated data"""
        scripts = set()
        
        # From ActivityTimer data
        if self.timer_data_dir.exists():
            for daily_file in self.timer_data_dir.glob("daily_*.json"):
                try:
                    with open(daily_file, 'r') as f:
                        data = json.load(f)
                        for script_name in data.get('scripts', {}):
                            scripts.add(script_name)
                except Exception as e:
                    print(f"Warning: Could not read {daily_file}: {e}")
        
        # From FileTracker logs
        if self.file_ops_dir.exists():
            for log_file in self.file_ops_dir.glob("*.log"):
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                if 'script' in data:
                                    scripts.add(data['script'])
                except Exception as e:
                    print(f"Warning: Could not read {log_file}: {e}")
        
        return sorted(list(scripts))
    
    def load_activity_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Load and process ActivityTimer data"""
        records = []
        
        if not self.timer_data_dir.exists():
            return records
        
        for daily_file in self.timer_data_dir.glob("daily_*.json"):
            date_str = daily_file.stem.replace('daily_', '')
            
            # Filter by date range if specified
            if start_date and date_str < start_date.replace('-', ''):
                continue
            if end_date and date_str > end_date.replace('-', ''):
                continue
                
            try:
                with open(daily_file, 'r') as f:
                    data = json.load(f)
                    
                    # Process each script's data
                    for script_name, script_data in data.get('scripts', {}).items():
                        for session in script_data.get('sessions', []):
                            record = {
                                'date': date_str,
                                'script': script_name,
                                'session_id': session.get('session_id'),
                                'start_time': session.get('start_time'),
                                'end_time': session.get('end_time'),
                                'active_time': session.get('active_time', 0),
                                'total_time': session.get('total_time', 0),
                                'efficiency': session.get('efficiency', 0),
                                'files_processed': session.get('files_processed', 0),
                                'operations': session.get('operations', {}),
                                'batches_completed': session.get('batches_completed', 0)
                            }
                            
                            # Convert timestamps to datetime objects
                            if record['start_time']:
                                record['start_datetime'] = datetime.fromtimestamp(record['start_time'])
                            if record['end_time']:
                                record['end_datetime'] = datetime.fromtimestamp(record['end_time'])
                            
                            records.append(record)
            except Exception as e:
                print(f"Warning: Could not process {daily_file}: {e}")
        
        return records
    
    def calculate_file_operation_work_time(self, file_operations: List[Dict], break_threshold_minutes: int = 5) -> Dict[str, Any]:
        """
        Calculate work time from file operations using intelligent break detection.
        
        This method is used for file-heavy tools that no longer use ActivityTimer.
        It analyzes FileTracker logs to determine actual work time.
        
        Args:
            file_operations: List of file operation dictionaries from FileTracker
            break_threshold_minutes: Minutes of inactivity considered a break (default: 5)
            
        Returns:
            Dictionary with work time metrics
        """
        if not file_operations:
            return {
                'work_time_seconds': 0.0,
                'work_time_minutes': 0.0,
                'total_operations': 0,
                'files_processed': 0,
                'efficiency_score': 0.0,
                'timing_method': 'file_operations'
            }
        
        # Use the centralized function from companion_file_utils
        metrics = get_file_operation_metrics(file_operations)
        
        # Add timing method identifier
        metrics['timing_method'] = 'file_operations'
        
        return metrics
    
    def get_combined_timing_data(self, script_name: str, date: str) -> Dict[str, Any]:
        """
        Get combined timing data for a script on a specific date.
        
        For file-heavy tools, uses file-operation timing.
        For scroll-heavy tools, uses ActivityTimer data.
        
        Args:
            script_name: Name of the script
            date: Date string (YYYYMMDD format)
            
        Returns:
            Dictionary with combined timing data
        """
        # Define which tools use which timing method
        file_heavy_tools = {
            '01_web_image_selector',
            '01_desktop_image_selector_crop', 
            '02_web_character_sorter',
            '04_multi_crop_tool'
        }
        
        scroll_heavy_tools = {
            '05_web_multi_directory_viewer',
            '06_web_duplicate_finder'
        }
        
        # Get file operations for this script and date
        file_ops = self.load_file_operations(date, date)
        script_file_ops = [op for op in file_ops if op.get('script') == script_name]
        
        if script_name in file_heavy_tools:
            # Use file-operation timing
            return self.calculate_file_operation_work_time(script_file_ops)
        elif script_name in scroll_heavy_tools:
            # Use ActivityTimer data
            timer_data = self.load_activity_data(date, date)
            script_timer_data = [t for t in timer_data if t.get('script') == script_name]
            
            if script_timer_data:
                # Calculate total work time from timer data
                total_work_time = sum(session.get('active_time', 0) for session in script_timer_data)
                total_files = sum(session.get('files_processed', 0) for session in script_timer_data)
                
                return {
                    'work_time_seconds': total_work_time,
                    'work_time_minutes': total_work_time / 60.0,
                    'total_operations': len(script_timer_data),
                    'files_processed': total_files,
                    'efficiency_score': total_files / (total_work_time / 60.0) if total_work_time > 0 else 0.0,
                    'timing_method': 'activity_timer'
                }
            else:
                return {
                    'work_time_seconds': 0.0,
                    'work_time_minutes': 0.0,
                    'total_operations': 0,
                    'files_processed': 0,
                    'efficiency_score': 0.0,
                    'timing_method': 'activity_timer'
                }
        else:
            # Unknown script - try file operations first, fallback to timer
            if script_file_ops:
                return self.calculate_file_operation_work_time(script_file_ops)
            else:
                return {
                    'work_time_seconds': 0.0,
                    'work_time_minutes': 0.0,
                    'total_operations': 0,
                    'files_processed': 0,
                    'efficiency_score': 0.0,
                    'timing_method': 'unknown'
                }
    
    def load_file_operations(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Load and process daily summaries (consolidated data) with fallback to detailed logs"""
        records = []
        
        # Load from both sources and merge
        summaries_dir = self.data_dir / "data" / "daily_summaries"
        if summaries_dir.exists():
            records.extend(self._load_from_daily_summaries(summaries_dir, start_date, end_date))
        
        # Always also load from detailed logs to get historical data
        records.extend(self._load_from_detailed_logs(start_date, end_date))
        
        return records
    
    def _load_from_daily_summaries(self, summaries_dir: Path, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Load records from daily summary files"""
        records = []
        
        for summary_file in summaries_dir.glob("daily_summary_*.json"):
            # Extract date from filename
            date_match = re.search(r'(\d{8})', summary_file.name)
            if not date_match:
                continue
                
            file_date = date_match.group(1)
            if start_date and file_date < start_date.replace('-', ''):
                continue
            if end_date and file_date > end_date.replace('-', ''):
                continue
            
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                    
                # Convert daily summary to individual records for compatibility
                for script_name, script_data in summary_data.get('scripts', {}).items():
                    # Create records for each operation type
                    for operation_type, file_count in script_data.get('operations', {}).items():
                        if file_count > 0:
                            record = {
                                'timestamp_str': summary_data['date'] + 'T00:00:00Z',
                                'script': script_name,
                                'session_id': f"daily_{summary_data['date']}",
                                'operation': operation_type,
                                'source_dir': None,
                                'dest_dir': None,
                                'file_count': file_count,
                                'files': [],
                                'notes': f"Daily summary for {summary_data['date']}",
                                'work_time_seconds': script_data.get('work_time_seconds', 0),
                                'work_time_minutes': script_data.get('work_time_minutes', 0)
                            }
                            
                            # Convert timestamp
                            try:
                                record['timestamp'] = datetime.fromisoformat(record['timestamp_str'])
                                record['date'] = record['timestamp'].date()
                            except:
                                record['timestamp'] = None
                                record['date'] = None
                            
                            records.append(record)
                            
            except Exception as e:
                print(f"Warning: Could not process {summary_file}: {e}")
        
        return records
    
    def _load_from_detailed_logs(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Fallback: Load records from detailed log files"""
        records = []
        
        file_ops_dir = self.data_dir / "data" / "file_operations_logs"
        if not file_ops_dir.exists():
            return records
        
        # Check both regular log files and archived files
        log_files = list(file_ops_dir.glob("*.log"))
        archive_dir = self.data_dir / "data" / "log_archives"
        if archive_dir.exists():
            log_files.extend(archive_dir.glob("*.gz"))
        
        for log_file in log_files:
            # Extract date from filename if possible
            date_match = re.search(r'(\d{8})', log_file.name)
            if date_match:
                file_date = date_match.group(1)
                if start_date and file_date < start_date.replace('-', ''):
                    continue
                if end_date and file_date > end_date.replace('-', ''):
                    continue
            
            try:
                # Handle compressed files
                if log_file.suffix == '.gz':
                    import gzip
                    with gzip.open(log_file, 'rt') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                if data.get('type') == 'file_operation':
                                    record = {
                                        'timestamp_str': data.get('timestamp'),
                                        'script': data.get('script'),
                                        'session_id': data.get('session_id'),
                                        'operation': data.get('operation'),
                                        'source_dir': data.get('source_dir'),
                                        'dest_dir': data.get('dest_dir'),
                                        'file_count': data.get('file_count', 0),
                                        'files': data.get('files', []),
                                        'notes': data.get('notes', '')
                                    }
                                    
                                    # Convert timestamp
                                    try:
                                        record['timestamp'] = datetime.fromisoformat(record['timestamp_str'])
                                        record['date'] = record['timestamp'].date()
                                    except:
                                        record['timestamp'] = None
                                        record['date'] = None
                                    
                                    records.append(record)
                else:
                    # Handle regular files
                    with open(log_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                if data.get('type') == 'file_operation':
                                    record = {
                                        'timestamp_str': data.get('timestamp'),
                                        'script': data.get('script'),
                                        'session_id': data.get('session_id'),
                                        'operation': data.get('operation'),
                                        'source_dir': data.get('source_dir'),
                                        'dest_dir': data.get('dest_dir'),
                                        'file_count': data.get('file_count', 0),
                                        'files': data.get('files', []),
                                        'notes': data.get('notes', '')
                                    }
                                    
                                    # Convert timestamp
                                    try:
                                        record['timestamp'] = datetime.fromisoformat(record['timestamp_str'])
                                        record['date'] = record['timestamp'].date()
                                    except:
                                        record['timestamp'] = None
                                        record['date'] = None
                                    
                                    records.append(record)
            except Exception as e:
                print(f"Warning: Could not process {log_file}: {e}")
        
        return records
    
    def aggregate_by_time_slice(self, records: List[Dict], time_slice: str, 
                               value_field: str, group_field: str = 'script') -> List[Dict]:
        """
        Aggregate data by time slices for dashboard visualization
        
        Args:
            records: Input records
            time_slice: '15min', '1H', 'D' (daily), 'W' (weekly), 'M' (monthly)
            value_field: Field to aggregate (e.g., 'active_time', 'file_count')
            group_field: Grouping field (e.g., 'script', 'operation')
        """
        if not records:
            return []
        
        # Group by time slice and group field
        aggregated = defaultdict(lambda: defaultdict(float))
        
        for record in records:
            # Get time slice key
            time_key = self._get_time_slice_key(record, time_slice)
            if not time_key:
                continue
            
            group_key = record.get(group_field, 'unknown')
            
            # Convert script names to display names for better readability
            if group_field == 'script':
                group_key = self.get_display_name(group_key)
            
            value = record.get(value_field, 0)
            if value is None:
                value = 0
            
            aggregated[time_key][group_key] += value
        
        # Convert to list format
        result = []
        for time_key, groups in aggregated.items():
            for group_key, value in groups.items():
                result.append({
                    'time_slice': time_key,
                    group_field: group_key,
                    value_field: value
                })
        
        return sorted(result, key=lambda x: x['time_slice'])
    
    def _get_time_slice_key(self, record: Dict, time_slice: str) -> Optional[str]:
        """Get time slice key for a record"""
        # Determine which timestamp to use
        timestamp = None
        if 'timestamp' in record and record['timestamp']:
            timestamp = record['timestamp']
        elif 'start_datetime' in record and record['start_datetime']:
            timestamp = record['start_datetime']
        
        if not timestamp:
            return None
        
        # Generate time slice key
        if time_slice == '15min':
            # Round to 15-minute intervals
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0).isoformat()
        elif time_slice == '1H':
            # Round to hour
            return timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
        elif time_slice == 'D':
            # Daily
            return timestamp.date().isoformat()
        elif time_slice == 'W':
            # Weekly (Monday as start of week)
            days_since_monday = timestamp.weekday()
            monday = timestamp - timedelta(days=days_since_monday)
            return monday.date().isoformat()
        elif time_slice == 'M':
            # Monthly
            return f"{timestamp.year}-{timestamp.month:02d}-01"
        
        return None
    
    def calculate_historical_averages(self, records: List[Dict], time_slice: str, 
                                    value_field: str, group_field: str = 'script',
                                    lookback_days: int = 30) -> List[Dict]:
        """Calculate historical averages for 'cloud' overlay backgrounds"""
        if not records:
            return []
        
        # Filter to historical data (exclude recent data)
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        historical_records = []
        
        for record in records:
            timestamp = None
            if 'timestamp' in record and record['timestamp']:
                timestamp = record['timestamp']
            elif 'start_datetime' in record and record['start_datetime']:
                timestamp = record['start_datetime']
            
            if timestamp and timestamp < cutoff_date:
                historical_records.append(record)
        
        if not historical_records:
            return []
        
        # Aggregate historical data
        aggregated = self.aggregate_by_time_slice(historical_records, time_slice, value_field, group_field)
        
        # Calculate averages by time pattern
        pattern_groups = defaultdict(lambda: defaultdict(list))
        
        for record in aggregated:
            time_pattern = self._get_time_pattern(record['time_slice'], time_slice)
            group_key = record[group_field]
            value = record[value_field]
            
            pattern_groups[time_pattern][group_key].append(value)
        
        # Calculate averages
        averages = []
        for time_pattern, groups in pattern_groups.items():
            for group_key, values in groups.items():
                avg_value = sum(values) / len(values) if values else 0
                averages.append({
                    'time_pattern': time_pattern,
                    group_field: group_key,
                    f'{value_field}_avg': avg_value
                })
        
        return averages
    
    def _get_time_pattern(self, time_slice_key: str, time_slice: str) -> str:
        """Get time pattern for historical averaging"""
        try:
            if time_slice in ['15min', '1H']:
                # Intraday: pattern by hour of day
                dt = datetime.fromisoformat(time_slice_key)
                return str(dt.hour)
            elif time_slice == 'D':
                # Daily: pattern by day of week
                dt = datetime.fromisoformat(time_slice_key)
                return str(dt.weekday())
            elif time_slice == 'W':
                # Weekly: pattern by week of year (simplified)
                dt = datetime.fromisoformat(time_slice_key)
                return str(dt.isocalendar()[1])
            else:
                # Monthly: pattern by month
                dt = datetime.fromisoformat(time_slice_key)
                return str(dt.month)
        except:
            return 'unknown'
    
    def load_script_updates(self) -> List[Dict]:
        """Load script update tracking data"""
        if not self.script_updates_file.exists():
            # Create empty file with headers
            with open(self.script_updates_file, 'w') as f:
                f.write("date,script,description\n")
            return []
        
        try:
            updates = []
            with open(self.script_updates_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.strip().split(',', 2)
                        if len(parts) >= 3:
                            updates.append({
                                'date': parts[0],
                                'script': parts[1],
                                'description': parts[2]
                            })
            return updates
        except Exception as e:
            print(f"Warning: Could not load script updates: {e}")
            return []
    
    def add_script_update(self, script: str, description: str, date: Optional[str] = None):
        """Add a script update entry"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Append to CSV
        with open(self.script_updates_file, 'a') as f:
            f.write(f"{date},{script},{description}\n")
        
        print(f"Added script update: {script} - {description}")
    
    def generate_dashboard_data(self, time_slice: str = 'D', lookback_days: int = 30, 
                               production_scripts: List[str] = None) -> Dict[str, Any]:
        """
        Generate complete dashboard data package
        
        Returns:
            Dictionary with all data needed for dashboard visualization
        """
        print(f"Generating dashboard data for {time_slice} time slice...")
        
        # Define production workflow scripts for dashboard filtering
        # Map current script names to log names
        if production_scripts is None:
            production_scripts = [
                # Current script names
                '01_web_image_selector',
                '01_desktop_image_selector_crop',
                '03_web_character_sorter',
                '04_multi_crop_tool',
                
                # Log script names (from actual data)
                'desktop_image_selector_crop',
                'character_sorter',
                'image_version_selector',
                'multi_batch_crop_tool',
                
                # Legacy names
                'batch_crop_tool'
            ]
        
        # Load raw data
        activity_records = self.load_activity_data()
        file_ops_records = self.load_file_operations()
        script_updates = self.load_script_updates()
        
        # Filter to production scripts only for dashboard
        activity_records = [r for r in activity_records if r.get('script') in production_scripts]
        file_ops_records = [r for r in file_ops_records if r.get('script') in production_scripts]
        
        # Discover all scripts
        scripts = self.discover_scripts()
        
        # Calculate date ranges
        activity_dates = [r.get('date') for r in activity_records if r.get('date')]
        file_ops_dates = [r.get('date') for r in file_ops_records if r.get('date')]
        
        dashboard_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'time_slice': time_slice,
                'lookback_days': lookback_days,
                'scripts_found': scripts,
                'data_range': {
                    'activity_start': min(activity_dates) if activity_dates else None,
                    'activity_end': max(activity_dates) if activity_dates else None,
                    'file_ops_start': min(file_ops_dates) if file_ops_dates else None,
                    'file_ops_end': max(file_ops_dates) if file_ops_dates else None
                }
            },
            'activity_data': {},
            'file_operations_data': {},
            'historical_averages': {},
            'script_updates': script_updates
        }
        
        # Process activity data
        if activity_records:
            # Active time by script
            dashboard_data['activity_data']['active_time'] = self.aggregate_by_time_slice(
                activity_records, time_slice, 'active_time', 'script'
            )
            
            # Files processed by script
            dashboard_data['activity_data']['files_processed'] = self.aggregate_by_time_slice(
                activity_records, time_slice, 'files_processed', 'script'
            )
            
            # Efficiency by script
            dashboard_data['activity_data']['efficiency'] = self.aggregate_by_time_slice(
                activity_records, time_slice, 'efficiency', 'script'
            )
            
            # Historical averages
            dashboard_data['historical_averages']['active_time'] = self.calculate_historical_averages(
                activity_records, time_slice, 'active_time', 'script', lookback_days
            )
            
            dashboard_data['historical_averages']['files_processed'] = self.calculate_historical_averages(
                activity_records, time_slice, 'files_processed', 'script', lookback_days
            )
        
        # Process file operations data
        if file_ops_records:
            # File operations by script
            dashboard_data['file_operations_data']['by_script'] = self.aggregate_by_time_slice(
                file_ops_records, time_slice, 'file_count', 'script'
            )
            
            # File operations by type
            dashboard_data['file_operations_data']['by_operation'] = self.aggregate_by_time_slice(
                file_ops_records, time_slice, 'file_count', 'operation'
            )
            
            # Deleted files specifically (for granular tracking)
            delete_ops = [r for r in file_ops_records if r.get('operation') in ['delete', 'send_to_trash']]
            if delete_ops:
                dashboard_data['file_operations_data']['deletions'] = self.aggregate_by_time_slice(
                    delete_ops, time_slice, 'file_count', 'script'
                )
        
        return dashboard_data

def main():
    """Test the data engine"""
    engine = DashboardDataEngine()
    
    print("🔍 Discovering scripts...")
    scripts = engine.discover_scripts()
    print(f"Found scripts: {scripts}")
    
    print("\n📊 Loading activity data...")
    activity_data = engine.load_activity_data()
    print(f"Activity records: {len(activity_data)}")
    
    print("\n📁 Loading file operations...")
    file_ops_data = engine.load_file_operations()
    print(f"File operation records: {len(file_ops_data)}")
    
    print("\n📈 Generating dashboard data...")
    data = engine.generate_dashboard_data(time_slice='D', lookback_days=7)
    
    print(f"\nMetadata: {data['metadata']}")
    print(f"Activity data keys: {list(data['activity_data'].keys())}")
    print(f"File operations keys: {list(data['file_operations_data'].keys())}")
    print(f"Script updates: {len(data['script_updates'])}")
    
    # Save sample output
    output_file = Path("dashboard_data_sample.json")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n💾 Sample data saved to: {output_file}")

if __name__ == "__main__":
    main()
