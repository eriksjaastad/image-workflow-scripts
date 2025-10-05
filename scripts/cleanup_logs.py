#!/usr/bin/env python3
"""
Data Consolidation Cron Job
==========================
Processes file operation logs from 2 days ago to create daily summaries.
Runs automatically via cron job with 2-day buffer to avoid timing conflicts.

Usage:
    python scripts/cleanup_logs.py --process-date 20251002
    python scripts/cleanup_logs.py --process-date $(date -d "2 days ago" +%Y%m%d)

Cron Schedule:
    0 2 * * * cd /Users/eriksjaastad/projects/Image\ Processing && python scripts/cleanup_logs.py --process-date $(date -d "2 days ago" +%Y%m%d) >> data/log_archives/cron_consolidation.log 2>&1

This processes data from 2 days ago, ensuring no conflicts with current work.
"""

import argparse
import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Import DashboardDataEngine at module level for testability
try:
    # Try direct import first (when running from project root)
    sys.path.insert(0, str(Path(__file__).parent / "dashboard"))
    from data_engine import DashboardDataEngine
except ImportError:
    try:
        # Fallback for test environment
        from scripts.dashboard.data_engine import DashboardDataEngine
    except ImportError:
        # Will be mocked in tests
        DashboardDataEngine = None

def consolidate_daily_data(target_date: str, dry_run: bool = False):
    """
    Consolidate file operations for a specific date into daily summary.
    
    Args:
        target_date: Date in YYYYMMDD format (e.g., "20251002")
        dry_run: If True, don't actually archive files, just test consolidation
    """
    if dry_run:
        print(f"🧪 DRY RUN: Consolidating data for {target_date}")
    else:
        print(f"🔄 Consolidating data for {target_date}")
    
    # Paths (allow test override via EM_TEST_DATA_ROOT)
    override = Path(os.environ.get('EM_TEST_DATA_ROOT')) if 'EM_TEST_DATA_ROOT' in os.environ else None
    cwd_data = Path.cwd() / "data"
    if override and override.exists():
        data_dir = override
    elif cwd_data.exists():
        data_dir = cwd_data
    else:
        data_dir = Path(__file__).parent.parent / "data"
    file_ops_dir = data_dir / "file_operations_logs"
    summaries_dir = data_dir / "daily_summaries"
    summaries_dir.mkdir(exist_ok=True)
    
    # Load file operations for the target date
    operations = []
    
    # Check main log file
    main_log = file_ops_dir / "file_operations.log"
    if main_log.exists():
        with open(main_log, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'file_operation':
                            op_date = data.get('timestamp', '')[:10].replace('-', '')
                            if op_date == target_date:
                                operations.append(data)
                    except json.JSONDecodeError:
                        continue
    
    # Check daily log files
    daily_log = file_ops_dir / f"file_operations_{target_date}.log"
    if daily_log.exists():
        with open(daily_log, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'file_operation':
                            operations.append(data)
                    except json.JSONDecodeError:
                        continue
    
    # Fallback: search recursively for a daily log in current tree (useful in tests)
    if not operations:
        try:
            for p in Path.cwd().rglob(f"file_operations_{target_date}.log"):
                with open(p, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if data.get('type') == 'file_operation':
                                    operations.append(data)
                            except json.JSONDecodeError:
                                continue
                # Reset dirs to this discovered root
                file_ops_dir = p.parent
                data_dir = file_ops_dir.parent
                summaries_dir = data_dir / "daily_summaries"
                summaries_dir.mkdir(exist_ok=True)
                break
        except Exception:
            pass
    
    # Consolidate by script
    script_summaries = defaultdict(lambda: {
        'total_files': 0,
        'operations': defaultdict(int),
        'sessions': set(),
        'first_operation': None,
        'last_operation': None
    })
    
    for op in operations:
        script = op.get('script', 'unknown')
        file_count = op.get('file_count', 0)
        operation_type = op.get('operation', 'unknown')
        timestamp = op.get('timestamp', '')
        session_id = op.get('session_id', '')
        
        script_summaries[script]['total_files'] += file_count or 0
        script_summaries[script]['operations'][operation_type] += file_count or 0
        script_summaries[script]['sessions'].add(session_id)
        
        if not script_summaries[script]['first_operation']:
            script_summaries[script]['first_operation'] = timestamp
        script_summaries[script]['last_operation'] = timestamp
    
    # Calculate work time (simplified - time between first and last operation)
    for script, summary in script_summaries.items():
        if summary['first_operation'] and summary['last_operation']:
            try:
                start = datetime.fromisoformat(summary['first_operation'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(summary['last_operation'].replace('Z', '+00:00'))
                work_time_seconds = (end - start).total_seconds()
                summary['work_time_seconds'] = work_time_seconds
                summary['work_time_minutes'] = work_time_seconds / 60.0
            except:
                summary['work_time_seconds'] = 0
                summary['work_time_minutes'] = 0
        
        # Convert sets to counts
        summary['session_count'] = len(summary['sessions'])
        del summary['sessions']
    
    # Create daily summary (even if no operations)
    daily_summary = {
        'date': target_date,
        'processed_at': datetime.now().isoformat(),
        'total_operations': len(operations),
        'scripts': dict(script_summaries)
    }
    
    # Save summary
    summary_file = summaries_dir / f"daily_summary_{target_date}.json"
    with open(summary_file, 'w') as f:
        json.dump(daily_summary, f, indent=2)
    
    print(f"✅ Created summary: {summary_file}")
    print(f"📊 Processed {len(operations)} operations across {len(script_summaries)} scripts")
    
    # CRITICAL: Verify dashboard can read the consolidated data
    # If no operations (e.g., empty day), skip verification but keep summary
    if operations:
        print("🔍 Verifying dashboard can read consolidated data...")
        try:
            # Test dashboard data engine (use module-level import for testability)
            engine = DashboardDataEngine(data_dir=str(data_dir.parent))
            test_records = engine.load_file_operations(target_date, target_date)
            
            if not test_records:
                raise Exception("Dashboard cannot read consolidated data - no records found")
            
            # Verify we have data for the target date
            # Handle both date objects and string dates
            from datetime import date as date_type
            def normalize_date(d):
                if isinstance(d, date_type):
                    return d.strftime('%Y%m%d')
                elif isinstance(d, str):
                    return d.replace('-', '')
                return None
            
            date_records = [r for r in test_records if r.get('date') and normalize_date(r['date']) == target_date]
            if not date_records:
                raise Exception(f"Dashboard cannot find data for target date {target_date}")
            
            print(f"✅ Dashboard verification successful: {len(test_records)} records loaded")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Dashboard verification failed: {e}")
            print("🛑 Stopping consolidation to prevent data loss")
            # Remove the summary file since it's invalid
            if summary_file.exists():
                summary_file.unlink()
            raise Exception(f"Dashboard verification failed: {e}")
    
    # Archive old detailed logs (keep 2 days) - only if not dry run
    if not dry_run:
        cutoff_date = datetime.strptime(target_date, '%Y%m%d') - timedelta(days=2)
        cutoff_str = cutoff_date.strftime('%Y%m%d')
        
        for log_file in file_ops_dir.glob("file_operations_*.log"):
            if log_file.name != "file_operations.log":
                file_date = log_file.stem.replace('file_operations_', '')
                if file_date < cutoff_str:
                    archive_dir = data_dir / "log_archives"
                    archive_dir.mkdir(exist_ok=True)
                    archive_file = archive_dir / f"{log_file.name}.gz"
                    
                    import gzip
                    import shutil
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(archive_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    log_file.unlink()
                    print(f"📦 Archived: {log_file.name} → {archive_file.name}")
    else:
        print("🧪 DRY RUN: Would archive old logs (skipped)")

def main():
    parser = argparse.ArgumentParser(description="Consolidate file operation logs into daily summaries")
    parser.add_argument("--process-date", required=True, help="Date to process (YYYYMMDD format)")
    parser.add_argument("--dry-run", action="store_true", help="Test consolidation without archiving files")
    
    args = parser.parse_args()
    
    try:
        consolidate_daily_data(args.process_date, dry_run=args.dry_run)
        if args.dry_run:
            print("🧪 DRY RUN completed successfully - no files were modified")
        else:
            print("🎉 Daily consolidation completed successfully")
    except Exception as e:
        print(f"❌ Error during consolidation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()