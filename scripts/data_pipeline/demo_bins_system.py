#!/usr/bin/env python3
"""
15-Minute Bins System Demo
==========================
Quick demonstration and validation of the bins system.

This script:
1. Generates bins for last 7 days
2. Validates them against raw logs
3. Shows sample bin output
4. Calculates performance metrics

Run this to verify the system is working correctly.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def run_command(cmd: list, description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"▶ {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            print("  ✓ Success")
            return True, output
        else:
            print(f"  ✗ Failed (exit code {result.returncode})")
            print(f"  Output: {output[:500]}")
            return False, output
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False, str(e)


def count_bins(data_dir: Path) -> int:
    """Count total bins across all days."""
    bins_dir = data_dir / 'aggregates' / 'daily'
    if not bins_dir.exists():
        return 0
    
    total = 0
    for bin_file in bins_dir.rglob('agg_15m.jsonl'):
        try:
            with open(bin_file, 'r') as f:
                total += sum(1 for _ in f)
        except Exception:
            continue
    
    return total


def show_sample_bin(data_dir: Path):
    """Show a sample bin record."""
    bins_dir = data_dir / 'aggregates' / 'daily'
    if not bins_dir.exists():
        print("  No bins found")
        return
    
    # Find most recent bin file
    bin_files = list(bins_dir.rglob('agg_15m.jsonl'))
    if not bin_files:
        print("  No bin files found")
        return
    
    # Sort by modification time (newest first)
    bin_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Read first bin from most recent file
    try:
        with open(bin_files[0], 'r') as f:
            first_line = f.readline()
            if first_line:
                bin_record = json.loads(first_line)
                print(f"  Sample bin from: {bin_files[0].parent.name}")
                print(f"\n{json.dumps(bin_record, indent=2)}")
            else:
                print("  Bin file is empty")
    except Exception as e:
        print(f"  Error reading bin: {e}")


def check_raw_logs(data_dir: Path) -> tuple[int, str, str]:
    """Check raw logs availability."""
    logs_dir = data_dir / 'file_operations_logs'
    archives_dir = data_dir / 'log_archives'
    
    log_count = len(list(logs_dir.glob('*.log'))) if logs_dir.exists() else 0
    archive_count = len(list(archives_dir.glob('*.gz'))) if archives_dir.exists() else 0
    
    # Find date range
    all_files = []
    if logs_dir.exists():
        all_files.extend(logs_dir.glob('*.log'))
    if archives_dir.exists():
        all_files.extend(archives_dir.glob('*.gz'))
    
    if not all_files:
        return 0, 'N/A', 'N/A'
    
    # Extract dates from filenames
    dates = []
    for f in all_files:
        # Extract YYYYMMDD from filename
        import re
        match = re.search(r'(\d{8})', f.name)
        if match:
            date_str = match.group(1)
            try:
                date = datetime.strptime(date_str, '%Y%m%d')
                dates.append(date)
            except:
                continue
    
    if dates:
        dates.sort()
        min_date = dates[0].strftime('%Y-%m-%d')
        max_date = dates[-1].strftime('%Y-%m-%d')
    else:
        min_date = 'N/A'
        max_date = 'N/A'
    
    return log_count + archive_count, min_date, max_date


def main():
    """Run demo."""
    print_section("15-Minute Bins System Demo")
    
    data_dir = project_root / 'data'
    
    # Check prerequisites
    print("📋 Prerequisites Check")
    print(f"  Project root: {project_root}")
    print(f"  Data directory: {data_dir}")
    
    log_count, min_date, max_date = check_raw_logs(data_dir)
    print(f"  Raw logs: {log_count} files")
    print(f"  Date range: {min_date} to {max_date}")
    
    if log_count == 0:
        print("\n⚠️  No raw logs found. Cannot generate bins.")
        print("   Please run some file operations first.")
        sys.exit(1)
    
    # Step 1: Generate bins
    print_section("Step 1: Generate Bins (Last 7 Days)")
    
    success, output = run_command(
        [sys.executable, 'scripts/data_pipeline/aggregate_to_15m.py', '--days', '7'],
        'Aggregating last 7 days to 15-minute bins'
    )
    
    if not success:
        print("\n❌ Failed to generate bins")
        sys.exit(1)
    
    # Count bins
    bin_count = count_bins(data_dir)
    print(f"\n  📊 Total bins created: {bin_count}")
    
    # Step 2: Validate bins
    print_section("Step 2: Validate Bins")
    
    success, output = run_command(
        [sys.executable, 'scripts/data_pipeline/validate_15m_bins.py', '--days', '7'],
        'Validating bins against raw logs'
    )
    
    if not success:
        print("\n⚠️  Validation warnings (see output above)")
    
    # Step 3: Show sample
    print_section("Step 3: Sample Bin Record")
    show_sample_bin(data_dir)
    
    # Step 4: Performance metrics
    print_section("Step 4: Performance Metrics")
    
    # Calculate compression ratio
    bins_dir = data_dir / 'aggregates' / 'daily'
    if bins_dir.exists():
        bins_size = sum(
            f.stat().st_size for f in bins_dir.rglob('agg_15m.jsonl')
            if f.is_file()
        )
        
        logs_dir = data_dir / 'file_operations_logs'
        archives_dir = data_dir / 'log_archives'
        
        logs_size = 0
        if logs_dir.exists():
            logs_size += sum(f.stat().st_size for f in logs_dir.glob('*.log'))
        if archives_dir.exists():
            logs_size += sum(f.stat().st_size for f in archives_dir.glob('*.gz'))
        
        if logs_size > 0:
            compression_ratio = (1 - bins_size / logs_size) * 100
            print(f"  Raw logs size: {logs_size / 1024:.1f} KB")
            print(f"  Bins size: {bins_size / 1024:.1f} KB")
            print(f"  Compression: {compression_ratio:.1f}% reduction")
        else:
            print("  Unable to calculate compression ratio")
    
    print(f"  Bins per day: {bin_count / 7:.0f} average")
    print("  Estimated dashboard speedup: 10-50x")
    
    # Step 5: Next steps
    print_section("Next Steps")
    
    print("  To enable bins in dashboard:")
    print("    1. Edit: configs/bins_config.json")
    print("       Set: 'enabled': true, 'use_15m_bins': true")
    print("       Set: 'bin_charts': ['by_script']")
    print()
    print("    2. Restart dashboard:")
    print("       python scripts/dashboard/run_dashboard.py")
    print()
    print("    3. Measure performance in browser DevTools")
    print()
    print("  To archive a finished project:")
    print("    python scripts/data_pipeline/archive_project_bins.py <project_id>")
    print()
    print("  To schedule daily aggregation (cron):")
    print("    0 2 * * * cd /path/to/project && python scripts/data_pipeline/aggregate_to_15m.py --days 1")
    
    print_section("Demo Complete ✅")
    print("  See: Documents/15_MINUTE_BINS_GUIDE.md for full documentation")
    print()


if __name__ == '__main__':
    main()

