#!/usr/bin/env python3
"""
Generate detailed inspection report showing exactly what rows will be backfilled.
This helps verify the script is targeting the correct rows before running the actual backfill.
"""

import csv
from datetime import datetime
from pathlib import Path

# Paths
WORKSPACE = Path(__file__).resolve().parents[2]
CSV_PATH = WORKSPACE / "data" / "training" / "select_crop_log.csv"
REPORT_PATH = WORKSPACE / "data" / "training" / "backfill_inspection_report.txt"

# Date ranges
MOJO1_START = datetime(2025, 10, 1)
MOJO1_END = datetime(2025, 10, 11, 23, 59, 59)
MOJO2_START = datetime(2025, 10, 12)
MOJO2_END = datetime(2025, 10, 20, 23, 59, 59)

def get_project_from_timestamp(timestamp_str):
    """Map timestamp to project"""
    try:
        ts = datetime.fromisoformat(timestamp_str.replace('Z', ''))
        ts_date = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if MOJO1_START <= ts_date <= MOJO1_END:
            return "mojo1", ts.strftime('%Y-%m-%d %H:%M:%S')
        if MOJO2_START <= ts_date <= MOJO2_END:
            return "mojo2", ts.strftime('%Y-%m-%d %H:%M:%S')
        return "unknown", ts.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return "unknown", str(timestamp_str or "")

def main():
    
    with CSV_PATH.open('r') as f:
        reader = csv.DictReader(f)
        
        rows_to_report = []
        
        for i, row in enumerate(reader, start=2):
            # Check crop coords
            has_crop = (row.get('crop_x1', '') != '' and 
                       row.get('crop_y1', '') != '' and
                       row.get('crop_x2', '') != '' and
                       row.get('crop_y2', '') != '')
            
            # CRITICAL: Validate timestamp to avoid corrupted rows
            timestamp = row.get('timestamp', '')
            has_valid_timestamp = False
            if timestamp and timestamp != 'None' and timestamp.strip() != '':
                try:
                    datetime.fromisoformat(timestamp.replace('Z', ''))
                    has_valid_timestamp = True
                except Exception:
                    pass
            
            # Check dimensions
            w0 = row.get('width_0')
            h0 = row.get('height_0')
            w1 = row.get('width_1')
            h1 = row.get('height_1')
            
            w0_valid = w0 is not None and w0 not in {'', '0'}
            h0_valid = h0 is not None and h0 not in {'', '0'}
            w1_valid = w1 is not None and w1 not in {'', '0'}
            h1_valid = h1 is not None and h1 not in {'', '0'}
            
            has_dims = ((w0_valid and h0_valid) or (w1_valid and h1_valid))
            
            # ONLY process rows with valid timestamp + crop coords + missing dims
            if has_crop and has_valid_timestamp and not has_dims:
                timestamp = row.get('timestamp', '')
                project, date_str = get_project_from_timestamp(timestamp)
                filename = Path(row.get('chosen_path', '') or '').name
                
                rows_to_report.append({
                    'row': i,
                    'filename': filename,
                    'timestamp': date_str,
                    'project': project,
                    'crop_x1': row.get('crop_x1', ''),
                    'crop_y1': row.get('crop_y1', ''),
                    'crop_x2': row.get('crop_x2', ''),
                    'crop_y2': row.get('crop_y2', ''),
                    'width_0': repr(w0),
                    'height_0': repr(h0),
                    'width_1': repr(w1),
                    'height_1': repr(h1),
                    'w0_valid': w0_valid,
                    'h0_valid': h0_valid,
                    'w1_valid': w1_valid,
                    'h1_valid': h1_valid,
                    'has_dims': has_dims
                })
    
    # Write report
    
    with REPORT_PATH.open('w') as f:
        f.write("="*100 + "\n")
        f.write("BACKFILL INSPECTION REPORT\n")
        f.write("="*100 + "\n")
        f.write(f"Total rows to backfill: {len(rows_to_report)}\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        
        # Group by project
        mojo1_rows = [r for r in rows_to_report if r['project'] == 'mojo1']
        mojo2_rows = [r for r in rows_to_report if r['project'] == 'mojo2']
        unknown_rows = [r for r in rows_to_report if r['project'] == 'unknown']
        
        f.write("BREAKDOWN:\n")
        f.write(f"  Mojo1 (Oct 8-11):  {len(mojo1_rows)} rows\n")
        f.write(f"  Mojo2 (Oct 16-19): {len(mojo2_rows)} rows\n")
        f.write(f"  Unknown:           {len(unknown_rows)} rows\n")
        f.write("\n" + "="*100 + "\n\n")
        
        # Show first 50 mojo1 rows in detail
        f.write("FIRST 50 MOJO1 ROWS (showing EXACT values the script sees):\n")
        f.write("-"*100 + "\n")
        for r in mojo1_rows[:50]:
            f.write(f"\nRow {r['row']}:\n")
            f.write(f"  Filename:  {r['filename']}\n")
            f.write(f"  Timestamp: {r['timestamp']}\n")
            f.write(f"  Project:   {r['project']}\n")
            f.write(f"  Crop box:  x1={r['crop_x1']}, y1={r['crop_y1']}, x2={r['crop_x2']}, y2={r['crop_y2']}\n")
            f.write("  Dimensions (showing Python repr()):\n")
            f.write(f"    width_0  = {r['width_0']:>8} → valid={r['w0_valid']}\n")
            f.write(f"    height_0 = {r['height_0']:>8} → valid={r['h0_valid']}\n")
            f.write(f"    width_1  = {r['width_1']:>8} → valid={r['w1_valid']}\n")
            f.write(f"    height_1 = {r['height_1']:>8} → valid={r['h1_valid']}\n")
            f.write(f"  → has_dims = {r['has_dims']} (MUST be False to trigger backfill)\n")
        
        f.write("\n" + "="*100 + "\n\n")
        
        # Show first 50 mojo2 rows in detail
        f.write("FIRST 50 MOJO2 ROWS (showing EXACT values the script sees):\n")
        f.write("-"*100 + "\n")
        for r in mojo2_rows[:50]:
            f.write(f"\nRow {r['row']}:\n")
            f.write(f"  Filename:  {r['filename']}\n")
            f.write(f"  Timestamp: {r['timestamp']}\n")
            f.write(f"  Project:   {r['project']}\n")
            f.write(f"  Crop box:  x1={r['crop_x1']}, y1={r['crop_y1']}, x2={r['crop_x2']}, y2={r['crop_y2']}\n")
            f.write("  Dimensions (showing Python repr()):\n")
            f.write(f"    width_0  = {r['width_0']:>8} → valid={r['w0_valid']}\n")
            f.write(f"    height_0 = {r['height_0']:>8} → valid={r['h0_valid']}\n")
            f.write(f"    width_1  = {r['width_1']:>8} → valid={r['w1_valid']}\n")
            f.write(f"    height_1 = {r['height_1']:>8} → valid={r['h1_valid']}\n")
            f.write(f"  → has_dims = {r['has_dims']} (MUST be False to trigger backfill)\n")
        
        f.write("\n" + "="*100 + "\n\n")
        
        # Summary table of all rows
        f.write("COMPLETE LIST OF ALL 7,193 ROWS (summary format):\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Row':<6} {'Project':<8} {'Date':<12} {'Filename':<60} {'w0':<8} {'h0':<8}\n")
        f.write("-"*100 + "\n")
        
        for r in rows_to_report:
            w0_display = str(r['width_0']).strip("'\"")
            h0_display = str(r['height_0']).strip("'\"")
            ts_short = (r['timestamp'] or '')[:10]
            fname = r['filename'] or ''
            fname_short = fname[:55] + '...' if len(fname) > 58 else fname
            f.write(f"{r['row']:<6} {r['project']:<8} {ts_short:<12} {fname_short:<60} {w0_display:<8} {h0_display:<8}\n")
    

if __name__ == '__main__':
    main()

