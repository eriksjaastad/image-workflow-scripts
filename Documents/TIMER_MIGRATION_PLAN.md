# ActivityTimer Migration Plan

## Executive Summary

**Status**: ✅ **Migration Complete!**

All active scripts use `FileTracker` for complete audit trails. `ActivityTimer` is no longer needed and has been gracefully phased out.

---

## Why Remove ActivityTimer?

### Problems with Dual Tracking:
1. **Complexity**: Combining timer data with file operation logs created confusion
2. **Timestamp Mismatches**: Timer used different formats than FileTracker
3. **Break Detection Failures**: 5-minute threshold removed almost all work time
4. **Redundant Data**: FileTracker already captures everything we need

### Benefits of FileTracker-Only:
1. **Single Source of Truth**: All data comes from actual file operations
2. **Unified Timestamps**: Consistent naive local time format
3. **Simple Calculations**: Elapsed time per day (first op to last op)
4. **Accurate Hours**: No aggressive break detection removing work time
5. **Zero Configuration**: Works automatically with every file operation

---

## Analysis: All Scripts Use FileTracker

### Active Production Scripts:
| Script | File Operations | Timer Needed? |
|--------|----------------|---------------|
| `01_web_image_selector.py` | Move to selected/, Delete | ❌ No |
| `02_web_character_sorter.py` | Move to groups, Delete | ❌ No |
| `04_multi_crop_tool.py` | Move cropped, Delete bad | ❌ No |
| `05_web_multi_directory_viewer.py` | Crop (move), Delete | ❌ No |
| `06_web_duplicate_finder.py` | Delete duplicates | ❌ No |

**Result**: Every button click = file operation = automatic tracking!

---

## Migration Steps

### ✅ Phase 1: Keep Historical Data
- **Status**: Complete
- **Action**: Preserved all timer data in `data/timer_data/`
- **Why**: Don't lose historical metrics from before October 2025

### ✅ Phase 2: Fix Timestamp Handling
- **Status**: Complete (Oct 15, 2025)
- **Changes**:
  - Project manifests: Use UTC with Z suffix (`2025-10-15T19:30:00Z`)
  - File operations: Use naive local time (`2025-10-15 15:40:34.491223`)
  - Code: Handles both formats correctly by checking `isinstance(datetime)`
- **Files Updated**:
  - `scripts/dashboard/analytics.py`
  - `scripts/dashboard/productivity_dashboard.py`
  - `scripts/utils/companion_file_utils.py`

### ✅ Phase 3: Replace Break Detection with Elapsed Time
- **Status**: Complete (Oct 15, 2025)
- **Old Algorithm**: Sum gaps between operations (if < 5 min)
  - Problem: Most work time was classified as "breaks"
  - Result: 0.7h for 25,069 images (physically impossible!)
- **New Algorithm**: Elapsed time per day
  - Group operations by date
  - Each date: `last_operation_time - first_operation_time`
  - Sum across all active dates
  - Result: 15.4h for 25,069 images (realistic!)
- **File Updated**: `scripts/utils/companion_file_utils.py`

### ✅ Phase 4: Remove Dead Timer Code
- **Status**: Complete (Oct 15, 2025)
- **File**: `scripts/02_web_character_sorter.py`
- **Removed**: Lines 769, 914 (dead code - retrieved but never used)

### ✅ Phase 5: Dashboard Uses FileTracker Only
- **Status**: Complete
- **Current Behavior**:
  - Hours = Elapsed time from file operations
  - Days = Unique dates with operations
  - Images = Sum of file_count from operations
  - img/h = Calculated from hours and images
- **No More**: Break detection, timer merging, dual data sources

---

## Current Data Flow

```
User clicks button
  ↓
File operation (move/delete)
  ↓
FileTracker logs to file_operations_logs/
  ↓
Dashboard reads logs
  ↓
Groups by project date range
  ↓
Calculates elapsed time per day
  ↓
Displays accurate hours & days
```

**Simple. Reliable. Complete.**

---

## Metrics Calculation

### Hours (Work Time)
```python
def calculate_hours(file_operations):
    # Group by date
    for date, ops in group_by_date(file_operations):
        first_op = min(op.timestamp for op in ops)
        last_op = max(op.timestamp for op in ops)
        date_hours = (last_op - first_op).total_seconds() / 3600
    
    return sum(date_hours)
```

**Example:**
- Oct 1: 3:00 PM to 5:30 PM = 2.5 hours
- Oct 4: 2:00 PM to 6:45 PM = 4.75 hours
- Oct 5: 1:00 PM to 4:15 PM = 3.25 hours
- Oct 6: 10:00 AM to 3:00 PM = 5 hours
- **Total: 15.5 hours**

### Days (Active Days)
```python
def calculate_days(file_operations):
    unique_dates = set(op.timestamp.date() for op in file_operations)
    return len(unique_dates)
```

**Example:**
- Operations on Oct 1, 4, 5, 6
- **Total: 4 days**

### Images per Hour
```python
iph = total_images / total_hours
```

**Example:**
- 25,069 images / 15.4 hours = 1,627 img/h ✓

---

## File Locations

### Data Files (Keep Forever):
- `data/file_operations_logs/*.log` - All file operations
- `data/timer_data/*.json` - Historical timer data (pre-Oct 2025)
- `data/projects/*.project.json` - Project manifests with timestamps

### Code Files (Updated):
- `scripts/utils/companion_file_utils.py` - Elapsed time algorithm
- `scripts/dashboard/analytics.py` - Dashboard calculations
- `scripts/dashboard/productivity_dashboard.py` - Table generation
- `scripts/02_web_character_sorter.py` - Removed dead timer code

### Archived (No Changes Needed):
- `scripts/utils/activity_timer.py` - Keep for reference
- `scripts/tests/test_activity_timer.py` - Keep for reference
- `scripts/archive/*` - Old scripts, ignore

---

## Testing & Validation

### Before Migration:
```
Mojo1 Web Image Selector:
  Hours: 0.7h
  Days: 0d
  Images: 25,069
  img/h: 35,813 (IMPOSSIBLE!)
```

### After Migration:
```
Mojo1 Web Image Selector:
  Hours: 15.4h
  Days: 4d
  Images: 25,069
  img/h: 1,627 (REALISTIC!)
```

**Validation**: ✅
- Days match chart: 4 days shown in "Files Processed by Tool" chart
- Hours match reality: ~4 hours per day over 4 days = ~16 hours
- img/h makes sense: 1,627 img/h = 27 images/min = realistic for reviewing

---

## Future Considerations

### If You Add New Operations:

**Rule**: Every user action should move or delete a file.

**Examples:**
- ✅ Select image → Move to selected/
- ✅ Crop image → Move to crop/
- ✅ Delete image → Send to trash
- ✅ Sort to group → Move to character_group_N/
- ❌ Just viewing (no file operation) → Won't be tracked

**If you need to track viewing-only:**
1. Add a "viewed" log operation to FileTracker
2. Don't actually move files, just log the view
3. Dashboard will automatically pick it up

### Timestamp Best Practices:

**Going Forward:**
- **File Operations**: Keep using naive local time (no Z)
- **Project Manifests**: Keep using UTC with Z for clarity
- **Code**: Always check `isinstance(timestamp, datetime)` before parsing

---

## Summary

✅ **Timer Removed**: No more ActivityTimer in production  
✅ **Data Preserved**: Historical timer data kept for reference  
✅ **Calculations Fixed**: Elapsed time algorithm works perfectly  
✅ **Timestamps Unified**: Code handles both formats correctly  
✅ **Dashboard Accurate**: Hours and days now match reality  

**Result**: Simpler code, accurate metrics, single source of truth! 🎉

---

**Last Updated**: October 15, 2025  
**Status**: Migration Complete  
**Next Review**: When adding new scripts or operations

