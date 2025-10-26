# Work Time Calculation Guide
**Status:** Active
**Audience:** Developers


**CRITICAL: Single Source of Truth for Accurate Work Hours**

Last Updated: October 18, 2025

---

## 📋 Table of Contents

1. [The Correct Method](#the-correct-method)
2. [Common Pitfalls](#common-pitfalls)
3. [Implementation Examples](#implementation-examples)
4. [Validation & Data Integrity](#validation--data-integrity)
5. [Where This Applies](#where-this-applies)

---

## 🎯 The Correct Method

### Core Principle

**Work time MUST be calculated using the hour-blocking method from `get_file_operation_metrics()`.**

This is our **SINGLE SOURCE OF TRUTH** for work time calculation.

### The Hour-Blocking Algorithm

```python
from scripts.utils.companion_file_utils import get_file_operation_metrics

# Step 1: Get file operations (list of dicts with 'timestamp' field)
file_operations = [...]  # From FileTracker logs

# Step 2: Calculate work time using the centralized function
metrics = get_file_operation_metrics(file_operations)

# Step 3: Extract work time
work_minutes = float(metrics.get('work_time_minutes') or 0.0)

# Step 4: Apply 15-minute precision (for dashboard display)
work_hours = round((work_minutes / 60.0) / 0.25) * 0.25
```

### How Hour-Blocking Works

**From `calculate_work_time_from_file_operations()`:**

1. **Collect Unique Hour Blocks:**
   - For each file operation, extract the timestamp
   - Format as "YYYY-MM-DD HH" (e.g., "2025-10-18 14")
   - Add to a set (automatically deduplicates)

2. **Count Hours:**
   - Each unique hour block = 1 hour of work
   - Total hours = count of unique hour blocks

3. **Convert to Seconds:**
   - `work_time_seconds = num_hour_blocks * 3600`

**Example:**
```
Operations at:
- 2025-10-18 14:23:15 → Hour block "2025-10-18 14"
- 2025-10-18 14:45:30 → Hour block "2025-10-18 14" (duplicate)
- 2025-10-18 15:10:22 → Hour block "2025-10-18 15"
- 2025-10-18 16:05:00 → Hour block "2025-10-18 16"

Result: 3 unique hour blocks = 3 hours = 10,800 seconds
```

### Why This Method Is Correct

✅ **No Subjective Thresholds:** No "break detection" parameters to tune  
✅ **Honest Accounting:** If you moved files in an hour, that hour counts  
✅ **Works Across Midnight:** Date is part of the hour block key  
✅ **Productivity Variation:** Shown via img/h metric, not inflated work time  
✅ **No Rounding Inflation:** Each hour counted exactly once  
✅ **Deterministic:** Same operations = same result, always  

---

## ⚠️ Common Pitfalls

### PITFALL #1: Per-Operation Rounding (The Killer)

**WRONG:**
```python
# DO NOT DO THIS!
total_hours = 0
for op in file_operations:
    hour_of_operation = extract_hour(op['timestamp'])
    total_hours += 1  # Counts every operation as 1 hour!
```

**Why This Is Wrong:**
- If you process 100 files in one hour, this counts it as 100 hours
- Completely inflates work time
- Makes data unusable for billing or comparison

**Result:** 2 hours of actual work becomes 50+ hours reported

---

### PITFALL #2: Re-calculating Instead of Reusing

**WRONG:**
```python
# Dashboard displays project totals
project_total = project_metrics[project_id]['totals']['work_hours']  # 49.0 hours

# Later, for daily breakdown...
daily_ops = get_operations_for_day(date)
daily_metrics = get_file_operation_metrics(daily_ops)
daily_hours = daily_metrics['work_time_minutes'] / 60.0  # Could be 52.3 hours!
```

**Why This Is Wrong:**
- Different calculation runs may have slightly different inputs (caching, filtering)
- Causes data discrepancies between views
- Users lose trust in the data

**Result:** Daily view shows 52.3h but project view shows 49.0h for the same project

---

### PITFALL #3: Averaging/Distribution Without Real Data

**WRONG:**
```python
# Distribute hours evenly across days
hours_per_day = total_hours / num_days
daily_breakdown = [hours_per_day] * num_days  # All days identical!
```

**Why This Is Wrong:**
- Hides actual work patterns (some days heavy, some light)
- Looks "fishy" and unbelievable
- Provides zero insight into productivity variation

**Result:** Graph shows perfectly flat bars - obviously fake data

---

## ✅ Implementation Examples

### Example 1: Project-Level Metrics (project_metrics_aggregator.py)

```python
from scripts.utils.companion_file_utils import get_file_operation_metrics

class ProjectMetricsAggregator:
    def aggregate(self):
        # ... load project operations ...
        
        # Convert operations to format expected by metrics calculator
        ops_for_metrics = []
        for op in proj_ops:
            op_copy = dict(op)
            ts = op_copy.get('timestamp')
            if isinstance(ts, datetime):
                op_copy['timestamp'] = ts.isoformat()
            elif not isinstance(ts, str):
                ts_str = op_copy.get('timestamp_str')
                if isinstance(ts_str, str):
                    op_copy['timestamp'] = ts_str
            ops_for_metrics.append(op_copy)
        
        # CORRECT: Use centralized calculation
        metrics = get_file_operation_metrics(ops_for_metrics)
        work_minutes = float(metrics.get('work_time_minutes') or 0.0)
        
        # Apply 15-minute precision for display
        work_hours = round((work_minutes / 60.0) / 0.25) * 0.25
        
        return {
            'work_hours': work_hours,
            'work_minutes': work_minutes,  # Store both for flexibility
        }
```

---

### Example 2: Daily Breakdown with Real Variation (analytics.py)

```python
def calculate_daily_breakdown(project_id, all_operations):
    """Calculate daily work hours with REAL daily variation."""
    from collections import defaultdict
    from scripts.utils.companion_file_utils import get_file_operation_metrics
    
    # Step 1: Get project's TOTAL work hours (single source of truth)
    project_metrics = get_project_metrics(project_id)
    total_work_hours = project_metrics['totals']['work_hours']
    
    # Step 2: Group operations by day
    ops_by_day = defaultdict(list)
    for op in all_operations:
        date_key = extract_date(op['timestamp'])  # "YYYY-MM-DD"
        ops_by_day[date_key].append(op)
    
    # Step 3: Calculate work hours for each day INDIVIDUALLY
    daily_hours = {}
    for date_key, day_ops in ops_by_day.items():
        # Convert to format expected by metrics calculator
        ops_for_metrics = prepare_ops_for_metrics(day_ops)
        
        # Calculate using SAME method as project totals
        metrics = get_file_operation_metrics(ops_for_metrics)
        work_minutes = float(metrics.get('work_time_minutes') or 0.0)
        work_hours = round((work_minutes / 60.0) / 0.25) * 0.25
        
        daily_hours[date_key] = work_hours
    
    # Step 4: Validate and scale if needed
    daily_sum = sum(daily_hours.values())
    
    if abs(daily_sum - total_work_hours) > 0.5:
        # Scale proportionally to match project total
        # (maintains variation while ensuring sum matches)
        scale_factor = total_work_hours / daily_sum
        daily_hours = {
            date: round(hours * scale_factor, 2)
            for date, hours in daily_hours.items()
        }
    
    return daily_hours
```

**Why This Is Correct:**
- Uses `get_file_operation_metrics()` for both totals and daily (same method)
- Shows real daily variation (not fake uniform distribution)
- Validates sum matches project total
- Scales proportionally if needed (rare, only for edge cases)

---

### Example 3: Dashboard Display with 15-Minute Precision

```python
def display_work_hours(work_minutes):
    """Display work hours with 15-minute precision."""
    # Convert minutes to hours
    hours = work_minutes / 60.0
    
    # Round to 15-minute increments (0.25 hour)
    display_hours = round(hours / 0.25) * 0.25
    
    return display_hours

# Examples:
# 49.2 min →  0.82 h → rounds to 0.75 h (45 min)
# 63.0 min →  1.05 h → rounds to 1.00 h (60 min)
# 78.5 min →  1.31 h → rounds to 1.25 h (75 min)
# 135 min  →  2.25 h → rounds to 2.25 h (135 min) - exact
```

---

## 🔍 Validation & Data Integrity

### Critical Validation Points

**1. Single Source of Truth:**
```python
# ALL work hours MUST come from project_metrics
project_work_hours = project_metrics[project_id]['totals']['work_hours']

# Daily breakdowns MUST sum to (approximately) the same total
daily_sum = sum(daily_breakdown.values())
assert abs(daily_sum - project_work_hours) < 0.5, \
    f"Data integrity error: daily sum ({daily_sum}h) != project total ({project_work_hours}h)"
```

**2. Validation Logging:**
```python
# Log validation results for debugging
print(f"  Daily sum: {daily_sum:.2f}h vs Project total: {project_work_hours:.2f}h")
if abs(daily_sum - project_work_hours) > 0.5:
    print(f"  ⚠️  WARNING: Discrepancy detected! Applying scale factor.")
```

**3. User-Facing Verification:**
- Project view shows: 49.0 hours
- Daily view bars sum to: 49.0 hours
- User can trust the data

---

### Data Integrity Checks

**Run These Checks Regularly:**

```python
def validate_work_time_integrity():
    """Validate all projects have consistent work time across views."""
    projects = load_all_projects()
    errors = []
    
    for project_id, project in projects.items():
        # Get project total
        project_total = project['totals']['work_hours']
        
        # Calculate daily sum
        daily_breakdown = get_daily_breakdown(project_id)
        daily_sum = sum(daily_breakdown.values())
        
        # Check for discrepancies
        if abs(daily_sum - project_total) > 0.5:
            errors.append({
                'project': project_id,
                'project_total': project_total,
                'daily_sum': daily_sum,
                'difference': daily_sum - project_total
            })
    
    if errors:
        print("⚠️  DATA INTEGRITY ISSUES FOUND:")
        for err in errors:
            print(f"  - {err['project']}: {err['difference']:.2f}h discrepancy")
    else:
        print("✅ All projects have consistent work time across views")
    
    return len(errors) == 0
```

---

## 📍 Where This Applies

### Files That MUST Use This Method

1. **`scripts/utils/companion_file_utils.py`**
   - Contains `get_file_operation_metrics()` - the source of truth
   - Contains `calculate_work_time_from_file_operations()` - the core algorithm

2. **`scripts/dashboard/project_metrics_aggregator.py`**
   - Calculates project-level work hours
   - Uses `get_file_operation_metrics()` for totals

3. **`scripts/dashboard/analytics.py`**
   - Builds dashboard charts
   - Uses `get_file_operation_metrics()` for daily breakdowns
   - Must use project totals as source of truth

4. **`scripts/dashboard/data_engine.py`**
   - Loads and processes file operations
   - Provides data to analytics

5. **Any new feature that displays work hours**
   - MUST call `get_file_operation_metrics()`
   - MUST NOT implement custom calculation logic

---

### Tools That Already Use This Correctly

✅ **Productivity Dashboard** - Uses centralized calculation  
✅ **Project Metrics Aggregator** - Uses centralized calculation  
✅ **Billed vs Actual Chart** - Now fixed to use single source of truth  

---

## 🚨 Critical Rules

1. **NEVER implement custom work time calculation**
   - Always use `get_file_operation_metrics()`
   - No exceptions

2. **NEVER recalculate what's already calculated**
   - Use `project_metrics['totals']['work_hours']` as source of truth
   - Don't re-process operations for the same project/period

3. **NEVER distribute hours evenly**
   - Calculate each day individually with real data
   - Show actual variation, not fake uniformity

4. **ALWAYS validate totals match**
   - Daily sum should match project total (within 0.5h tolerance)
   - Log warnings for discrepancies
   - Scale proportionally if needed (rare)

5. **ALWAYS use 15-minute precision for display**
   - Round to 0.25 hour increments
   - Store full precision internally
   - Display rounded for user

---

## 📝 Quick Reference

### The One-Liner

```python
from scripts.utils.companion_file_utils import get_file_operation_metrics

metrics = get_file_operation_metrics(file_operations)
work_hours = round((metrics['work_time_minutes'] / 60.0) / 0.25) * 0.25
```

### The Full Pattern

```python
# 1. Load operations
ops = load_file_operations(project_id, date_range)

# 2. Prepare for metrics calculator
ops_for_metrics = []
for op in ops:
    op_copy = dict(op)
    ts = op_copy.get('timestamp')
    if isinstance(ts, datetime):
        op_copy['timestamp'] = ts.isoformat()
    ops_for_metrics.append(op_copy)

# 3. Calculate with centralized function
from scripts.utils.companion_file_utils import get_file_operation_metrics
metrics = get_file_operation_metrics(ops_for_metrics)
work_minutes = float(metrics.get('work_time_minutes') or 0.0)

# 4. Apply 15-minute precision
work_hours = round((work_minutes / 60.0) / 0.25) * 0.25
```

---

## 🎯 Summary

**The Formula:**
- One unique hour block with file operations = 1 hour of work
- No rounding inflation
- No fake uniformity
- Single source of truth
- Data integrity guaranteed

**Use `get_file_operation_metrics()` for everything. No exceptions.**

---

*Last Updated: October 18, 2025*  
*Related: `Documents/TECHNICAL_KNOWLEDGE_BASE.md` (lines 44-55)*

