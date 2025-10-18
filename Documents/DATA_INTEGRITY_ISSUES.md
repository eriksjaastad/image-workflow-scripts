# Data Integrity Issues - Billed vs Actual

**Date:** October 18, 2025

## Issue 1: Daily vs Project Totals Don't Match

### Observed Discrepancy:
- **Daily view (sum of all daily bars):** Mojo-1 shows MORE actual hours than billed hours
- **Project view (total bars):** Mojo-1 shows MORE billed hours than actual hours

**This is mathematically impossible** - the sum of daily actual hours should equal the project total actual hours.

### Root Cause Analysis Needed:

1. **Date Range Filtering:**
   - Daily view filters ops by `startedAt` to `finishedAt` from manifest
   - Project view uses `work_hours` from project_metrics totals
   - Are these using the same date ranges?

2. **Project Matching:**
   - Daily view matches timesheet name → project ID
   - Are we accidentally including operations from other projects?

3. **Calculation Method:**
   - Daily: Calculates work hours day-by-day using `get_file_operation_metrics` per day
   - Project: Uses pre-calculated `work_hours` from project_metrics
   - Are these using the same algorithm?

4. **Data Source:**
   - Daily: Uses `self.engine.load_file_operations()` - all raw file ops
   - Project: Uses `project_metrics` which may use daily_summaries or archived bins
   - Are these loading the same data?

### Proposed Solution:

**Immediate action:**
1. Add logging to show:
   - Total actual hours calculated in daily view (sum of all days)
   - Total actual hours from project_metrics
   - Number of operations included in each calculation
   - Date ranges used in each calculation

2. Verify project date ranges are correct in manifests

3. Consider using **the same data source and calculation** for both views

**Long-term:**
- Implement data validation script
- Add assertions to catch discrepancies
- Create daily data integrity report

---

## Issue 2: Can't Tell When Projects Start/End in Daily View

### Problem:
Daily view shows all days mixed together - no visual indication of when Mojo-1 ends and Mojo-2 begins.

### Solutions:
1. **Vertical separators** between projects
2. **Different background colors** for each project's date range
3. **Gap in the chart** between projects
4. **Project labels on X-axis** showing date ranges

---

## Next Steps:

1. **Add debug logging** to compare calculations
2. **Verify manifest dates** are correct
3. **Use consistent data source** for both views
4. **Add data validation checks**
5. **Create data integrity dashboard**

