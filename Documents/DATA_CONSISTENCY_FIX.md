# Data Consistency Fix - Table vs Chart Mismatch

**Date:** October 15, 2025  
**Issue:** Table showed 4 projects, chart showed 3 different projects

## The Problem

**Before fix:**
- **Table:** Mojo1, Mojo2, jmlimages-random, tattersail-0918
- **Chart:** Mojo1, Mojo2, **dalia** (missing jmlimages & tattersail)

## Root Cause

**Substring matching was too permissive:**
- Path matching looked for "dalia" substring in paths
- Found "dalia" in "crop/**dalia**_hannah" → false positive!
- Matched **October** "dalia_hannah" operations to **September** "dalia" project

**Why different results?**
- **Chart:** Used path matching (Strategy 1) → picked up false positives
- **Table:** Used date-based filtering → correctly excluded "dalia_hannah"

## The Fix

Updated path matching logic in `data_engine.py` lines 644-649 to use **word boundary matching**:

**Before (substring):**
```python
if pid_lower in src or pid_lower in dst:
    project_id = pid
```

**After (word boundary):**
```python
# Check for exact match or word-boundary match (separated by / or _)
# This prevents "dalia" from matching "dalia_hannah"
if (f'/{pid_lower}/' in f'/{src}/' or f'/{pid_lower}/' in f'/{dst}/' or
    src == pid_lower or dst == pid_lower):
    project_id = pid
```

## Results

**After fix:**
- ✅ Table: Mojo1, Mojo2, jmlimages-random, tattersail-0918
- ✅ Chart: Mojo1, Mojo2, jmlimages-random, tattersail-0918
- ✅ **Perfect consistency!**

## Test Cases

| Path | Project "dalia" | Project "dalia_hannah" |
|------|----------------|------------------------|
| `crop/dalia` | ✅ Match | ❌ No match |
| `crop/dalia_hannah` | ❌ No match | ✅ Match |
| `dalia` | ✅ Match | ❌ No match |
| `content/mojo1/selected` | ❌ No match | ❌ No match |

## Why This Matters

**False positives cause:**
1. Inflated metrics (dalia showing 334 files it didn't process)
2. Data inconsistency between table and chart
3. Confusion about project performance
4. Incorrect attribution of work to wrong projects

**The fix ensures:**
- Projects only show data they actually generated
- Table and chart always agree
- Accurate performance metrics

---

**Files Modified:**
- `scripts/dashboard/data_engine.py` - Added word boundary matching (lines 644-649)

**Related Issues:**
- See `PROJECT_CHART_FIX.md` for date-based matching (archived projects)
- See `TABLE_DATA_FIX.md` for PNG-only counting
- See `SEPTEMBER_DATA_FIX.md` for historical data backfill

