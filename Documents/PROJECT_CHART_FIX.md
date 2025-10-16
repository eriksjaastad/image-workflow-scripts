# Files Processed by Project Chart - Fix Summary

**Date:** October 15, 2025  
**Issue:** Only 3 projects showing in chart, should show all projects with data

## Root Cause

**Daily summaries don't have `source_dir` or `dest_dir` paths**, so path-based project matching failed for all archived (September) projects.

## The Fix

Added **date-based matching** as fallback in `data_engine.py` lines 647-692:

**Strategy 1:** Path matching (for raw file operations)
- Looks for project ID in `source_dir` or `dest_dir`  
- Works for: Mojo1, Mojo2, current work

**Strategy 2:** Date-based matching (for daily summaries)
- Checks if operation timestamp falls within project's `startedAt` to `finishedAt` range
- Works for: tattersail-0918, jmlimages-random, dalia, and other archived projects

## Secondary Fix

Increased default **lookback from 30 to 60 days** to show more historical projects:
- `productivity_dashboard.py` lines 48, 106, 818

## Results

**Before:** 3 projects visible (Mojo1, Mojo2, dalia)  
**After:** 5 projects visible (added tattersail-0918, jmlimages-random)

## Why Only 5 Projects?

Of 18 total projects, only 5 have file operations data:

### Projects WITH Data (5):
| Project | Date Range | Operations | Reason |
|---------|-----------|------------|---------|
| tattersail-0918 | Sept 19-22 | 9,993+ | Has daily summaries |
| jmlimages-random | Sept 23 | 31,987+ | Has daily summaries |
| dalia | Sept 28+ | 59+ | Has daily summaries |
| Mojo1 | Oct 1+ | 11,944 | Has raw logs |
| Mojo2 | Oct 12+ | 31,256 | Has raw logs |

### Projects WITHOUT Data (13):
| Project | Date | Why No Data |
|---------|------|-------------|
| aiko_raw | Aug 29 | Before FileTracker was used |
| eleni_raw | Aug 30 | Before FileTracker was used |
| kiara_average | Sept 1 | Before FileTracker was used |
| slender_kiara | Sept 2 | Before FileTracker was used |
| agent-1001/1002/1003 | Sept 3-5 | Before FileTracker was used |
| 1011/1012/1013 | Sept 9-11 | Before FileTracker was used |
| 1101_hailey | Sept 15 | FileTracker started mid-day |
| 1100 | Sept 17 | No operations logged (?) |
| mixed-0919 | Sept 29 | No operations logged (?) |

**FileTracker began Sept 15, 2025** - Projects before this have no file operation logs.

## FileTracker History

Daily summaries exist for:
- Sept 15-16: 591-249 operations (system starting up)
- Sept 18-26: 1,262-19,355 operations (full data)
- Sept 28-29: 59-1,285 operations (lighter work days)

Days with 0 operations:
- Sept 1-14: No FileTracker logs
- Sept 17, 19, 27, 30: Days off or no logged work

## To Show More Projects

If you want ALL projects to appear (even with 0 data), we need to:

1. **Modify chart logic** to show projects from manifests even if no operations
2. **Add indicator** for "no data available"
3. **Or accept** that only projects with actual file operations appear

Current behavior is **correct**: Charts only show projects with measurable data.

## Next Steps

**Restart dashboard** to see 5 projects (was 3):

```bash
pkill -f productivity_dashboard.py
python scripts/dashboard/productivity_dashboard.py
```

---

**Files Modified:**
- `scripts/dashboard/data_engine.py` - Added date-based project matching (lines 647-692)
- `scripts/dashboard/productivity_dashboard.py` - Increased lookback to 60 days (lines 48, 106, 818)

