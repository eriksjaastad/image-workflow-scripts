# Project Productivity Table - Data Fixes

**Date:** October 15, 2025  
**Issues:** Missing data for archived projects, inflated image counts

## Problems Found

### 1. Missing Script Name Mappings ✅ FIXED

Historical projects used different script names that weren't being mapped to display names:
- `image_version_selector` → "Web Image Selector"
- `character_sorter` → "Web Character Sorter"
- `batch_crop_tool` → "Multi Crop Tool"
- `hybrid_grouper` → "Face Grouper"
- `test_web_selector` → "Web Image Selector"

**Fix:** Updated `get_display_name()` in `data_engine.py` to include all legacy script names.

### 2. Inflated Image Counts (Companion Files) ✅ FIXED

The table was counting **all files** (PNG + YAML + captions) instead of just PNGs.

**Example:** Mojo 2 showing 21,818 images when it should show ~14,480 PNGs.

**Root cause:** Lines 603, 606-607 in `analytics.py` used `file_count` (all files) instead of counting PNGs.

**Fix:** Added `count_pngs()` helper function that:
1. Counts PNG files only from `files` list
2. Falls back to `file_count` for daily summaries (which are already PNG-only after backfill)

### 3. Workflow Change: Select vs Crop

The user noticed "Selected" and "Cropped" columns showing unexpected values.

**What changed:**
- **Old workflow (Sept):** `image_version_selector` → "selected" dir → "crop" dir (two-stage)
- **New workflow (Oct+):** `01_web_image_selector` → directly to "crop" or "selected" (one-stage)

**Current behavior:**
- Mojo 2: Selected = 1,212, Cropped = 2,697 ✅ Correct
- Archived projects (Sept): No dest_dir data (daily summaries don't preserve paths)

For **archived projects**, the selected/cropped breakdown will be **missing** because:
1. Only daily summaries exist (raw logs are archived/compressed)
2. Daily summaries don't preserve individual `dest_dir` paths
3. To get this data, we'd need to reprocess raw archived logs

## Verified Data

**Tattersail-0918 (Sept 19-22):**
- Web Image Selector: 10,401 PNGs ✅
- Web Character Sorter: 11,229 PNGs ✅
- Multi Crop Tool: 2,682 PNGs ✅
- Face Grouper: 31,072 PNGs ✅

**Mojo 2 (Oct 12+):**
- Web Image Selector:
  - Selected: 1,212 PNGs
  - Cropped: 2,697 PNGs
  - Total: 14,480 operations (including deletes)
- Initial images: 17,935 ✅

## Next Steps

**Restart your dashboard** to see the fixes:

```bash
pkill -f productivity_dashboard.py
python scripts/dashboard/productivity_dashboard.py
```

Expected changes:
- ✅ Tattersail, JM Images Random will now show tool breakdowns
- ✅ Image counts will be ~50% lower (PNG-only, not all companions)
- ✅ Mojo 2 will show ~14,480 instead of 21,818
- ⚠️  Archived projects won't have selected/cropped breakdown (data not available in summaries)

## Files Modified

- `scripts/dashboard/data_engine.py` - Added legacy script name mappings, PNG-only counting in project aggregation
- `scripts/dashboard/analytics.py` - Added PNG-only counting in table breakdown
- `scripts/cleanup_logs.py` - Added PNG-only counting in daily summaries (from earlier fix)

---

**Note:** If you need selected/cropped breakdown for archived projects, we'd need to:
1. Keep raw FileTracker logs (don't compress/archive them)
2. Or reprocess archived logs to extract dest_dir information
3. Or accept that historical data doesn't have this granularity

