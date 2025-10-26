# Corrupted Rows Investigation Report
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26

**Date:** October 21, 2025  
**File:** `data/training/select_crop_log.csv`  
**Rows Affected:** 5,431 (rows 7196-12626)

---

## Summary

All 5,431 corrupted rows are from the `tattersail-0918_extraction` session and have **column shift corruption** where data was written to the wrong CSV columns.

---

## Root Cause

**Column Shift Bug:** During the `tattersail-0918_extraction` session, data was written with columns misaligned:
- Real timestamp ended up in `crop_x2` column
- Fake values (`0.0`, `1.0`) in earlier columns  
- All columns after `crop_y2` are empty

---

## Evidence

### **First Corrupted Row (7196):**
```
session_id:    tattersail-0918_extraction
directory:     PROJECT_ROOT/training data/tattersail-0918_final/_white young skinny brunette arched eyebrows/20250705_173158_stage2_upscaled.png
image_count:   2.0
chosen_index:  0.0
chosen_path:   0.0              ← WRONG (should be file path)
crop_x1:       1.0              ← WRONG (fake crop coord)
crop_y1:       1.0              ← WRONG (fake crop coord)
crop_x2:       2025-09-23T00:20:47+00:00  ← THIS IS THE TIMESTAMP! (shifted)
crop_y2:       (empty)
timestamp:     (empty)          ← SHOULD BE HERE
image_0_path:  (empty)
...all remaining columns empty...
```

### **Pattern Identified:**
- ✅ `session_id` = correct
- ✅ `set_id` = correct (tattersail-0918)
- ✅ `directory` = correct (full path to final image)
- ✅ `image_count` = correct (1.5, 2.0, etc.)
- ❌ `chosen_index` = `'0.0'` (suspicious but might be valid)
- ❌ `chosen_path` = `'0.0'` (DEFINITELY WRONG)
- ❌ `crop_x1` = `'1.0'` (fake)
- ❌ `crop_y1` = `'1.0'` (fake)
- ❌ `crop_x2` = **TIMESTAMP** (shifted from correct column!)
- ❌ `crop_y2` = empty (should have value)
- ❌ `timestamp` = empty (data is in `crop_x2`)
- ❌ All remaining columns = empty

---

## Likely Cause

The bug was probably in the **historical data extraction script** (`import_historical_projects.py` or similar) that logged these rows. Possible causes:

1. **Missing columns during write:** Script skipped writing some columns
2. **Wrong field order:** Script wrote fields in wrong order
3. **CSV writer misconfiguration:** Wrong delimiter or quote handling
4. **Data source issue:** Source data (tattersail-0918_final) already had shifted columns

---

## Recovery Assessment

### **Can We Recover This Data?**

**Partially YES** - We can extract some useful information:

1. ✅ **Timestamp:** In `crop_x2` column (can be extracted)
2. ✅ **Image path:** In `directory` column (full path to final image)
3. ✅ **Project:** `tattersail-0918`
4. ❌ **Crop coordinates:** FAKE (`1.0, 1.0, timestamp, empty`)
5. ❌ **Selection data:** Chosen path is `'0.0'` (useless)

**Conclusion:** These rows have **NO VALID CROP COORDINATES**, so they cannot be used for crop training. They were selection-only logs that got corrupted during write.

---

## Impact

### **Training Data:**
- ❌ **Cannot use** for crop training (no valid crop coords)
- ✅ **Already filtered out** by our backfill script (timestamp validation caught them!)
- ✅ **No impact** on the successful 7,193 row backfill

### **Current State:**
- ✅ Backfill script correctly skipped all 5,431 corrupted rows
- ✅ Only processed 7,193 rows with valid timestamps + crop coords
- ✅ Data integrity maintained

---

## Recommendations

### **Immediate:**
1. ✅ **DONE:** Corrupted rows automatically filtered out by timestamp validation
2. ✅ **DONE:** 7,193 valid rows successfully backfilled
3. ⚠️  **Keep corrupted rows** in CSV for now (historical record)

### **Future Prevention:**
1. **Add validation** to historical extraction scripts:
   - Verify crop coordinates are valid floats (not '1.0, 1.0')
   - Verify chosen_path is a real path (not '0.0')
   - Verify timestamp is in correct column
   
2. **Add column count check:**
   ```python
   expected_cols = 19
   if len(row) != expected_cols:
       raise ValueError(f"Row has {len(row)} columns, expected {expected_cols}")
   ```

3. **Add data integrity check before writing:**
   ```python
   def validate_crop_log_row(row):
       # Check timestamp is ISO format
       datetime.fromisoformat(row['timestamp'].replace('Z', ''))
       
       # Check crop coords are floats, not '1.0'
       assert float(row['crop_x1']) != 1.0 or float(row['crop_y1']) != 1.0
       
       # Check chosen_path is not '0.0'
       assert row['chosen_path'] != '0.0'
       
       return True
   ```

### **Cleanup (Optional):**
We could remove these 5,431 rows from the CSV since they provide no value, but keeping them as a historical record of the bug is also valid.

---

## Session Affected

**Only `tattersail-0918_extraction`:** All 5,431 corrupted rows are from this single extraction session on `2025-09-22` and `2025-09-23`.

No other extraction sessions were affected.

---

## Files

- **Corrupted CSV:** `data/training/select_crop_log.csv` (rows 7196-12626)
- **Backup (pre-backfill):** `data/training/select_crop_log_backup_20251021_163041.csv`
- **This Report:** `Documents/CORRUPTED_ROWS_INVESTIGATION_2025-10-21.md`

---

**Status:** ✅ Investigated, documented, and automatically handled by validation.

