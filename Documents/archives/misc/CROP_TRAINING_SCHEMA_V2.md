# Crop Training Data Schema Evolution
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26

**Date:** October 21, 2025

---

## Problem with Old Schema

The original `select_crop_log.csv` schema had **19 columns** with massive redundancy and noise:

### **Old Schema (19 columns - BLOATED):**
```csv
session_id, set_id, directory, image_count, chosen_index, chosen_path,
crop_x1, crop_y1, crop_x2, crop_y2, timestamp,
image_0_path, image_0_stage, width_0, height_0,
image_1_path, image_1_stage, width_1, height_1
```

### **Problems:**
- ❌ **Full paths stored** - Break when files move
- ❌ **No project tracking** - Must deduce from timestamps/directories  
- ❌ **Redundant data** - Multiple width/height columns
- ❌ **Irrelevant fields** - session_id, set_id, image_count, chosen_index
- ❌ **Directory storage** - Meaningless (files move!)
- ❌ **Complex dynamic columns** - image_0_*, image_1_*, etc.

---

## New Minimal Schema (8 columns - CLEAN)

### **Schema:**
```csv
timestamp,project_id,filename,crop_x1,crop_y1,crop_x2,crop_y2,width,height
```

### **Field Descriptions:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `timestamp` | ISO 8601 UTC | When crop decision was made | `2025-10-08T18:47:32Z` |
| `project_id` | string | Project identifier | `mojo1`, `mojo2`, `mojo3` |
| `filename` | string | Image filename (no path!) | `20250705_230713_stage3_enhanced.png` |
| `crop_x1` | float | Normalized left edge (0-1) | `0.0` |
| `crop_y1` | float | Normalized top edge (0-1) | `0.0215` |
| `crop_x2` | float | Normalized right edge (0-1) | `0.5820` |
| `crop_y2` | float | Normalized bottom edge (0-1) | `0.6030` |
| `width` | int | Original image width (pixels) | `3072` |
| `height` | int | Original image height (pixels) | `3072` |

### **Example Row:**
```csv
2025-10-08T18:47:32Z,mojo1,20250705_230713_stage3_enhanced.png,0.0,0.0215,0.5820,0.6030,3072,3072
```

---

## Why This Is Better

### **Benefits:**
1. ✅ **File-move resilient** - No paths, just filenames
2. ✅ **Project tracking built-in** - No timestamp deduction needed
3. ✅ **Minimal storage** - 8 columns vs 19 (58% reduction)
4. ✅ **Faster processing** - Less data to parse
5. ✅ **Clearer intent** - Every field has a purpose
6. ✅ **Standard format** - Similar to COCO/YOLO annotation formats

### **What We Removed (and why it's OK):**
- ❌ `session_id` - Internal detail, irrelevant to training
- ❌ `set_id` - Redundant with timestamp/filename grouping
- ❌ `directory` - Files move, paths break
- ❌ `image_count` - Irrelevant to crop prediction
- ❌ `chosen_index` - Internal detail
- ❌ `chosen_path` - Full path (replaced with filename)
- ❌ `image_0_path`, `image_1_path` - Redundant
- ❌ `image_0_stage`, `image_1_stage` - Can parse from filename if needed
- ❌ `width_1`, `height_1` - Duplicate data (same image)

---

## Validation Rules

The new schema enforces strict validation:

```python
# 1. Timestamp must be valid ISO 8601
datetime.fromisoformat(timestamp.replace('Z', ''))

# 2. Project ID must not be empty
assert project_id and project_id.strip() != ''

# 3. Filename must not be empty or contain paths
assert filename and '/' not in filename and '\\' not in filename

# 4. Crop coordinates must be normalized [0, 1] with x1 < x2, y1 < y2
assert 0 <= crop_x1 < crop_x2 <= 1
assert 0 <= crop_y1 < crop_y2 <= 1

# 5. Dimensions must be positive
assert width > 0 and height > 0
```

---

## Migration Path

### **Step 1: Keep Old Format for Historical Data**
- Keep `select_crop_log.csv` as-is (7,194 rows)
- Rename to `select_crop_log_LEGACY_pre_2025-10-21.csv`

### **Step 2: Create New Log with New Schema**
- New file: `crop_training_data.csv`
- All future logging uses new 8-column schema

### **Step 3: Migrate Old Data (Optional)**
Convert 7,194 rows from old format to new format:
```python
# Extract: timestamp, project (from timestamp), filename (from path), 
#          crop_x1-y2, width_0, height_0
```

### **Step 4: Update Training Scripts**
- Update `train_crop_proposer_v2.py` to read new format
- Falls back to old format if new file doesn't exist

---

## Research Validation

✅ **Researched standard ML annotation formats** (COCO, YOLO, Pascal VOC)

All require:
- Image identifier (✅ filename)
- Bounding box coordinates (✅ crop_x1, crop_y1, crop_x2, crop_y2)
- Original dimensions for normalization (✅ width, height)
- Optional: timestamp, category/class (✅ we have timestamp, don't need class)

**Conclusion:** Our new schema aligns with industry standards and contains everything needed for crop prediction training.

---

## File Locations

- **New log:** `data/training/crop_training_data.csv` (new schema)
- **Legacy log:** `data/training/select_crop_log_LEGACY_pre_2025-10-21.csv` (old schema, archived)
- **Current working file:** `data/training/select_crop_log.csv` (old schema, 7,194 rows)

---

## Implementation

- **Logging function:** `log_crop_decision()` in `scripts/utils/companion_file_utils.py`
- **Training reader:** Updated in `scripts/ai/train_crop_proposer_v2.py`
- **Documentation:** This file

---

**Status:** ✅ Schema designed, validated, ready for implementation  
**Next:** Implement new logging function and update training scripts

