# AI Data Collection: Lessons Learned

**Created:** 2025-10-21  
**Purpose:** Document critical lessons from Phase 2 data collection issues to prevent future problems

---

## 🚨 Executive Summary

During Phase 2 (AI Model Training), we discovered multiple data integrity issues that prevented us from using ~90% of our historical crop training data. This document captures what went wrong, how we detected it, and how to prevent it in the future.

**Impact:** 
- Expected: 7,189 crop training examples from mojo1/mojo2
- Actually usable: ~490 examples
- Lost: ~6,700 examples due to bugs and missing metadata

---

## 💔 What Went Wrong

### Issue #1: Desktop Multi-Crop Tool Logging Bug
**Problem:** The tool logged crop coordinates but set image dimensions to `(0, 0)` instead of the actual dimensions.

**Root Cause:**
```python
# BUG - was using wrong key
image_sizes = [image_info.get('size', (0, 0))]  # ❌ WRONG

# FIXED - now uses correct key
image_sizes = [image_info.get('original_size', (0, 0))]  # ✅ CORRECT
```

**Impact:** 
- 7,189 crop entries logged with pixel coordinates (e.g., `x2=1787`) but no dimensions
- Cannot normalize pixel coordinates without knowing image dimensions
- Made 93% of mojo1/mojo2 crop data unusable

**When Detected:** 3 weeks after data collection, during model training

**Fixed:** October 21, 2025 (commit in `scripts/04_desktop_multi_crop.py`)

---

### Issue #2: Embeddings Were Deleted/Never Generated
**Problem:** mojo2 had 3,206 crop entries in CSV but only 5 actual embedding files existed.

**Root Cause:** Embeddings were computed at one point but:
- Either deleted during cleanup
- Or generation was interrupted/incomplete
- Cache file said embeddings existed, but `.npy` files were missing from disk

**Impact:**
- mojo2: Only 5 usable crops instead of 3,211
- Lost 3,206 training examples

**When Detected:** 3 weeks after data collection, during model training

**Solution:** Re-computed all mojo2 embeddings (17,834 new embeddings, took ~30 min)

---

### Issue #3: Historical Extraction Placeholder Data
**Problem:** Historical project extraction wrote placeholder crop coordinates `(0.0, 1.0, 1.0)` for all images.

**Root Cause:** The extraction script had no way to know actual crop coordinates from completed projects. It could only detect IF an image was cropped (by file modification time), not WHERE it was cropped.

**Impact:**
- 3,462 entries in CSV with invalid/placeholder crop data
- Filled logs with noise, making real data harder to identify

**When Detected:** During troubleshooting, after seeing "Invalid crop coords: 3,462"

**Lesson:** Only log crop data when actual cropping happens. Don't generate placeholder data.

---

### Issue #4: Silent Failures During Data Loading
**Problem:** Data loading failed silently with generic counts:
- "No embedding: 9,383"
- "Invalid crop coords: 3,462"

No details about WHICH files, WHICH projects, or WHY they failed.

**Root Cause:** Error handling swallowed exceptions and just incremented counters.

**Impact:** Took hours of debugging to understand what was actually failing.

**Solution:** See "Validation Tools" section below.

---

### Issue #5: Path Format Mismatches
**Problem:** Multiple path format issues caused embedding lookups to fail:
- CSV: `/Users/eriksjaastad/.../crop/file.png` (absolute)
- Embeddings cache: `mojo1/file.png` (relative)
- Embeddings directory: `data/ai_data/cache/embeddings/` (not `data/ai_data/embeddings/`)

**Root Cause:** Different scripts used different path conventions.

**Impact:** Initially appeared as "No embedding" errors until path normalization was added.

**Lesson:** Standardize on one path format (relative from project root) across all scripts.

---

## ✅ What We Did Right

### 1. **File Tracker Logs Everything**
The `FileTracker` logs all file operations to `data/file_operations_logs/`. This audit trail was invaluable for debugging.

### 2. **Session-Based Logging**
Using `session_id` in CSVs made it easy to:
- Group entries by workflow run
- Identify which tool created which data
- Track down problems to specific time periods

### 3. **Multi-Layer Data Collection**
- Selection decisions → `selection_only_log.csv`
- Crop decisions → `select_crop_log.csv`  
- Embeddings → `processed_images.jsonl`

Having multiple data sources allowed cross-validation.

### 4. **Filename-Based Fallback**
When direct path matching failed, falling back to filename matching (ignoring directory structure) saved ~4,000 mojo1 training examples.

---

## 🛠️ Prevention Strategies

### Strategy #1: Validate Data at Collection Time
**Don't wait until training to discover data problems.**

Add validation to `log_select_crop_entry()`:
```python
def log_select_crop_entry(...):
    # BEFORE logging, validate:
    
    # Check 1: Are dimensions valid?
    for w, h in image_sizes:
        if w <= 0 or h <= 0:
            raise ValueError(f"❌ CRITICAL: Invalid dimensions ({w}, {h}) for {image_paths[chosen_idx]}")
    
    # Check 2: Are crop coords in valid range?
    x1, y1, x2, y2 = crop_coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"❌ CRITICAL: Invalid crop box ({x1}, {y1}, {x2}, {y2})")
    
    # Check 3: Does embedding exist?
    from pathlib import Path
    filename = Path(image_paths[chosen_idx]).name
    # ... check embeddings cache ...
    
    # Only log if ALL checks pass
    writer.writerow(...)
```

**Result:** Get IMMEDIATE feedback when something is broken, not weeks later.

---

### Strategy #2: Daily Data Validation Script
**Create:** `scripts/ai/validate_training_data.py`

**Run nightly** (via cron or manual) to check:
1. Are embeddings missing for recent selections?
2. Do crop entries have valid dimensions?
3. Are embedding files actually on disk (not just in cache)?
4. Are there orphaned entries (CSV without embeddings, or embeddings without CSV)?

**Output:**
```
=== TRAINING DATA VALIDATION REPORT ===
Date: 2025-10-21

✅ Selection data: 21,250 entries
✅ Crop data: 7,189 entries

⚠️  WARNINGS:
   - 24 selections missing embeddings (mojo1)
   - 12 crop entries with dimensions = (0, 0)

❌ ERRORS:
   - 3,206 crop entries point to missing embedding files!
   
🔧 RECOMMENDATIONS:
   1. Run: python scripts/ai/compute_embeddings.py
   2. Check: scripts/04_desktop_multi_crop.py for dimension logging bug
```

---

### Strategy #3: Embedding Generation Health Checks
**Problem:** We generated embeddings but didn't verify the files were written.

**Solution:** After computing embeddings, verify:
```python
# In compute_embeddings.py
def verify_embeddings():
    """Verify all cached embeddings actually exist on disk."""
    cache_file = Path('data/ai_data/cache/processed_images.jsonl')
    embeddings_dir = Path('data/ai_data/cache/embeddings')
    
    missing = []
    with open(cache_file) as f:
        for line in f:
            entry = json.loads(line)
            hash_val = entry['hash']
            emb_file = embeddings_dir / f"{hash_val}.npy"
            if not emb_file.exists():
                missing.append(entry['image_path'])
    
    if missing:
        print(f"❌ ERROR: {len(missing)} embeddings in cache but missing files!")
        print(f"   First 10: {missing[:10]}")
        return False
    
    print(f"✅ All {count} embeddings verified on disk")
    return True

# Call after generation
compute_all_embeddings()
if not verify_embeddings():
    sys.exit(1)
```

---

### Strategy #4: Data Collection Smoke Test
**Create:** `scripts/ai/test_data_collection.py`

**Run after ANY changes to data collection code:**
```python
def test_crop_logging():
    """Test that crop logging works end-to-end."""
    
    # 1. Create test image
    test_img = create_test_image(1920, 1080)
    
    # 2. Log a fake crop
    log_select_crop_entry(
        session_id="test",
        set_id="test_set",
        directory="test_dir",
        image_paths=[test_img],
        image_stages=["stage1"],
        image_sizes=[(1920, 1080)],  # Must be provided!
        chosen_idx=0,
        crop_coords=(100, 100, 800, 800)
    )
    
    # 3. Read back from CSV
    df = pd.read_csv('data/training/select_crop_log.csv')
    latest = df.iloc[-1]
    
    # 4. Validate
    assert latest['width_0'] == 1920, "❌ Width not logged!"
    assert latest['height_0'] == 1080, "❌ Height not logged!"
    assert latest['crop_x1'] == 100, "❌ Crop x1 wrong!"
    
    print("✅ Crop logging test PASSED")
```

**Run:** `python scripts/ai/test_data_collection.py` before committing changes.

---

### Strategy #5: Require Embeddings Before Cropping
**Enforce workflow order:**

In Desktop Multi-Crop tool, check if embedding exists BEFORE allowing crop:
```python
def validate_image_before_crop(image_path):
    """Ensure image has embedding before allowing crop."""
    filename = Path(image_path).name
    
    # Check embeddings cache
    if filename not in embeddings_cache:
        raise RuntimeError(
            f"❌ CANNOT CROP: No embedding exists for {filename}\n"
            f"   Run: python scripts/ai/compute_embeddings.py first!"
        )
```

**Result:** Can't create "orphaned" crop data without embeddings.

---

## 📊 Key Metrics to Track

### Daily Dashboard
Track these metrics in a simple dashboard or log file:

```
=== AI Training Data Health ===
Updated: 2025-10-21 10:30 AM

Selections:
  Total: 21,250
  With embeddings: 21,226 (99.9%) ✅
  
Crops:
  Total: 7,189
  With embeddings: 7,189 (100%) ✅
  With valid dimensions: 7,189 (100%) ✅
  
Embeddings:
  Cache entries: 77,304
  Files on disk: 77,304 (100%) ✅
  
Last 24 hours:
  New selections: 125
  New crops: 47
  New embeddings: 172
  
⚠️ Recent warnings: None
❌ Recent errors: None
```

### Red Flags
Alert immediately if:
- Crop entries logged with dimensions = (0, 0)
- Selection/crop logged without embedding
- Embedding cache entry created but file doesn't exist
- Crop coordinates outside [0, 1] range after normalization

---

## 🔍 Debugging Checklist

When training fails with "No data loaded" or low counts:

### Step 1: Check CSV Files Exist and Have Data
```bash
wc -l data/training/selection_only_log.csv
wc -l data/training/select_crop_log.csv
head -5 data/training/select_crop_log.csv
```

### Step 2: Check Embeddings Cache vs Disk
```python
# Count cache entries
cache_count = len(open('data/ai_data/cache/processed_images.jsonl').readlines())

# Count actual files
from pathlib import Path
files_count = len(list(Path('data/ai_data/cache/embeddings').glob('*.npy')))

print(f"Cache: {cache_count}, Files: {files_count}")
# Should match!
```

### Step 3: Sample 10 Random CSV Entries
```python
import csv, random
from pathlib import Path

with open('data/training/select_crop_log.csv') as f:
    rows = list(csv.DictReader(f))
    sample = random.sample(rows, 10)
    
    for row in sample:
        filename = Path(row['chosen_path']).name
        width = row.get('width_0', 0)
        height = row.get('height_0', 0)
        
        print(f"{filename}: {width}x{height}")
        # Should NOT be 0x0!
```

### Step 4: Check Path Formats
```python
# CSV paths
with open('data/training/select_crop_log.csv') as f:
    reader = csv.DictReader(f)
    row = next(reader)
    print(f"CSV path: {row['chosen_path']}")

# Embeddings cache paths
with open('data/ai_data/cache/processed_images.jsonl') as f:
    entry = json.loads(f.readline())
    print(f"Cache path: {entry['image_path']}")

# Should use consistent format!
```

---

## 🚀 Implementation Priority

### Priority 1 (Do Immediately)
1. ✅ Fix Desktop Multi-Crop dimension logging bug
2. ✅ Re-compute missing mojo2 embeddings
3. Create `scripts/ai/validate_training_data.py`

### Priority 2 (Before Next Training Run)
4. Add validation to `log_select_crop_entry()`
5. Create data collection smoke tests
6. Add embedding verification after generation

### Priority 3 (Before Production Use)
7. Create daily validation dashboard
8. Add "require embeddings" check to crop tool
9. Set up automated alerts

---

## 📝 Final Thoughts

**The root problem:** Data collection and model training were separated by weeks. By the time we tried to use the data, we had no way to fix problems retroactively.

**The solution:** Validate EARLY and validate OFTEN.
- Validate at collection time (immediate feedback)
- Validate daily (catch issues within 24 hours)
- Validate before training (last line of defense)

**Cost-benefit:**
- Investment: ~4 hours to build validation tools
- Savings: Avoided weeks of wasted work and lost training data
- ROI: Priceless 💰

---

## Related Documents
- `AI_PROJECT_IMPLEMENTATION_PLAN.md` - Phase 2 overview
- `AI_TRAINING_PHASE2_QUICKSTART.md` - Technical training guide
- `FILE_SAFETY_SYSTEM.md` - File operation safety rules

