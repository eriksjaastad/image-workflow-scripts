# AI-Assisted Reviewer - Batch Processing Implementation
**Audience:** Developers

**Last Updated:** 2025-10-26

**Date:** October 21, 2025  
**Status:** ✅ COMPLETE

---

## 🎯 **What Changed**

Converted AI-Assisted Reviewer from **single-group navigation** to **web_image_selector.py batch processing model**.

---

## 📋 **Changes Made**

### **1. Configuration at Top**
```python
# Line 152-154
DEFAULT_BATCH_SIZE = 100  # Number of groups to process per batch (match web selector)
THUMBNAIL_MAX_DIM = 768   # Thumbnail size for web display
STAGE_NAMES = ("stage1_generated", "stage1.5_face_swapped", "stage2_upscaled")
```

### **2. Batch Info Structure**
```python
# Line 680-688 in build_app()
batch_info = {
    "current_batch": 1,  # 1-indexed
    "total_batches": (total_groups + batch_size - 1) // batch_size,
    "batch_start": batch_start,
    "batch_end": batch_end,
    "batch_size": batch_size,
    "total_groups": total_groups,
    "current_batch_size": len(current_batch_groups)
}
```

### **3. Template: Show ALL Groups**
**Before:** Single `<div class="group-card">` with one group  
**After:** Loop through ALL groups in batch:
```html
{% for group in groups %}
<section class="group" data-group-id="{{ group.group_id }}">
  <!-- images for this group -->
</section>
{% endfor %}
```

### **4. JavaScript: Queue Decisions (Don't Execute)**
**Before:** `submitImageDecision()` → fetch `/submit` → file ops → navigate  
**After:** Decision queuing in memory:
```javascript
let groupStates = {}; // { groupId: { selectedImage: idx, crop: bool } }

function selectImage(imageIndex, groupId) {
  groupStates[groupId] = { selectedImage: imageIndex, crop: false };
  updateVisualState();  // Show selection visually
  updateSummary();      // Update counters
  // NO fetch, NO navigation!
}
```

### **5. Hotkeys Match Web Selector**
```javascript
// Lines 1130-1180
'1', '2', '3', '4'  → selectImage() - approve without crop
'q', 'w', 'e', 'r'  → selectImageWithCrop() - approve WITH crop
'Enter' / 'Space'   → Scroll to next group
'ArrowUp'           → Scroll to previous group
```

### **6. Visual Feedback**
```css
.image-card.selected {
  border: 3px solid var(--success);  /* Green */
}
.image-card.crop-selected {
  border-color: var(--warning);  /* Yellow */
}
.image-card.delete-hint {
  opacity: 0.5;
  border-color: var(--danger);  /* Red/faded */
}
```

### **7. "Finalize selections" Button**
- Fixed at bottom of page
- Disabled until scroll to 90% (safety feature)
- On click:
  - Converts `groupStates` to selections array
  - POST to `/process-batch`
  - Executes all file operations
  - Reloads to show next batch

### **8. `/process-batch` Endpoint**
**Clone of web_image_selector.py logic (lines 1283-1358):**
- Receives selections array
- Loops through each selection
- Calls `perform_file_operations()` for each
- Returns counts (kept, crop, deleted)

### **9. Routes Updated**
- **`/` (index):** Now shows ALL groups in batch, not single group
- **`/submit`:** REMOVED (no longer needed for batch processing)
- **`/next`, `/prev`:** REMOVED (navigation now via scroll)
- **`/process-batch`:** NEW - processes queued selections

---

## 🔧 **How It Works**

### **User Workflow:**
1. **Page loads** → Shows 100 groups (configurable via `--batch-size`)
2. **User scrolls** → Reviews each group
3. **Presses hotkeys** → `1-4` to select, `Q-E` to select+crop
4. **Decisions queued** → Stored in `groupStates{}` JavaScript object
5. **Visual feedback** → Selected images highlighted green/yellow, others faded red
6. **Scroll to bottom** → "Finalize selections" button enables
7. **Click finalize** → All file operations execute at once
8. **Reload** → Shows next batch of 100 groups

### **Default Behavior:**
- **No selection:** All images in group deleted
- **Select image (1-4):** Selected image → `selected/`, others → `delete_staging/`
- **Select with crop (Q-E):** Selected image → `crop/`, others → `delete_staging/`

---

## 🚀 **Benefits**

1. **Speed:** Batch processing is MUCH faster than one-at-a-time
2. **Review:** Can review decisions before executing (change mind by clicking again)
3. **Familiar:** Exact same workflow as web_image_selector.py
4. **Safe:** Scroll-to-enable button prevents accidental batch processing
5. **Configurable:** `--batch-size` flag to adjust batch size (default 100)

---

## 📊 **Performance**

**Batch size recommendations:**
- **Testing:** `--batch-size 10` (quick feedback)
- **Production:** `--batch-size 100` (default, matches web selector)
- **Large projects:** `--batch-size 200` (even faster)

**Expected speed:**
- AI-Assisted Reviewer: 600 images/hour (10/min)
- AI-Assisted Reviewer: **Similar or faster** (no dragging crop boxes!)

---

## 🧪 **Testing**

To test with a small batch:
```bash
source .venv311/bin/activate
python scripts/01_ai_assisted_reviewer.py mojo3/faces/ --batch-size 10
```

Then:
1. Press `1` or `2` or `3` to select images
2. Press `Q` or `W` or `E` to select+crop
3. Press `Enter` to scroll to next group
4. Scroll to bottom
5. Click "Finalize selections"
6. Check that files moved correctly

---

## 📝 **Command-Line Usage**

```bash
# Default (100 groups per batch)
python scripts/01_ai_assisted_reviewer.py mojo3/faces/

# Custom batch size
python scripts/01_ai_assisted_reviewer.py mojo3/faces/ --batch-size 20

# Custom port
python scripts/01_ai_assisted_reviewer.py mojo3/faces/ --port 8082
```

---

## ✅ **Checklist: What Was Completed**

- [x] Added `DEFAULT_BATCH_SIZE` config at top of file
- [x] Matched hotkeys to web_image_selector.py (1-4, Q-E, Enter, ArrowUp)
- [x] Changed template to show ALL groups in batch
- [x] Added `groupStates{}` to queue decisions in memory
- [x] Added visual feedback (selected, crop-selected, delete-hint)
- [x] Added "Finalize selections" button with scroll-to-enable
- [x] Added `/process-batch` endpoint (clone of web selector logic)
- [x] Updated `/` route to render all groups
- [x] Removed single-group navigation (`/next`, `/prev`)
- [x] Removed immediate file operations from `/submit`
- [x] Added batch_info to app.config
- [x] Updated CSS for batch processing UI
- [x] Tested for linter errors (none found)

---

## 🎉 **Result**

**AI-Assisted Reviewer is now a direct clone of AI-Assisted Reviewer's batch processing workflow, with AI recommendations overlaid on top!**

User gets:
- Same speed as web selector
- Same hotkeys
- Same batch workflow
- **PLUS:** AI recommendations for selection and cropping

