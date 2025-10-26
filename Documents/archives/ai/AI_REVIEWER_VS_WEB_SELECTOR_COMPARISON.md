# AI Reviewer vs AI-Assisted Reviewer - Logic Comparison
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26


**Date:** 2025-10-22  
**Purpose:** Ensure AI Reviewer aligns with AI-Assisted Reviewer's proven, battle-tested logic

---

## 🎯 CORE WORKFLOW COMPARISON

### AI-Assisted Reviewer (`01_ai_assisted_reviewer.py`)

**Route:** `/submit` (line 1599)

**Logic Flow:**
```
1. Validate selections payload
2. For each selection in batch:
   a. If skipped → leave files in place, log skip
   b. If per_image_skips → leave files in place, log partial skip
   c. If selected_index is None → DELETE ALL images in group
   d. Otherwise:
      - Move selected image to selected/ OR crop/
      - Move other images to delete_staging/
      - Log training data
3. Remove processed groups from current batch
4. Return summary (kept, deleted counts)
```

**Key Features:**
- ✅ Handles `selected_index is None` → delete all
- ✅ Uses `move_file_with_all_companions()` for deselected images
- ✅ Logs file operations with `tracker.log_operation()`
- ✅ Removes processed groups from batch
- ✅ Simple, clean logic flow

---

### AI Reviewer (`01_ai_assisted_reviewer.py`)

**Route:** `/process-batch` (line 1646)

**Logic Flow:**
```
1. Validate selections payload (allows empty)
2. For each selection in batch:
   a. Get AI recommendation (for training data)
   b. Log to SQLite v3 database
   c. Create .decision sidecar for crop/
   d. Call perform_file_operations():
      - Move selected image to selected/ OR crop/
      - Move other images to delete_staging/
3. Handle unselected groups (separate loop):
   - Move ALL images to delete_staging/
4. Remove processed groups from batch
5. Return summary (kept, crop, deleted counts)
```

**Key Differences:**
- ⚠️ AI recommendation retrieval (NEW, AI-specific)
- ⚠️ SQLite v3 logging (NEW, AI-specific)
- ⚠️ .decision sidecar files (NEW, for Desktop Multi-Crop integration)
- ⚠️ Two-pass approach: selected groups first, then unselected groups
- ⚠️ Does NOT handle `selected_index is None` within selections array

---

## 🔍 KEY INSIGHT: Missing Case

### AI-Assisted Reviewer Handles:
```javascript
// Selection with selected_index: null → DELETE ALL
{groupId: "abc123", selectedIndex: null}
```

This allows a user to explicitly say "I looked at this group and want ALL images deleted."

### AI Reviewer Does NOT Handle:
- Current code assumes if a group is in `selections`, it has a valid `selected_idx`
- If user somehow sends `selected_idx: null`, the code will crash with `group.images[selected_idx]` → `group.images[None]`

---

## 🛠️ RECOMMENDED REFACTOR

### **Goal:** Align AI Reviewer with AI-Assisted Reviewer's proven logic

### **Changes Needed:**

#### 1. **Add `selected_index is None` Check**
```python
# After line 1678:
selected_idx = selection.get("selectedImage")
crop = selection.get("crop", False)

# ADD THIS CHECK (like web_image_selector line 1669-1672):
if selected_idx is None:
    # Delete all images in group
    for img in group.images:
        moved_files = move_file_with_all_companions(img, delete_staging_dir, dry_run=False)
        tracker.log_operation("delete", str(img.parent), delete_staging_dir.name, 
                             len(moved_files), files=moved_files,
                             notes=f"User explicitly deleted all images from group {group_id}")
    deleted_count += len(group.images)
    continue  # Skip to next selection
```

#### 2. **Consolidate Deletion Logic**
Currently we have:
- **Path A:** Deselected images within selected groups (in `perform_file_operations()`)
- **Path B:** All images in unselected groups (separate loop, lines 1773-1789)

**Problem:** These should use the SAME logic to avoid bugs!

**Solution:** Extract a helper function:
```python
def delete_group_images(group, delete_staging_dir, tracker, reason=""):
    """Delete all images in a group."""
    for img in group.images:
        moved_files = move_file_with_all_companions(img, delete_staging_dir, dry_run=False)
        tracker.log_operation("delete", str(img.parent), delete_staging_dir.name,
                             len(moved_files), files=moved_files,
                             notes=f"{reason} - group {group.group_id}")
    return len(group.images)
```

Then use it in both places!

#### 3. **Simplify AI Logic Separation**
**Keep AI-specific code clearly separated:**

```python
# AI-SPECIFIC: Get recommendation for training data
ai_rec = get_ai_recommendation(group, ...)
ai_selected_idx = ai_rec.get("selected_index", 0)
ai_crop_coords = ai_rec.get("crop_coords")

# AI-SPECIFIC: Log to SQLite v3
if db_path:
    log_ai_decision(db_path, ...)

# AI-SPECIFIC: Create .decision sidecar
if crop:
    create_decision_sidecar(...)

# STANDARD WEB SELECTOR LOGIC: File operations
perform_file_operations(...)  # Same as web_image_selector
```

This makes it crystal clear what's AI-specific vs. standard file handling.

#### 4. **Match Parameter Names**
AI-Assisted Reviewer uses:
- `selectedIndex` (not `selectedImage`)
- `crop` (boolean flag)

AI Reviewer should match these exactly to avoid confusion.

---

## 📋 ACTION ITEMS

### High Priority (Fix Today):
- [ ] Add `selected_idx is None` check (safety fix)
- [ ] Extract `delete_group_images()` helper function
- [ ] Consolidate deletion logic to use helper

### Medium Priority (Next Session):
- [ ] Refactor to clearly separate AI-specific vs. standard logic
- [ ] Add inline comments marking AI-specific sections
- [ ] Match parameter naming conventions

### Low Priority (Future):
- [ ] Consider merging `/process-batch` and `/submit` into shared utility
- [ ] Extract AI training data logging into separate middleware

---

## 🧪 TEST SCENARIOS

After refactor, test:
1. ✅ Select one image → others deleted, logs correct
2. ✅ Select image + crop → goes to crop/, others deleted
3. ✅ Don't select any image in group → all deleted (unselected groups path)
4. ✅ **NEW:** Explicitly pass `selectedIndex: null` → all deleted
5. ✅ AI recommendation differs from user → both logged correctly
6. ✅ Batch with mix of above scenarios

---

## 🎓 LESSONS LEARNED

1. **Don't reinvent the wheel** - AI-Assisted Reviewer works great, build on it!
2. **Clear separation of concerns** - AI logic should be additive, not intertwined
3. **Handle edge cases** - `selected_index: null` is a valid use case
4. **DRY principle** - Two deletion code paths = double the bugs
5. **Test with production patterns** - Process a real batch to shake out issues

---

## 📊 COMPLEXITY COMPARISON

| Feature | AI-Assisted Reviewer | AI Reviewer | Assessment |
|---------|-------------------|-------------|------------|
| Core file routing | ✅ Clean | ⚠️ Split across 2 functions | **Simplify** |
| Deletion handling | ✅ 1 code path | ⚠️ 2 code paths | **Consolidate** |
| Training data | ✅ Simple CSV | ⚠️ SQLite + CSV + sidecars | **Document well** |
| Edge case handling | ✅ Comprehensive | ⚠️ Missing `None` check | **Add check** |
| Code clarity | ✅ Very clear | ⚠️ AI logic mixed in | **Separate better** |

---

## ✅ CONCLUSION

**Current Status:** AI Reviewer works but has unnecessary complexity and missing edge cases.

**Goal:** Match AI-Assisted Reviewer's proven logic exactly, with AI features as clean additions.

**Benefit:** Easier to maintain, fewer bugs, clearer code.

**Next Step:** Implement High Priority action items above.

