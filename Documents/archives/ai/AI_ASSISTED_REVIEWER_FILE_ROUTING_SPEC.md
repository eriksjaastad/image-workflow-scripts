# AI-Assisted Reviewer - File Routing Specification
## Created: October 21, 2025

## 🎯 **Purpose**

AI-Assisted Reviewer should **REPLACE** both:
- 01 Web Image Selector
- 04 Desktop Multi-Crop

By combining selection + crop proposal + file routing into ONE integrated tool.

---

## 📁 **File Routing Logic**

### **User Actions → File Destinations:**

| User Action | Crop Proposal | File Routing | Notes |
|------------|---------------|--------------|-------|
| **Approve** | No crop needed | Selected image → `selected/`<br>Others → `delete_staging/` | Most common path |
| **Approve** | Crop proposed | Perform crop → `selected/`<br>Others → `delete_staging/` | Integrated cropping |
| **Override** | Any | Selected image → `selected/`<br>Others → `delete_staging/` | User picks different image |
| **Manual Crop** | Any | Selected image → `crop/`<br>Others → `delete_staging/` | User wants to crop manually later |
| **Reject/Delete** | Any | ALL images → `delete_staging/` | User doesn't want any of them |
| **Skip** | Any | NO file moves | Review later |

---

## 🗂️ **Directory Structure**

```
project/
├── raw_images/           # Input: Unsorted images from multiple projects
├── selected/             # Output: Final selected images (cropped or uncropped)
├── crop/                 # Staging: Images that need manual cropping
└── delete_staging/       # Staging: Fast deletion (move to Trash later)
```

---

## 🔄 **Workflow Comparison**

### **OLD Workflow (2 tools):**
1. Web Image Selector → Move winners to `selected/`, losers to `delete_staging/`
2. Desktop Multi-Crop → Crop images in `selected/`, move to Trash if deleted

### **NEW Workflow (1 tool):**
1. AI-Assisted Reviewer → Select + Crop + Route in ONE PASS

---

## 💾 **Data Integrity: Training Data Logging**

ALL file operations that involve image selection/cropping must log to training data:

### **Selection Logging:**
- **Function:** `log_selection_only_entry()` from `companion_file_utils.py`
- **When:** User approves/overrides AI pick (no crop performed)
- **Log to:** `data/training/selection_only_log.csv`

### **Crop Logging:**
- **Function:** `log_select_crop_entry()` from `companion_file_utils.py`
- **When:** User approves crop proposal OR performs crop
- **Log to:** `data/training/select_crop_log.csv`

### **What Gets Logged:**
```csv
# selection_only_log.csv
session_id, set_id, chosen_path, negative_path_1, negative_path_2, ...

# select_crop_log.csv
session_id, set_id, directory, image_path_1, stage_1, width_1, height_1, chosen_index, crop_x1_norm, crop_y1_norm, crop_x2_norm, crop_y2_norm
```

---

## ⚡ **Implementation Requirements**

### **1. File Operations (move_file_with_all_companions)**
- Use `move_file_with_all_companions()` from `companion_file_utils.py`
- Ensures YAML/caption/etc. move with the image
- Logs via FileTracker for audit trail

### **2. Delete Staging (Fast Delete)**
- Pattern from Web Image Selector: `delete_staging/` at project root
- Use `Path.rename()` for instant move (much faster than Trash)
- User can manually empty staging directory later

### **3. Crop Execution (if crop approved)**
- Use PIL to open image
- Apply normalized crop coordinates (x1, y1, x2, y2 in range [0, 1])
- Save cropped image to `selected/` with same filename
- Log crop operation via FileTracker

### **4. Keyboard Shortcuts**
- `A` - Approve (with crop if proposed)
- `1/2/3/4` - Override to specific image
- `C` - Send to crop directory (manual crop later)
- `R` - Reject/Delete all
- `S` - Skip (no file moves)

---

## 🧪 **Testing Strategy**

1. **Sandbox Testing:** Use `sandbox/` directory with dummy images
2. **Verify Routes:**
   - Approve + no crop → `selected/`
   - Approve + crop → cropped file in `selected/`
   - Manual crop → `crop/`
   - Delete → `delete_staging/`
3. **Check Companions:** Verify YAML/caption files move with images
4. **Training Logs:** Verify CSV logs are written correctly
5. **Inline Validation:** Verify errors are caught immediately

---

## 📝 **Decision File Format (Unchanged)**

`.decision` sidecar files remain the same - they're the "single source of truth" for AI recommendations and user decisions:

```json
{
  "group_id": "20250719_143022",
  "images": ["stage1.png", "stage2.png", "stage3.png"],
  "ai_recommendation": {
    "selected_image": "stage3.png",
    "selected_index": 2,
    "reason": "Highest stage (rule-based)",
    "confidence": 1.0,
    "crop_needed": false,
    "crop_coordinates": null
  },
  "user_decision": {
    "action": "approve",  // or "override", "manual_crop", "reject", "skip"
    "selected_image": "stage3.png",
    "selected_index": 2,
    "crop_performed": false,
    "timestamp": "2025-10-21T12:00:00Z"
  }
}
```

---

## 🚨 **Safety Rules (Critical!)**

1. **NO in-place image modifications** (except crop output to `selected/`)
2. **ALWAYS move companion files with images**
3. **ALWAYS log training data**
4. **ALWAYS use FileTracker for audit trail**
5. **Use delete_staging/** for fast deletion (NOT direct to Trash)

---

## 📊 **Success Metrics**

- ✅ Combines 2 tools into 1 (saves 50% of manual steps)
- ✅ Instant feedback via inline validation
- ✅ All file moves logged for recovery
- ✅ Fast deletion via staging directory
- ✅ Training data collected automatically

