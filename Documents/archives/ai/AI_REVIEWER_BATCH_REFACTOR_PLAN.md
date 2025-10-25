# AI Reviewer Batch Processing Refactor
**Date:** October 21, 2025  
**Goal:** Clone web_image_selector.py batch processing workflow

---

## 🎯 **Changes Required**

### **1. Change UI from Single-Group to Batch View**
**Current:** Shows one group at a time (`/` route)  
**Target:** Shows all batch groups at once (scroll through them)

**Template Changes:**
- Change from single `.group-card` to loop through ALL groups in batch
- Each group becomes a `<section class="group">` (like web selector)
- Add scroll-to-enable "Finalize selections" button at bottom

### **2. JavaScript: Queue Decisions (Don't Submit Immediately)**
**Current:** `submitImageDecision()` → `/submit` → file ops → navigate  
**Target:** `submitImageDecision()` → update `groupStates` in memory → stay on page

**JS Changes:**
```javascript
let groupStates = {}; // { groupId: { selectedImage: idx, crop: bool } }

function selectImage(imageIndex, groupId) {
  groupStates[groupId] = { selectedImage: imageIndex, crop: false };
  updateVisualState();
  updateSummary();
  // NO navigation, NO fetch
}

function selectImageWithCrop(imageIndex, groupId) {
  groupStates[groupId] = { selectedImage: imageIndex, crop: true };
  updateVisualState();
  updateSummary();
}
```

### **3. Add Finalize Button Handler**
```javascript
processBatchButton.addEventListener('click', async () => {
  // Convert groupStates to selections array
  const selections = Object.keys(groupStates).map(groupId => ({
    groupId: groupId,
    selectedImage: groupStates[groupId].selectedImage,
    crop: groupStates[groupId].crop || false
  }));
  
  // Send to server
  const response = await fetch('/process-batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ selections })
  });
  
  // Reload or go to next batch
  if (response.ok) {
    window.location.href = '/next-batch';
  }
});
```

### **4. Backend: Add `/process-batch` Endpoint**
**Clone from web_image_selector.py lines 1654-1750**

```python
@app.route("/process-batch", methods=["POST"])
def process_batch():
    """Process all queued selections (clone of web selector logic)"""
    data = request.get_json()
    selections = data.get("selections", [])
    
    tracker = app.config["TRACKER"]
    selected_dir = app.config["SELECTED_DIR"]
    crop_dir = app.config["CROP_DIR"]
    delete_staging_dir = app.config["DELETE_STAGING_DIR"]
    
    kept_count = 0
    for selection in selections:
        group_id = selection["groupId"]
        selected_idx = selection["selectedImage"]
        crop = selection.get("crop", False)
        
        # Find group
        groups = app.config["GROUPS"]
        group = next((g for g in groups if g.group_id == group_id), None)
        if not group:
            continue
        
        # Perform file operations
        action = "manual_crop" if crop else "approve"
        perform_file_operations(group, action, selected_idx, ...)
        kept_count += 1
    
    return jsonify({"status": "ok", "kept": kept_count})
```

### **5. Remove Immediate File Operations from `/submit`**
**Current:** `/submit` → perform_file_operations()  
**Target:** `/submit` → ONLY save decision file

---

## 📋 **Execution Checklist**
- [ ] Update page template to show ALL groups in batch
- [ ] Add `groupStates` JS variable
- [ ] Replace `submitImageDecision` with state updates
- [ ] Add "Finalize selections" button + scroll-to-enable logic
- [ ] Add `/process-batch` endpoint (clone web selector)
- [ ] Remove file ops from `/submit` route
- [ ] Test with 20-group batch

---

**Result:** Exact clone of web selector workflow + AI recommendations overlaid

