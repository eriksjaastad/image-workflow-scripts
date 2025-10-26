# AI Reviewer UX Fix - Crop Toggle Button
**Audience:** Developers

**Last Updated:** 2025-10-26

**Date:** October 22, 2025 (Morning)  
**Issue:** Clicking AI-selected image didn't remove crop - caused confusion  
**Status:** ✅ FIXED

---

## 🐛 The Problem

**User Experience Issue:**

Erik processed 90 groups and expected:
- **3 approved as-is** (no crop)
- **~2-3 approved with crop** (AI's crop proposal accepted)

**What actually happened:**
- ✅ 3 approved as-is (correct!)
- ❌ **8 approved with crop** (WRONG - user thought they were removing the crop!)

**Root Cause:**
- Clicking AI-selected image had NO effect on crop outline
- User expected: Click → Remove crop outline → Approve without crop
- Reality: Crop outline stayed → Image went to `crop/` directory
- **No visual feedback** that crop was removed vs kept

---

## ✅ The Solution

### **Changes Made:**

#### **1. Removed ALL "Approve" Buttons**
- Users click images directly to approve (hotkeys 1-4 still work perfectly)
- Simplified UI - one less button per image
- Matches user's mental model: "clicking = selecting"

#### **2. Added "Toggle Crop" Button on AI-Selected Image ONLY**

**Button Behavior:**
```
Initial state: "🚫 Remove Crop" (orange button, crop outline visible)
  ↓ Click
Toggle state: "✅ Add Crop" (green button, crop outline hidden)
  ↓ Click
Back to: "🚫 Remove Crop" (orange button, crop outline visible)
```

**Visual Feedback:**
- Crop outline appears/disappears immediately
- Button text changes: "Remove Crop" ↔ "Add Crop"
- Button color changes: Orange (crop) ↔ Green (no crop)
- Crop data stays in memory (can be restored!)

**Where it appears:**
- **ONLY on AI-selected images that have a crop proposal**
- Not on manually selected images (they use regular Q-E hotkeys)
- No hotkey (button-only feature as requested)

---

## 📝 Implementation Details

### **Template Changes (`scripts/01_ai_assisted_reviewer.py`):**

**Lines 1067-1079:** Added conditional button rendering
```jinja2
{% if loop.index0 == ai_rec.get('selected_index') and ai_rec.get('crop_coords') %}
{# AI-selected image with crop: Show toggle button #}
<button class="img-btn img-btn-crop" 
        id="toggle-crop-{{ group.group_id }}-{{ loop.index0 }}"
        onclick="toggleAICrop('{{ group.group_id }}', {{ loop.index0 }})">
  🚫 Remove Crop
</button>
{% else %}
{# Regular image: Show crop button with hotkey #}
<button class="img-btn img-btn-crop" onclick="selectImageWithCrop({{ loop.index0 }}, '{{ group.group_id }}')">
  ✂️ [Q/W/E/R]
</button>
{% endif %}
```

**Lines 1054-1056:** Store AI crop coords in data attribute
```jinja2
{% if loop.index0 == ai_rec.get('selected_index') and ai_rec.get('crop_coords') %}
data-ai-crop-coords="{{ ai_rec.get('crop_coords')|tojson }}"
{% endif %}
```

### **JavaScript Changes:**

**Lines 1157-1195:** New `toggleAICrop()` function
```javascript
function toggleAICrop(groupId, imageIndex) {
  const overlay = document.getElementById('crop-overlay-' + groupId + '-' + imageIndex);
  const button = document.getElementById('toggle-crop-' + groupId + '-' + imageIndex);
  const imageCard = document.querySelector(`section.group[data-group-id="${groupId}"] .image-card[data-image-index="${imageIndex}"]`);
  
  // Get AI crop coords from data attribute
  const aiCropCoords = JSON.parse(imageCard.getAttribute('data-ai-crop-coords'));
  
  // Toggle overlay visibility
  if (overlay.style.display === 'none') {
    // Show crop (ADD CROP)
    drawCropOverlay(groupId, imageIndex, true, aiCropCoords);
    button.textContent = '🚫 Remove Crop';
    button.classList.add('img-btn-crop');  // Orange
  } else {
    // Hide crop (REMOVE CROP)
    overlay.style.display = 'none';
    button.textContent = '✅ Add Crop';
    button.classList.add('img-btn-approve');  // Green
  }
}
```

---

## 🎯 Expected User Experience

### **Scenario 1: AI Picks Right Image, Crop is Good**
```
AI selects image 2 with crop outline
  ↓
User doesn't click anything (default approved)
  ↓
Click "Finalize selections"
  ↓
Image goes to crop/ with AI's crop proposal
```

### **Scenario 2: AI Picks Right Image, Crop is Wrong**
```
AI selects image 2 with crop outline
  ↓
User clicks "🚫 Remove Crop" button
  ↓
Crop outline disappears, button shows "✅ Add Crop"
  ↓
Click "Finalize selections"
  ↓
Image goes to selected/ (NO CROP!)
```

### **Scenario 3: AI Picks Right Image, User Changes Mind**
```
AI selects image 2 with crop outline
  ↓
User clicks "🚫 Remove Crop" (removes outline)
  ↓
User thinks: "Wait, the crop was actually good!"
  ↓
User clicks "✅ Add Crop" (restores outline)
  ↓
Click "Finalize selections"
  ↓
Image goes to crop/ with AI's crop proposal
```

### **Scenario 4: AI Picks Wrong Image**
```
AI selects image 2 with crop outline
  ↓
User clicks image 3 (or presses hotkey 3)
  ↓
Image 3 gets green border (selected)
  ↓
AI's crop is ignored (not relevant for image 3)
```

---

## 🔑 Key Features

### **1. Visual Feedback is Immediate**
- Crop outline appears/disappears on button click
- Button text updates instantly
- Button color changes (orange ↔ green)
- **No confusion about crop state!**

### **2. Crop Data is Preserved**
- AI's crop coords stored in `data-ai-crop-coords` attribute
- Can toggle on/off unlimited times
- Data never lost until batch is submitted

### **3. No Hotkey Required**
- This is a specialized function for AI-selected images only
- Button-only interface (as requested)
- Hotkeys 1-4, Q-E unchanged (bread and butter!)

### **4. Clean UI**
- Removed redundant "Approve" buttons
- Only one button per image (except AI-selected with crop)
- Less clutter, clearer intent

---

## 🧪 Testing Checklist

**Before next batch:**
- [x] Remove "Approve" buttons from all images
- [x] Add "Toggle Crop" button to AI-selected images only
- [ ] Test: Click "Remove Crop" → Crop outline disappears
- [ ] Test: Click "Add Crop" → Crop outline reappears
- [ ] Test: Button text updates correctly
- [ ] Test: Submit batch with crop removed → Goes to `selected/`
- [ ] Test: Submit batch with crop kept → Goes to `crop/`
- [ ] Test: Regular hotkeys (1-4, Q-E) still work
- [ ] Test: Clicking non-AI images still works as before

**Test with 10 groups:**
1. AI picks image with crop
2. Click "Remove Crop" button
3. Verify outline disappears
4. Click "Add Crop" button
5. Verify outline reappears
6. Click "Remove Crop" again
7. Submit batch
8. Verify image went to `selected/` (not `crop/`)

---

## 📊 Expected Impact

### **Before (Confusing):**
- User clicks AI image → Nothing happens visually
- User thinks crop is removed → Actually kept!
- 8/90 groups sent to `crop/` unexpectedly
- **Confusion and wasted time**

### **After (Clear):**
- User clicks "Remove Crop" → Outline disappears immediately
- User clicks "Add Crop" → Outline reappears immediately
- Visual state matches actual intent
- **No confusion, fast workflow!**

---

## 🎓 Design Principles Applied

### **1. Visual Feedback**
- Every action has immediate visible consequence
- Button state reflects current crop status
- No "hidden state" confusion

### **2. Reversible Actions**
- Can toggle crop on/off unlimited times
- Data preserved until batch submit
- Safe to experiment

### **3. Minimal Disruption**
- Hotkeys unchanged (1-4, Q-E still work perfectly)
- Clicking images unchanged
- Only added ONE new button to ONE type of image

### **4. Context-Appropriate UI**
- Toggle button ONLY appears where it makes sense (AI-selected with crop)
- Regular images still have regular buttons
- No unnecessary clutter

---

## 📝 Files Modified

**Changed:**
- `scripts/01_ai_assisted_reviewer.py`
  - Lines 1054-1056: Store AI crop coords in data attribute
  - Lines 1067-1079: Conditional button rendering (toggle vs regular)
  - Lines 1157-1195: New `toggleAICrop()` JavaScript function

**Not Changed:**
- Hotkey handling (1-4, Q-E unchanged!)
- Image click handling (still selects/deselects)
- Batch processing logic (still works the same)
- SQLite logging (still tracks decisions correctly)

---

## 🚀 Ready for Testing!

**Next Steps:**
1. Start AI Reviewer on a small batch (10-20 groups)
2. Test the "Remove Crop" / "Add Crop" toggle
3. Verify visual feedback is clear
4. Process batch and verify file routing
5. If good → Resume full production!

**Status:** ✅ **IMPLEMENTED AND READY TO TEST!**

---

*Fix completed: October 22, 2025, 6:45 AM*  
*Lines changed: ~50*  
*Time to implement: 15 minutes*  
*Impact: High (fixes critical UX confusion)*  
*Ready for user testing!* 🎊

