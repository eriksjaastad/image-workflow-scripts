# AI-Assisted Reviewer Clone + AI Integration
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26

**Date:** October 21, 2025  
**Strategy:** Clone web_image_selector.py and add AI features, NOT build from scratch

---

## 🎯 **Core Principle:**
**"AI-Assisted Reviewer WITH AI integration"**

NOT: "AI tool that kinda works like web selector"  
YES: "Exact clone of web selector + AI recommendations"

---

## 📋 **Checklist: What Must Be Identical**

### **1. Configuration (Top of File)**
```python
# From web_image_selector.py
BATCH_SIZE = 100  # ← AT TOP, easy to find!
THUMBNAIL_MAX_DIM = 768
STAGE_NAMES = ("stage1_generated", "stage1.5_face_swapped", "stage2_upscaled")
FOCUS_TIMER_WORK_MIN = 15
FOCUS_TIMER_REST_MIN = 5
```

### **2. Hotkeys (Exact Match)**
```javascript
'1', '2', '3', '4'  → Select image (no crop) + auto-advance
'q', 'w', 'e', 'r'  → Select WITH crop + auto-advance
'Enter' / 'Space'   → Next group
'ArrowUp'           → Previous group
```

### **3. Workflow (Exact Match)**
1. Load batch of 100 groups
2. User makes selections (queued in memory)
3. Scroll to bottom to enable "Finalize selections" button
4. Click finalize → ALL operations execute
5. Option to load next batch

### **4. Button Names**
- "Finalize selections" (not "Process" or "Submit")

### **5. UI Structure**
- Compact header
- Batch count at top
- Groups as scrollable sections
- "Finalize" button at bottom (disabled until scroll)

---

## 🤖 **AI Integration Points (ADDITIONS ONLY)**

1. **On page load:** Run AI for current batch
2. **Visual indicators:** 
   - AI PICK badge
   - CROP badge if AI suggests crop
   - Crop overlay (red 1px box)
3. **Default behavior:**
   - AI pick is pre-selected visually
   - Pressing Enter accepts AI recommendation
4. **Override:**
   - Use hotkeys to override AI pick
   - AI suggestion doesn't block user

---

## 🔧 **Implementation Strategy**

**Phase 1:** Copy web selector structure
- Batch loading
- Hotkey system
- Finalize button
- Auto-advance logic

**Phase 2:** Add AI (minimal changes)
- Call AI on batch load
- Display recommendations
- Show crop overlay
- Log training data

**Result:** Web selector that happens to have AI helping you

---

**Key:** If web selector has it, we must have it. If web selector doesn't, we probably don't need it either.

