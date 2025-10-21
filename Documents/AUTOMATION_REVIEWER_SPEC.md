# Automation Reviewer - Technical Specification

**Created:** October 16, 2025  
**Status:** Specification / Planning  
**Target Start:** Sunday, October 20, 2025

---

## 📚 Quick Navigation - All AI Documents

**Current Document:** AUTOMATION_REVIEWER_SPEC.md (Phase 3 - Build Review UI)

### Core Workflow Documents:
1. **[AI_TRAINING_PHASE2_QUICKSTART.md](AI_TRAINING_PHASE2_QUICKSTART.md)** ⭐ Phase 2
   - Step-by-step: Install deps → Train models → Test inference
   - **Status:** ✅ Training data ready (17,032 examples from Mojo 1 + Mojo 2)

2. **[AUTOMATION_REVIEWER_SPEC.md](AUTOMATION_REVIEWER_SPEC.md)** ⭐ Phase 3 (YOU ARE HERE)
   - Complete spec for review UI (`scripts/07_automation_reviewer.py`)
   - **Status:** Documented, ready to build after Phase 2 models are trained

3. **[CURRENT_TODO_LIST.md](CURRENT_TODO_LIST.md)** - Automation Pipeline section
   - Lines 251-490: Full pipeline, data formats, testing plan
   - **Status:** Living document with task tracking

### Implementation Order:
```
Phase 1: ✅ Data Collection (COMPLETE - 17k examples)
         ↓
Phase 2: ⏳ Train AI Models (AI_TRAINING_PHASE2_QUICKSTART.md)
         ↓
Phase 3: ⏳ Build Review UI (THIS DOCUMENT)
         ↓
Phase 4: ⏳ Test in Sandbox + Iterate
```

---

## 🎯 **Purpose**

Web UI for reviewing AI automation decisions before applying them. Based on Web Image Selector platform with automation-specific features.

---

## 📋 **Core Requirements**

### **Platform Base**
- Built on existing Web Image Selector codebase (`scripts/01_web_image_selector.py`)
- Reuse: Flask server, HTML/CSS structure, keyboard shortcuts, batch processing
- New: Automation decision display, crop visualization overlay, approval workflow

### **Data Source**
- Input: `sandbox/automation_decisions.jsonl` (AI's marked decisions)
- Output: `sandbox/approved_decisions.jsonl` (user-approved only)

---

## 🖼️ **UI Layout**

### **Group Display (like Web Image Selector)**
```
┌─────────────────────────────────────────────────────────────┐
│  Group 3 of 45                               [Reviewed: 2/45]│
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Image 1  │  │ Image 2  │  │ Image 3  │  │ Image 4  │   │
│  │ (stage1) │  │ (stage2) │  │ (stage3) │  │ (stage2) │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                               ▲ AI PICK                      │
│                     ┌─────────────────────┐                 │
│                     │ ╔═══════════════╗   │ ← Crop overlay  │
│                     │ ║   PROPOSED    ║   │                 │
│                     │ ║   CROP AREA   ║   │                 │
│                     │ ╚═══════════════╝   │                 │
│                     └─────────────────────┘                 │
│                                                               │
│  AI Decision:                                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ STEP 1: Selected Image 3 (stage3)                       ││
│  │ Reason: Highest stage, passed anomaly checks            ││
│  │ Confidence: 95%                                          ││
│  │                                                          ││
│  │ STEP 2: Crop recommended                                ││
│  │ Reason: Body cut at waist, excess top background        ││
│  │ Confidence: 87%                                          ││
│  └─────────────────────────────────────────────────────────┘│
│                                                               │
│  Your Decision:                                              │
│  [ Approve ]  [ Reject ]  [ Override ]  [ Skip ]            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 **Visual Elements**

### **AI's Chosen Image**
- **Green border** (3px solid #00ff00)
- **"AI Pick" badge** (top-right corner)
- **Confidence score** (shown on hover)

### **Crop Visualization**
- **Dotted green rectangle** overlaid on image
- **Canvas-based drawing** (not modifying image file)
- **Toggle visibility** (C key to show/hide)
- **Coordinates shown** in tooltip

### **Decision Info Panel**
- **Step 1 (Selection)**: Chosen image, reasoning, confidence
- **Step 2 (Crop)**: Yes/No decision, reasoning (if yes: crop coords), confidence
- **Anomaly flags**: Icons for hands/feet detected, other warnings

---

## ⌨️ **Keyboard Shortcuts**

### **Navigation**
- `Enter` / `↓` - Next group
- `↑` - Previous group
- `Shift+Enter` - Submit batch (apply approved decisions)

### **Actions**
- `A` - **Approve** (accept AI decision)
- `R` - **Reject** (keep all images, no action)
- `S` - **Skip** (review later)
- `1/2/3/4` - **Override** (manually select different image)

### **Crop Adjustments** (when overriding)
- `Q/W/E/R` - Select image + mark needs crop
- `C` - Toggle crop overlay visibility
- `Drag handles` - Adjust crop area (if implemented)

---

## 📊 **Batch Processing**

### **Workflow**
1. Load 20-50 groups per batch
2. Review each group (keyboard shortcuts)
3. Submit batch → writes approved decisions
4. Progress bar: "Reviewed 45/200 groups (22.5%)"
5. Load next batch

### **State Tracking**
- `approved` - AI decision accepted
- `rejected` - No action (keep all images)
- `overridden` - User picked different image/crop
- `skipped` - Review later

---

## 🔄 **Data Flow**

```
1. Automation runs → sandbox/automation_decisions.jsonl
   {
     "group_id": "...",
     "images": [...],
     "step1_selection": { "chosen": "img3.png", ... },
     "step2_crop": { "needs_crop": true, "crop_coords": [...], ... }
   }

2. Reviewer loads decisions → Display in UI

3. User reviews → Actions tracked

4. Submit batch → sandbox/approved_decisions.jsonl
   {
     "group_id": "...",
     "action": "approved|rejected|overridden",
     "final_choice": "img3.png",  (if approved/overridden)
     "crop_coords": [...],         (if crop needed)
     "timestamp": "..."
   }

5. Commit script → Executes approved moves (sandbox only)
```

---

## 🛠️ **Technical Implementation**

### **Backend (Flask)**
- Load `automation_decisions.jsonl` into memory
- Serve groups in batches
- Track review state (approved/rejected/overridden/skipped)
- Write approved decisions to output file
- Endpoint: `/api/automation/load_batch`
- Endpoint: `/api/automation/submit_batch`

### **Frontend (HTML/CSS/JS)**
- Reuse Web Image Selector template structure
- Add crop overlay canvas (HTML5 Canvas or SVG)
- JavaScript for drawing crop rectangles
- Keyboard event handlers
- State management for batch review
- Progress tracking UI

### **Crop Overlay Rendering**
```javascript
// Pseudocode
function drawCropOverlay(imageElement, cropCoords) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Normalize coords to image display size
  const [x1, y1, x2, y2] = cropCoords; // 0-1 normalized
  const rect = {
    x: x1 * imageElement.width,
    y: y1 * imageElement.height,
    width: (x2 - x1) * imageElement.width,
    height: (y2 - y1) * imageElement.height
  };
  
  // Draw dotted green rectangle
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 3;
  ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
}
```

---

## 📏 **Image Size Considerations**

**Current Concern:** Image display size for seeing crop details

**Options:**
1. **Start with existing size** (like Web Image Selector)
   - Quick to implement
   - Test if sufficient
   - Adjust later if needed

2. **Add zoom on hover** (future enhancement)
   - Hover over image → shows larger preview
   - Crop details more visible
   - Doesn't clutter default view

3. **Full-size modal** (future enhancement)
   - Click image → opens full-size view
   - Better crop detail visibility
   - More clicks required

**Decision:** Start with option 1, iterate based on actual use

---

## 🧪 **Testing Plan (Sandbox Only)**

### **Phase 1: UI Development**
- Build reviewer UI (reuse Web Image Selector)
- Test with mock decision data
- Validate keyboard shortcuts
- Test crop overlay rendering

### **Phase 2: Integration**
- Run automation on sandbox subset
- Load real decisions into reviewer
- Test approval workflow
- Verify output file format

### **Phase 3: End-to-End**
- Full pipeline: Automation → Review → Commit
- Test on larger sandbox dataset
- Validate file operations (sandbox only)
- Check companion file handling

---

## 📅 **Timeline**

**Sunday, October 20:**
- Start automation pipeline testing
- Generate first batch of decisions
- Begin reviewer UI development

**Following Week:**
- Iterate on reviewer UI based on usage
- Tune automation thresholds
- Validate decisions match expectations

---

## 🔒 **Safety Guarantees**

- ✅ Loads decisions from `sandbox/` only
- ✅ Writes approved decisions to `sandbox/` only
- ✅ NO automatic file operations
- ✅ Separate commit script required to execute moves
- ✅ All testing in sandbox until proven reliable
- ✅ Production never touched

---

## 📋 **Future Enhancements (Not Required for v1)**

- Interactive crop adjustment (drag handles)
- Zoom on hover for detail viewing
- Bulk approve (approve all high-confidence decisions)
- Filtering (show only low-confidence, show only crops, etc.)
- Metrics dashboard (approval rates, override patterns)
- Export decisions to CSV for analysis

---

## 📝 **Notes from Erik**

- "Based on web image selector platform" ✅
- "Image size might be a concern for crop details" (start simple, iterate)
- "Start Sunday" ✅
- "Everything AI learns from me applies to sandbox" ✅

---

## ✅ **Ready to Build**

All specs defined. Ready to start implementation Sunday with automation pipeline testing.

