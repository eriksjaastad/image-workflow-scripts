# AI-Assisted Reviewer: Make It Match Web Image Selector
**Goal:** Seamless AI integration - as if we took 01_web_image_selector.py and added AI

---

## 🎯 **Priority Order:**

### **1. Layout/UI Match** ✅ (Mostly Done)
- [x] Compact header (already done - you like it!)
- [x] Minimal padding
- [x] Remove rounded corners on images
- [x] Clean, fast interface
- [ ] **Button naming:** Change "Process" to match web selector button name
- [ ] **Verify:** Does web selector say "Finalize selections" or "Process Batch"?

---

### **2. Hotkeys (CRITICAL - Must Match Exactly)**

**Web Image Selector Hotkeys:**
```
1, 2, 3, 4      → Select image (no crop)
q, w, e, r      → Select image WITH crop
Enter / Space   → Next group (advance after selection)
ArrowUp         → Previous group
```

**Current AI-Assisted Reviewer:**
```
1, 2, 3, 4      → Select image (visual selection only)
(No crop hotkeys yet)
(No navigation hotkeys - removed per your request earlier)
```

**TODO:**
- [ ] Add `q/w/e/r` hotkeys for "approve with crop"
- [ ] Add `Enter`/`Space` to advance to next group after decision
- [ ] Add `ArrowUp` to go back one group
- [ ] Remove individual per-image buttons (keep hotkeys only)
- [ ] Auto-advance after pressing 1/2/3/4 or q/w/e/r (like web selector)

---

### **3. Batch Processing (CRITICAL)**

**Web Image Selector Workflow:**
1. Load 100 groups
2. User makes selections (hotkeys)
3. Selections are **queued** (not processed immediately)
4. User clicks "Finalize selections" at bottom
5. **THEN** all file operations happen at once

**Current AI-Assisted Reviewer:**
- Processes files **immediately** on each decision (WRONG!)

**TODO:**
- [ ] Remove immediate file processing
- [ ] Store decisions in memory/JSON (like web selector)
- [ ] Add "Finalize selections" button at bottom (scroll to see it)
- [ ] Process ALL decisions when button clicked
- [ ] Safety: Only enable button after scrolling to bottom (like web selector)

---

### **4. AI Integration (Already Done!)**
- [x] AI runs on page load for all groups
- [x] AI recommendation shown (pick + crop)
- [x] Crop overlay displayed
- [x] AI scores shown in console

**Enhancement:**
- [ ] If AI suggests crop, default to "approve with crop" on Enter (not just approve)

---

### **5. Batch Size**
- [x] Already configurable (default: 20)
- [ ] Change default to 100 (match web selector)

---

### **6. File Operations (Current - Keep This!)**
**Destination logic (correct):**
- Approve (no crop) → `selected/`
- Approve (with AI crop) → `selected/` (crop performed)
- Manual crop → `crop/`
- Reject → `delete_staging/`

**But:** Delay ALL moves until "Finalize" clicked!

---

## 📋 **Implementation Plan:**

### **Phase 1: Hotkeys (30 min)**
1. Read web selector hotkey implementation
2. Copy exact hotkey mapping to AI reviewer
3. Add auto-advance on selection (Enter/Space triggers next group)
4. Test: Can you fly through with just keyboard?

### **Phase 2: Batch Processing (45 min)**
1. Remove immediate file operations
2. Store decisions in `decisions` dict (in memory)
3. Add "Finalize selections" button at bottom
4. Process all on finalize
5. Add scroll-to-bottom safety check

### **Phase 3: Button Names & Polish (15 min)**
1. Match button text to web selector
2. Verify header matches
3. Test full workflow

---

## 🧪 **Testing Checklist:**

**Must work exactly like web selector:**
- [ ] Press `1` → Selects image 1, advances to next group
- [ ] Press `q` → Selects image 1 WITH crop, advances to next group
- [ ] Press `Enter` → Advances to next group
- [ ] Press `ArrowUp` → Goes back one group
- [ ] Scroll to bottom → "Finalize" button enables
- [ ] Click "Finalize" → ALL moves happen at once
- [ ] Speed: Can process 100 groups in 5-10 minutes (like web selector)

---

## 🎨 **Key Difference from Web Selector:**

**Web Selector:** You decide which image to keep  
**AI Reviewer:** AI suggests which image + whether to crop, you can override with hotkeys

**But the UX should feel identical!**

---

**Want me to start with Phase 1 (Hotkeys)?** That's the biggest blocker to speed right now.

