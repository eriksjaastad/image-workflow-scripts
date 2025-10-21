# Visual Crop Overlay Tests

## Overview
These tests verify that crop overlays are rendered correctly in the browser.

## Running Visual Tests

### 1. Automated Tests (Python)
```bash
cd /Users/eriksjaastad/projects/Eros\ Mate
python scripts/tests/test_ai_assisted_reviewer.py
```

**What it tests:**
- ✅ Model loading
- ✅ Rule-based recommendations
- ✅ Crop coordinate validation
- ✅ Aspect ratio preservation
- ✅ Decision file creation
- ✅ Batch size configuration
- ✅ Crop overlay pixel math

### 2. Visual Browser Tests (Manual)
```bash
open scripts/tests/test_crop_overlay_visual.html
```

**What it tests:**
- ✅ Square images (1024x1024)
- ✅ Landscape images (1920x1080)
- ✅ Portrait images (1080x1920)
- ✅ Full-image crops (edge case)
- ✅ Visual accuracy of overlay positioning
- ✅ Dark overlay around crop
- ✅ Green border on crop area

**Expected Results:**
- Each test should show ✅ PASS
- Green crop overlay should match the described dimensions
- Area outside crop should be darkened
- Pixel coordinates should match expected values

## Test Cases

### Test 1: Square Image
- **Crop:** (0.1, 0.2, 0.9, 0.8)
- **Removes:** 10% left, 20% top, 10% right, 20% bottom
- **Crop Area:** 48% of original

### Test 2: Landscape Image
- **Crop:** (0.1, 0.2, 0.9, 0.8)
- **Aspect Ratio:** Should differ from original
- **Crop Area:** 48% of original

### Test 3: Portrait Image
- **Crop:** (0.25, 0.1, 0.75, 0.9)
- **Removes:** 25% left/right, 10% top, 10% bottom
- **Crop Area:** 40% of original

### Test 4: Full Image (No Crop)
- **Crop:** (0.0, 0.0, 1.0, 1.0)
- **Result:** Overlay covers entire image
- **Crop Area:** 100% (no actual crop)

## Interpreting Results

### ✅ PASS
All pixel coordinates match expected values within rounding tolerance.

### ❌ FAIL
Overlay positioning is incorrect. Check:
1. Image display dimensions vs natural dimensions
2. Normalized coordinate conversion
3. CSS positioning of overlay div

## Troubleshooting

If visual tests fail:
1. Check browser console for JavaScript errors
2. Verify CSS `position: relative` on image container
3. Verify overlay has `position: absolute`
4. Check that normalized coords are in [0, 1] range
5. Ensure image is fully loaded before drawing overlay

## Integration with AI-Assisted Reviewer

The same crop overlay logic is used in:
- `scripts/01_ai_assisted_reviewer.py` → `drawCropOverlay()` function

Any changes to overlay rendering should be:
1. Updated in both files
2. Tested with visual HTML tests
3. Verified with automated Python tests
4. Manually tested in actual reviewer UI

