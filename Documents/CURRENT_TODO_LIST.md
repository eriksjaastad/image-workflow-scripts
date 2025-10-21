# Current TODO List

**Last Updated:** 2025-10-21 (Morning)

---

## 🎯 Active Tasks

### Phase 3: AI-Assisted Reviewer Testing

#### Ready to Use Today!
- [ ] **Test AI-Assisted Reviewer on New Project** [PRIORITY: HIGH]
  - **Status:** Tool exists at `scripts/01_ai_assisted_reviewer.py`
  - **Purpose:** REPLACES Web Image Selector - looks at ALL images, selects best from each group, proposes crops
  - **Currently:** Rule-based (picks highest stage, no crop proposals)
  - **Future:** Will integrate Ranker v3 + Crop Proposer models
  - **Usage:** `python scripts/01_ai_assisted_reviewer.py <new-project-directory>/`
  - **Test Plan:**
    1. Start new project with raw images
    2. Run AI-Assisted Reviewer instead of Web Image Selector
    3. Review AI recommendations (A to approve, 1-4 to override)
    4. Check .decision sidecar files are created
    5. Evaluate if this workflow is better than manual selection

#### Model Integration (Optional - Already Have Great Models)
- [ ] Integrate Ranker v3 into AI-Assisted Reviewer
  - **Ranker v3 stats:** 94.4% anomaly accuracy, 98.1% overall
  - **Replace:** Rule-based logic with model predictions
  - **Add:** Confidence scores from model
- [ ] Integrate Crop Proposer v1 (if it completed training)
  - **Check:** Does `crop_proposer_v1.pt` exist?
  - **Add:** Crop suggestions to reviewer UI

---

## 📅 Backlog

### Historical Crop Data Extraction (Experiment)
- [ ] **Extract crop dimensions from before/after projects** [PRIORITY: MEDIUM]
  - **Goal:** Recover crop training data from historical projects by comparing originals vs cropped finals
  - **Method:** 
    1. For each historical project with RAW + FINAL directories
    2. Find matching images (same filename in both)
    3. Compare image dimensions: if FINAL is smaller, calculate crop coordinates
    4. Compute crop box coordinates based on aspect ratio and size difference
    5. Write extracted crop data to `select_crop_log.csv`
  - **Challenges:**
    - Crop might have been centered, top-aligned, or manual - hard to know which
    - Aspect ratio changes would be ambiguous
    - Only useful if crops were simple (centered or edge-aligned)
  - **Value:** Could recover thousands of crop training examples from old projects
  - **Status:** EXPERIMENT - may or may not yield useful data

### Documentation
- [ ] Document training data structure rules in `AI_TRAINING_DATA_STRUCTURE.md`
- [ ] Create troubleshooting guide for common training issues

### Automation
- [ ] Set up daily validation report (cron job or manual)
- [ ] Add email/Slack alerts when validation fails

---

## ✅ Recently Completed

**Phase 2: AI Training (90% Complete)**
- [x] Extract training data from 15 historical projects (21,250 selections, 12,679 crops)
- [x] Compute embeddings for all training images (77,304 total)
- [x] Train Ranker v2 with project boundary validation
- [x] **Train Ranker v3 with anomaly oversampling** ⭐ **94.4% anomaly accuracy!**
- [x] Analyze anomaly cases to identify model training gaps (518 cases)
- [x] Fix Desktop Multi-Crop dimension logging bug
- [x] Re-compute missing mojo2 embeddings (17,834 new embeddings)
- [x] Create validation script (`scripts/ai/validate_training_data.py`)
- [x] Document lessons learned (`Documents/AI_DATA_COLLECTION_LESSONS_LEARNED.md`)

**Data Integrity (Just Completed - Oct 21, 2025 Morning)**
- [x] **Integrate inline validation into all data collection tools** ⭐ **DONE!**
- [x] Add dimension validation to `log_select_crop_entry()`
- [x] Add path validation to `log_selection_only_entry()`
- [x] Create test suite (`scripts/tests/test_inline_validation.py`)
- [x] Documentation (`Documents/INLINE_VALIDATION_GUIDE.md`)

**Crop Training Data Schema Evolution (Oct 21, 2025 Afternoon)**
- [x] **Design and implement NEW minimal crop training schema** ⭐ **8 columns instead of 19!**
- [x] Create `log_crop_decision()` function with strict validation
- [x] Update AI-Assisted Reviewer to use new schema
- [x] Document schema evolution and benefits (`Documents/CROP_TRAINING_SCHEMA_V2.md`)
- [x] Add to Technical Knowledge Base
- [ ] **BACKLOG: Migrate 7,194 legacy rows to new schema** (Optional - keep both for now)

---

## 🗑️ Cancelled

- [x] ~~Extract crop data from jmlimages-random~~ (No crop data exists - only selections)
- [x] ~~Extract crop data from tattersail-0918~~ (No crop data exists - only selections)
- [x] ~~Extract crop data from historical projects~~ (Crop coordinates never logged before data collection system)

---

## 📝 Notes

- **Data Collection Crisis:** Discovered 3 weeks after the fact that Desktop Multi-Crop was logging dimensions as (0,0) instead of actual values. Lost ~6,700 potential training examples. Need real-time validation to prevent this in the future.
- **Embeddings Issue:** Some embeddings exist in cache but files are missing from disk. Need verification step after embedding generation.
- **Historical Data Limitation:** Only projects processed AFTER data collection system was built have usable crop data.
