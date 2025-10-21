# Current TODO List

**Last Updated:** 2025-10-21

---

## 🎯 Active Tasks

### Phase 2: AI Model Training

#### Data Validation & Collection
- [ ] **Real-time Data Validation in Collection Tools** [PRIORITY: HIGH]
  - **Goal:** Show validation errors IMMEDIATELY in the tools as you're working
  - **Where:** Web Image Selector, Desktop Multi-Crop, Character Sorter
  - **Ideas to explore:**
    - Run mini-validation after each logging action
    - Show warning banner in UI if data looks suspicious
    - Flash red/yellow indicator if embedding missing or dimensions = (0,0)
    - Add "Data Health" indicator in tool footer (✅ Green = good, ⚠️ Yellow = warnings, ❌ Red = errors)
  - **Benefits:**
    - Catch bugs within MINUTES instead of WEEKS
    - Immediate feedback loop while tool is fresh in mind
    - Can fix issues before thousands of entries are logged incorrectly
  - **Technical Questions:**
    - Performance impact of validation on every action?
    - How to show errors without disrupting workflow?
    - Validate synchronously or async in background?
  - **Status:** NEEDS INVESTIGATION - Added 2025-10-21 after discovering dimension logging bug 3 weeks too late

#### Model Training (Blocked until data issues resolved)
- [ ] Test different ranker anomaly weights (5x, 10x, 20x) to find optimal balance
- [ ] Integrate best ranker model into Phase 3 AI-Assisted Reviewer
- [ ] Re-train crop proposer once mojo1/mojo2 data is validated and usable
- [ ] Create integration test script for Phase 3 readiness

---

## 📅 Backlog

### Documentation
- [ ] Document training data structure rules in `AI_TRAINING_DATA_STRUCTURE.md`
- [ ] Create troubleshooting guide for common training issues

### Automation
- [ ] Set up daily validation report (cron job or manual)
- [ ] Add email/Slack alerts when validation fails

---

## ✅ Recently Completed

- [x] Extract training data from 15 historical projects (21,250 selections, 12,679 crops)
- [x] Compute embeddings for all training images (77,304 total)
- [x] Train Ranker v2 with project boundary validation
- [x] Train Ranker v3 with anomaly oversampling (518 anomaly cases)
- [x] Analyze anomaly cases to identify model training gaps
- [x] Fix Desktop Multi-Crop dimension logging bug
- [x] Re-compute missing mojo2 embeddings (17,834 new embeddings)
- [x] Create validation script (`scripts/ai/validate_training_data.py`)
- [x] Document lessons learned (`Documents/AI_DATA_COLLECTION_LESSONS_LEARNED.md`)

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
