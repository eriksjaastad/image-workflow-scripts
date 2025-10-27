# Current TODO List
**Status:** Active
**Audience:** Developers
**Policy:** This is the single authoritative TODO list. Do not create separate TODO docs; add sections here instead.


**Last Updated:** 2025-10-26 (Afternoon)

---

## 🎯 Active Tasks

### TOP PRIORITY: Artifact Groups (cross-group/misaligned images)
✅ Moved to Recently Completed (Oct 26, 2025)

### Phase 3: Two-Action Crop Flow (Reviewer)
- [ ] Add analytics view for “perfect crop” (final ≈ AI crop within 5%) in SQLite v3
- [ ] Optional migration: extend user_action enum to include approve_ai_suggestion and approve_ai_crop (Phase 3B)


### Phase 3: AI-Assisted Reviewer Testing

#### Ready to Use Today!
- [ ] **Test AI-Assisted Reviewer on New Project** [PRIORITY: HIGH]
  - **Status:** Tool exists at `scripts/01_ai_assisted_reviewer.py`
  - **Purpose:** Replaces Web Image Selector; integrated with Desktop Multi-Crop — looks at ALL images, selects best from each group, proposes crops
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

### AI Reviewer: Batch Summary Delete Count (Bug)
✅ Moved to Recently Completed (Oct 26, 2025)

### Test Follow-ups (Dashboard + Utils)
- [ ] Migrate tests off legacy desktop selector shim and remove file
  - File: `scripts/01_desktop_image_selector_crop.py` (now a compat shim)
  - Action: Update tests to import archived path `scripts/archive/01_desktop_image_selector_crop.py` or remove usages
  - Then: Delete the shim file once tests no longer reference it
- [ ] Fix prompt extraction tests (scripts/tests/test_prompt_extraction.py) — 3 failures
- [ ] Satisfy file safety audit (scripts/tests/test_file_safety_audit.py)
- [ ] Crop overlay rounding to integers (scripts/tests/test_ai_assisted_reviewer_batch.py)
- [ ] Full pytest rerun and address any remaining stragglers


---

## 📅 Backlog

### Historical Crop Data Extraction (Experiment)
- [ ] **Extract crop coordinates from historical projects using image matching** [PRIORITY: MEDIUM]
  - **Goal:** Recover thousands of crop training examples by comparing original vs cropped images
  - **Method:** 
    1. Use project manifests to identify date ranges
    2. Find cropped images (files within project date range in `_cropped/` or `_final/` directories)
    3. Find matching original images (same filename in original/raw directory)
    4. Use OpenCV template matching to find exact crop location
    5. Extract crop coordinates (x1, y1, x2, y2)
    6. Normalize coordinates (0.0-1.0 range)
    7. Write to training CSV or SQLite (decide which)
  - **Why This Will Work:**
    - No compression/resizing in workflow = exact pixel match
    - Template matching will find location with 99.9%+ confidence
    - Fast processing (seconds per image)
  - **Implementation:**
    - Proof of concept: Test on 10 image pairs first
    - Visual verification: Show matches overlaid
    - Batch processing: Process all historical projects
    - Validation: Manual review of sample matches
  - **Potential Value:**
    - Could recover 5,000-10,000 crop training examples
    - Dramatically improve Crop Proposer model
    - Learn from historical crop patterns
  - **Output Format:**
    - Training data (not decision tracking)
    - Either: Add to crop training CSV
    - Or: Store in separate "recovered_crops.csv"
    - Include: timestamp, project_id, filename, crop_coords, image_width, image_height
  - **Projects to Process:**
    - Mojo1, Mojo2 (and any other finished projects with original + cropped pairs)
  - **Status:** EXPERIMENT - Build proof of concept first, then scale if successful

### Documentation
- [ ] Document training data structure rules in `AI_TRAINING_DATA_STRUCTURE.md`
- [ ] Create troubleshooting guide for common training issues

### Dashboard Improvements
- [ ] **Reimagine/Simplify Productivity Dashboard** [PRIORITY: MEDIUM]
  - **Issue:** Too many graphs that aren't actually useful
  - **Keep:** 
    - Build vs Actual (helpful, locked in)
    - Billing Efficiency Tracker (fine)
    - Productivity Table (pretty good)
    - Input vs Output (favorite graph)
  - **Reconsider/Remove:**
    - Project Comparison (confusing - operations > total images?)
    - Files Processed by Project (lines too small, hard to read with many projects)
    - Other graphs that don't provide clear insights
  - **Improvements Needed:**
    - Make it easier to view just last 2-3 projects together (not all at once)
    - Better default filters (e.g., only show recent projects by default)
    - Larger/clearer visualizations (especially for time-series)
    - Consider minimum thresholds for graphs (500+ files?) to avoid flatlined data
  - **Goal:** Less overwhelming, more focused on actionable insights

- [ ] Composition Metrics (2-up vs 3-up) [PRIORITY: HIGH]
  - **Goal:** Establish baseline composition per project (group size distribution) and kept rates (approve/crop) from the decisions DB; inform predictions for future projects.
  - **Analyzer Script:** Create `scripts/tools/analyze_composition.py` to compute per-project:
    - groups_by_size (e.g., {2: N, 3: N})
    - by_action counts (approve/crop/reject)
    - kept rates per group size and overall
  - **Historical Baseline:** Extend snapshot pipeline to persist composition metrics across all projects (daily snapshot or per-project summary).
  - **API:** Expose metrics via dashboard API for current and historical projects.
  - **UI:** Add dashboard cards/charts for composition and kept rates; compare Mojo1 vs Mojo3.
  - **Data Source:** SQLite v3 (`data/training/ai_training_decisions/*.db`) — DB is single source of truth.

- [ ] **Add AI Performance Stats to Dashboard** [PRIORITY: HIGH]
  - **Data Source:** SQLite v3 databases (`data/training/ai_training_decisions/*.db`)
  - **Metrics to Show:**
    - **Selection Accuracy:** % of times AI picked the same image as user
    - **Crop Accuracy:** % of times AI's crop was within 5% of user's final crop
    - **Trend Over Time:** Is AI getting better as it trains on more data?
    - **Per-Project Stats:** Compare AI performance across Mojo1, Mojo2, Mojo3, etc.
    - **Confidence Calibration:** Does high AI confidence = correct prediction?
  - **Visualizations:**
    - Line graph: Selection accuracy over time (by project)
    - Bar chart: AI correct vs user override (per project)
    - Gauge: Current AI accuracy (like a speedometer)
    - Table: Detailed breakdown (total decisions, correct, wrong, accuracy %)
  - **Why This Is Awesome:**
    - See AI improvement in real-time as you work!
    - Know when to trust AI's suggestions more
    - Celebrate milestones (50% accuracy → 70% → 85%!)
    - Identify which types of images AI struggles with
  - **Implementation:**
    - Query SQLite databases for `selection_match` and `crop_match` flags
    - Group by project and timestamp
    - Calculate rolling accuracy (e.g., last 100 decisions)
    - Display prominently on dashboard (maybe top section?)

### Automation
- [ ] Set up daily validation report (cron job or manual)
- [ ] Add email/Slack alerts when validation fails

### Backups & Delivery Automation
- [ ] Weekly full rollup to cloud (tar+upload with manifest) [PRIORITY: HIGH]
  - Source: `~/project-data-archives/image-workflow/`
  - Compress: one tar.zst per week + manifest (counts, sizes, sha256)
  - Upload: `gbackup:weekly-rollups/`
  - Retention: keep 12 weeks locally + cloud
- [ ] Auto-upload finished ZIP from 07_finish_project to Drive [PRIORITY: MEDIUM]
  - Hook: after successful finish with `--commit`
  - Target: `gbackup:deliveries/<projectId>/`
  - Flow: copy → check → (optional) delete local zip
  - Reuse rclone remote `gbackup` and daily cron log

---

## ✅ Recently Completed

**SQLite v3 Training System (Oct 22, 2025 - Night/Morning)**
- [x] **Design and implement SQLite-based training data system** ⭐ **GAME CHANGER!**
- [x] Create database schema with decision tracking + crop matching
- [x] Build utility functions (`scripts/utils/ai_training_decisions_v3.py`)
- [x] Write comprehensive tests (18 tests, all passing)
- [x] Integrate into AI Reviewer (log AI decisions + create `.decision` files)
- [x] Integrate into Desktop Multi-Crop (read `.decision` files + update with final crops)
- [x] Document complete system (`Documents/AI_TRAINING_DECISIONS_V3_IMPLEMENTATION.md`)
- [x] Add to Technical Knowledge Base (365 lines)
- [x] Fix Desktop Multi-Crop performance lag (plt.draw → draw_idle, 20-40x faster!)
- [x] Test full workflow (90 decisions logged successfully)

**AI Reviewer UX Improvements (Oct 22, 2025 - Morning)**
- [x] **Add "Remove Crop" toggle button to AI-selected images** ⭐ **TESTED AND WORKING!**
- [x] Add regular crop button to AI-selected images (for manual cropping)
- [x] Auto-launch browser when starting AI Reviewer
- [x] Document --batch-size and other flags in header
- [x] Remove confusing "Approve" buttons from all images

**Phase 2: AI Training (90% Complete)**
- [x] Extract training data from 15 historical projects (21,250 selections, 12,679 crops)
- [x] Compute embeddings for all training images (77,304 total)
- [x] Train Ranker v2 with project boundary validation
- [x] **Train Ranker v3 with anomaly oversampling** ⭐ **94.4% anomaly accuracy!**
- [x] Analyze anomaly cases to identify model training gaps (518 cases)
- [x] Fix Desktop Multi-Crop dimension logging bug
- [x] Re-compute missing mojo2 embeddings (17,834 new embeddings)
- [x] Create validation script (`scripts/ai/validate_training_data.py`)
- [x] Document lessons learned (`Documents/archives/ai/AI_DATA_COLLECTION_LESSONS_LEARNED.md`)

**Data Integrity (Just Completed - Oct 21, 2025 Morning)**
- [x] **Integrate inline validation into all data collection tools** ⭐ **DONE!**
- [x] Add dimension validation to `log_select_crop_entry()`
- [x] Add path validation to `log_selection_only_entry()`
- [x] Create test suite (`scripts/tests/test_inline_validation.py`)
- [x] Documentation (`Documents/archives/misc/INLINE_VALIDATION_GUIDE.md`)

**Crop Training Data Schema Evolution (Oct 21, 2025 Afternoon)**
- [x] **Design and implement NEW minimal crop training schema** ⭐ **8 columns instead of 19!**
- [x] Create `log_crop_decision()` function with strict validation
- [x] Update AI-Assisted Reviewer to use new schema
- [x] Document schema evolution and benefits (`Documents/archives/misc/CROP_TRAINING_SCHEMA_V2.md`)
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


### AI Automation (Imported 2025-10-26)
- Backlog triage needed; items below are placeholders—convert to scoped tasks when prioritized:
  - Add configurable safe-zone allowlist (read from `configs/` and used in validation)
  - Add retry with backoff for per-crop failures and partial progress resume
  - Queue manager maintenance CLI: `clear_completed` + timing/log rotation helpers
  - Pre-commit installer script for root-file policy hook in `scripts/tools/`
  - Makefile shortcuts: `make timing`, `make queue-test`, `make process-fast`
  - CLI to delete/restore a batch to `__delete_staging` using companion utils
  - Processor: enforce decisions DB linkage in preflight (fail with clear error when missing)
  - Docs: Queue quickstart + analyzer usage (place in `Documents/guides/`)
  - Docs: Commit communication standard snippet in `Documents/README.md`
  - Dashboard: queue stats panel (pending/processing/completed/failed)
  - Dashboard: processing time trends and batches-per-session charts
  - Tool: audit of queue vs filesystem and DB (orphan/consistency report)
