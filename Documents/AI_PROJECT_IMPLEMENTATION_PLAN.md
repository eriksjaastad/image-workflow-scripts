# AI Automation Project - Implementation Plan & Progress Tracker

**Project Goal:** Build AI system to automate image selection and crop recommendations, with human review workflow

**Last Updated:** October 21, 2025  
**Training Data:** ✅ 21,250 selections + 12,679 crops from 15 historical projects  
**Phase 2 Status:** ~85% Complete - Data extraction done, model training in progress

---

## 🚨 CRITICAL: Training Data Structure (READ THIS FIRST)

### **Projects NEVER Mix**

Each project (Mojo 1, Mojo 2, Eleni, etc.) represents a **different character**. Training data from different projects **MUST NEVER be compared or mixed during training**.

**Key Rules:**
1. **Each CSV row is a complete image set** from ONE project
   - Winner + losers all from same project, same timestamp/group
   - Different stages (stage1, stage2, stage3) of the same generation

2. **Match by filename within project context**
   - Extract project ID from path: `/Users/.../mojo1/file.png` → `mojo1`
   - Match by filename only (ignore subdirectories)
   - Constrain matches to same project

3. **Training pairs must respect project boundaries**
   - ✅ CORRECT: Compare stage1 vs stage2 within mojo1
   - ❌ WRONG: Compare mojo1 image vs mojo2 image

4. **One unified model across all projects (recommended)**
   - Model learns your general aesthetic preferences
   - Training pairs always stay within same project
   - See `AI_TRAINING_PHASE2_QUICKSTART.md` for detailed implementation guide

---

## 📚 Quick Navigation

**Related Documents:**
- **[AI_TRAINING_PHASE2_QUICKSTART.md](AI_TRAINING_PHASE2_QUICKSTART.md)** - Step-by-step Phase 2 guide
- **[AUTOMATION_REVIEWER_SPEC.md](AUTOMATION_REVIEWER_SPEC.md)** - Phase 3 review UI spec
- **[AI_TRAINING_CROP_AND_RANKING.md](AI_TRAINING_CROP_AND_RANKING.md)** - Technical training details
- **[CURRENT_TODO_LIST.md](CURRENT_TODO_LIST.md)** - Lines 251-490: Full automation pipeline

---

## 🎯 High-Level Phases

```
Phase 1: ✅ Data Collection       (COMPLETE)
Phase 2: ⏳ Train AI Models       (IN PROGRESS - This section)
Phase 3: ⏳ Build Review UI       (DOCUMENTED, Ready to build)
Phase 4: ⏳ Test in Sandbox       (Planned)
Phase 5: ⏳ Production Rollout    (Future)
```

---

## ✅ PHASE 1: Data Collection Infrastructure (COMPLETE)

### Status: ✅ COMPLETE - 21,250 Selections + 12,679 Crops from 15 Projects

**Selection Data (Web Image Selector):**
- [x] **21,250 total selection decisions** from 15 historical projects
- [x] Major projects: Mojo 1 (5,244), Mojo 2 (4,120), jmlimages-random (2,739), tattersail-0918 (2,610)
- [x] Additional projects: Eleni_raw, Kiara_Slender, 1100, 1101_hailey, 1011-1013, agent-1001-1003, Aiko_raw
- [x] File: `data/training/selection_only_log.csv`
- [x] Format: session_id, set_id, chosen_path, neg_paths, timestamp
- [x] **Anomaly cases identified:** 518 cases (2.4%) where lower stage chosen over higher

**Crop Data (Multi Crop Tool):**
- [x] **12,679 total crop decisions** across all projects
- [x] Average crop rate: 59.7% (varies from 0% to 99.1% by project)
- [x] File: `data/training/select_crop_log.csv`
- [x] Format: session_id, directory, chosen_path, crop_x1/y1/x2/y2, timestamp

**Embeddings:**
- [x] **77,304 image embeddings** computed using OpenCLIP
- [x] Stored in: `data/ai_data/embeddings/`
- [x] Cache: `data/ai_data/cache/processed_images.jsonl`

**Infrastructure:**
- [x] Training logger integrated in Web Image Selector
- [x] Training logger integrated in Multi Crop Tool
- [x] CSV format stable and documented
- [x] Historical data extraction script: `scripts/ai/extract_project_training.py`
- [x] Anomaly analysis tool: `scripts/ai/analyze_anomalies.py`

**Key Insights:**
- ✅ AI generation quality improving over time (newer projects have fewer anomalies)
- ✅ Project 1101_hailey has highest anomaly rate (58.7%) - excellent for learning quality judgment
- ✅ Mojo1/Mojo2 have very low anomaly rates (0.5-0.6%) - mostly picked highest stage

**📁 Complete Details:** See `Documents/AI_PHASE2_STATUS.md` for full breakdown by project

---

## ⏳ PHASE 2: Train AI Models

**Status:** 🚧 85% COMPLETE - Data extraction done, ranker v2 trained, v3 blocked on path fix
**Estimated Time:** 8-12 hours total (10 hours elapsed)
**Dependencies:** ✅ Python 3.11+, PyTorch, Apple Silicon (M4 Pro)

### 2.1: Environment Setup (~30 min)

- [x] ✅ Activate Python 3.11 venv
- [x] ✅ Install PyTorch with MPS (Metal) support
- [x] ✅ Install training utilities (lightning, accelerate)
- [x] ✅ Install ML models (open_clip_torch, opencv, pillow, etc.)
- [x] ✅ Verify Apple GPU works (MPS available: True)

**Deliverable:** ✅ Working Python environment with all dependencies  
**Testing:** ✅ MPS available = True

---

### 2.2: Feature Extraction (~4 hours total)

**Purpose:** Convert images to features the AI can learn from

#### 2.2.1: CLIP Embeddings (Semantic Understanding)
- [x] ✅ **COMPLETE** - Created script: `scripts/ai/compute_embeddings.py`
- [x] ✅ Load OpenCLIP model (ViT-B-32)
- [x] ✅ Process ALL images from 15 historical projects:
  - [x] ✅ **77,304 total embeddings** computed successfully
  - [x] ✅ Extract 512-dim embeddings per image
  - [x] ✅ Save to `data/ai_data/embeddings/{hash}.npy`
  - [x] ✅ Cache metadata in `data/ai_data/cache/processed_images.jsonl`
- [x] ✅ Covered projects: Mojo1, Mojo2, tattersail, jmlimages-random, 1100, 1101_hailey, 1011-1013, agent-1001-1003, Aiko, Eleni, Kiara
- [x] ✅ Test: Load random embedding and verify shape (512,)

**Actual Time:** ~4 hours across multiple extraction sessions  
**Output:** `data/ai_data/embeddings/*.npy` (77,304 embeddings total)
**Status:** ✅ COMPLETE (99.9%+ success rate)

---

### 2.3: Historical Data Extraction (~2-3 hours)

**Purpose:** Extract training decisions from all past projects

- [x] ✅ Created generic extraction script: `scripts/ai/extract_project_training.py`
- [x] ✅ Extracted from 15 projects:
  - [x] ✅ **Selection data:** 21,250 total decisions
  - [x] ✅ **Crop data:** 12,679 total crop decisions
- [x] ✅ Validated project boundaries (no cross-project contamination)
- [x] ✅ Created anomaly analysis: `scripts/ai/analyze_anomalies.py`
  - [x] ✅ Identified 518 anomaly cases (2.4%)
  - [x] ✅ Exported to `data/training/anomaly_cases.csv`
  - [x] ✅ Created statistics by project
- [x] ✅ Documented critical training rules in 3 documents

**Key Insights:**
- ✅ 21,250 selections >> 10k minimum needed for robust training
- ✅ 518 anomalies (2.4%) where lower stage chosen - critical for quality learning
- ✅ Anomaly rate decreases in newer projects (AI generation improving)
- ✅ Project 1101_hailey: 58.7% anomaly rate (highest quality bar)

**Deliverables:**
- ✅ `data/training/selection_only_log.csv` (21,250 selections)
- ✅ `data/training/select_crop_log.csv` (12,679 crops)  
- ✅ `data/training/anomaly_cases.csv` (518 anomalies)
- ✅ `data/training/anomaly_summary.json` (statistics)
- ✅ `Documents/AI_PHASE2_STATUS.md` (comprehensive summary)

**Status:** ✅ COMPLETE

---

### 2.4: Train Ranking Model (~2-3 hours)

**Purpose:** Learn which images you prefer, especially nuanced quality judgments

#### 2.4.1: Ranker v2 (Partial Dataset)
- [x] ✅ Created script: `scripts/ai/train_ranker_v2.py`
- [x] ✅ Implemented anomaly oversampling (10x weight)
- [x] ✅ Added project boundary validation
- [x] ✅ Trained successfully on ~10k examples (Mojo1 + Mojo2)
- [x] ✅ Used WeightedMarginRankingLoss with focal loss
- [x] ✅ Separate validation tracking for normal vs anomaly accuracy

**Status:** ✅ COMPLETE  
**Model:** `data/ai_data/models/ranker_v2.pt` + metadata
**Limitation:** Only used subset of available data

#### 2.4.2: Ranker v3 (Full Dataset) 
- [x] ✅ Created script: `scripts/ai/train_ranker_v3.py`
- [x] ✅ Designed to use full 21,250 selections
- [x] ✅ Configurable anomaly weights (5x, 10x, 20x)
- [x] ✅ Proper train/val split strategy
- [⚠️] ⚠️ **BLOCKED:** Path normalization issue
  - Issue: Embeddings computed for `training data/mojo1/...` paths
  - CSV has `/Users/.../Eros Mate/mojo1/...` paths  
  - No matches found → "No training data loaded" error
- [ ] Pending: Fix path matching logic
- [ ] Pending: Train on full dataset with 10x anomaly weight
- [ ] Pending: Evaluate on held-out anomaly validation set

**Status:** ⚠️ BLOCKED - Need to fix path normalization  
**Next Action:** Either fix v3 script OR recompute embeddings for actual mojo1/mojo2 dirs  
**Target Model:** `data/ai_data/models/ranker_v3_w10.pt`

---

### 2.5: Train Crop Proposer (~1-2 hours)

**Purpose:** Learn where you crop images

- [ ] Create script: `scripts/ai/train_crop_proposer_v2.py`
- [ ] Load crop data (12,679 examples from all projects)
- [ ] Define model architecture
  - [ ] Input: 514-dim (512 CLIP + 2 image dimensions)
  - [ ] Architecture: MLP (514→512→256→128→4)
  - [ ] Output: 4 values (x1, y1, x2, y2) normalized [0,1]
  - [ ] Loss: MSE or IoU loss
- [ ] Train on Apple GPU (MPS)
  - [ ] Split: 85% train / 15% val
  - [ ] Epochs: 50-100
- [ ] Evaluate metrics
  - [ ] IoU (Intersection over Union) on validation set
  - [ ] Visual inspection of proposals
- [ ] Save model and metadata

**Status:** ⏳ PENDING - Waiting for ranker v3 completion  
**Training Data:** 12,679 crop examples (robust dataset)  
**Target Model:** `data/ai_data/models/crop_proposer_v2.pt`
**Success Criteria:** IoU > 0.7 on validation set

---

### 2.6: Integration Testing (~30 min)

- [ ] Create inference script: `scripts/ai/test_models.py`
- [ ] Test end-to-end pipeline
  - [ ] Load both models (ranker v3 + crop proposer v2)
  - [ ] Test on 20 random groups from held-out validation set
  - [ ] For each group:
    - [ ] Rank images
    - [ ] Propose crop for winner
    - [ ] Display results with confidence scores
- [ ] Manual validation
  - [ ] Do rankings match your intuition?
  - [ ] Do crop proposals look reasonable?
  - [ ] Special focus: Does it handle anomaly cases well?
- [ ] Document results
  - [ ] Save test results to `data/ai_data/logs/phase2_testing.jsonl`
  - [ ] Note any systematic errors or biases
  - [ ] Compare v2 vs v3 ranker performance

**Status:** ⏳ PENDING  
**Success Criteria:** 
- Ranking: 75%+ accuracy on anomalies, 90%+ on normal cases
- Cropping: IoU > 0.7, visual inspection passes

---

### 2.7: Phase 2 Completion (~30 min)

- [ ] Update all documentation with final results
- [ ] Create Phase 2 summary report
  - [ ] Training metrics and performance
  - [ ] Anomaly analysis insights
  - [ ] Recommendations for Phase 3
  - [ ] Known limitations and future improvements
- [ ] Clean up temporary files
- [ ] Mark Phase 2 complete in project plan

**Status:** ⏳ PENDING  
**Deliverable:** Phase 2 completion report

---

## PHASE 2 SUMMARY (In Progress)

**Completed:**
✅ Environment setup (Python 3.11 + PyTorch + MPS)  
✅ 77,304 embeddings computed across 15 projects  
✅ 21,250 selections + 12,679 crops extracted  
✅ 518 anomaly cases identified and analyzed  
✅ Project boundary rules documented  
✅ Ranker v2 trained (10k examples)  
✅ Comprehensive documentation created

**In Progress:**
⚠️ Ranker v3 path normalization fix  
⏳ Ranker v3 training on full dataset

**Pending:**
⏳ Crop proposer v2 training  
⏳ Integration testing  
⏳ Phase 2 completion report

**Estimated Completion:** 2-4 more hours (90% done)

---

### 2.6: Documentation Updates

- [ ] Update `scripts/ai/README.md` with new scripts
- [ ] Document model architectures and training hyperparameters
- [ ] Document inference usage
- [ ] Update `AI_TRAINING_PHASE2_QUICKSTART.md` with actual results
- [ ] Mark Phase 2 as COMPLETE in this document

---

## ⏳ PHASE 3: Build Automation Reviewer UI

**Status:** 📋 Fully Spec'd in AUTOMATION_REVIEWER_SPEC.md  
**Estimated Time:** 8-12 hours  
**Dependencies:** Phase 2 models trained

### 3.1: Backend Development (~3-4 hours)

- [ ] Create Flask app: `scripts/07_automation_reviewer.py`
- [ ] Implement data loading
  - [ ] Load `sandbox/automation_decisions.jsonl`
  - [ ] Parse group data, AI choices, crop proposals
  - [ ] Track review state (approved/rejected/overridden/skipped)
- [ ] Implement API endpoints
  - [ ] `GET /api/automation/load_batch` - Get next batch of groups
  - [ ] `POST /api/automation/submit_decision` - Save user decision
  - [ ] `POST /api/automation/submit_batch` - Finalize batch
  - [ ] `GET /api/automation/progress` - Get review progress
- [ ] Implement decision output
  - [ ] Write approved decisions to `sandbox/approved_decisions.jsonl`
  - [ ] Include: group_id, action, final_choice, crop_coords, timestamp
- [ ] Add safety checks
  - [ ] Verify all paths are in `sandbox/` only
  - [ ] No file moves in this tool (review only)
  - [ ] Validate all user inputs

**Deliverable:** Working Flask backend  
**Testing:** Can load decisions, track state, write output

---

### 3.2: Frontend Development (~4-5 hours)

- [ ] Base on Web Image Selector HTML/CSS/JS
- [ ] Implement group display
  - [ ] Show all images in group (thumbnails)
  - [ ] Highlight AI's choice (green border + "AI Pick" badge)
  - [ ] Display metadata (stage, confidence)
- [ ] Implement crop overlay
  - [ ] HTML5 Canvas or SVG for crop rectangle
  - [ ] Dotted green line for proposed crop
  - [ ] Toggle visibility with 'C' key
  - [ ] Show crop coordinates on hover
- [ ] Implement decision panel
  - [ ] Show AI's reasoning (Step 1: selection, Step 2: crop)
  - [ ] Display confidence scores
  - [ ] Show any anomaly flags
- [ ] Implement action buttons
  - [ ] [A] Approve - Accept AI decision
  - [ ] [R] Reject - Keep all images, no action
  - [ ] [S] Skip - Review later
  - [ ] [1/2/3/4] Override - Pick different image
- [ ] Implement keyboard shortcuts
  - [ ] Enter/↓ - Next group
  - [ ] ↑ - Previous group
  - [ ] Shift+Enter - Submit batch
  - [ ] All action shortcuts (A/R/S/1-4)
- [ ] Implement progress tracking
  - [ ] Show "Reviewed X/Y groups (Z%)"
  - [ ] Batch progress bar
  - [ ] Overall progress

**Deliverable:** Working web UI  
**Testing:** Can review groups, make decisions, navigate with keyboard

---

### 3.3: Integration & Testing (~1-2 hours)

- [ ] End-to-end test
  - [ ] Generate mock decisions file
  - [ ] Load in UI
  - [ ] Review 10 groups
  - [ ] Submit batch
  - [ ] Verify output file format
- [ ] Test keyboard shortcuts
- [ ] Test crop overlay rendering
- [ ] Test state persistence (reload page mid-batch)
- [ ] Test error handling (missing files, invalid data)

**Success Criteria:** Can review 50 groups without bugs

---

### 3.4: Documentation

- [ ] Update `AUTOMATION_REVIEWER_SPEC.md` with implementation notes
- [ ] Document usage in README
- [ ] Document keyboard shortcuts
- [ ] Add screenshots to docs (optional)
- [ ] Mark Phase 3 as COMPLETE

---

## ⏳ PHASE 4: Automation Decision Pipeline

**Status:** 📋 Planned  
**Estimated Time:** 4-6 hours  
**Dependencies:** Phase 2 models trained

### 4.1: Build Decision Generator (~2-3 hours)

- [ ] Create script: `scripts/ai/generate_automation_decisions.py`
- [ ] Implement grouping logic
  - [ ] pHash near-duplicate detection
  - [ ] CLIP semantic similarity
  - [ ] Group 2-4 similar images
- [ ] Implement selection logic
  - [ ] Load ranker model
  - [ ] Score all images in group
  - [ ] Pick highest score as winner
  - [ ] Record confidence and reasoning
- [ ] Implement crop logic
  - [ ] Load crop proposer model
  - [ ] Generate crop proposal for winner
  - [ ] Decide if crop is needed (threshold on crop amount)
  - [ ] Record confidence and reasoning
- [ ] Output decisions
  - [ ] Write to `sandbox/automation_decisions.jsonl`
  - [ ] Format per spec (group_id, images, step1_selection, step2_crop)
  - [ ] **NO FILE MOVES** - decision marking only
- [ ] Add safety checks
  - [ ] Sandbox only
  - [ ] No file modifications
  - [ ] Log all decisions

**Deliverable:** `scripts/ai/generate_automation_decisions.py`  
**Testing:** Generate decisions for 100 sandbox groups

---

### 4.2: Build Commit Script (~1-2 hours)

- [ ] Create script: `scripts/tools/apply_automation_decisions.py`
- [ ] Implement decision loader
  - [ ] Read `sandbox/approved_decisions.jsonl`
  - [ ] Parse approved actions only
- [ ] Implement file operations
  - [ ] Move selected image (with companions)
  - [ ] Move others to review directory if needed
  - [ ] Execute crops if approved
  - [ ] Log all operations via FileTracker
- [ ] Add safety features
  - [ ] Dry-run mode (default)
  - [ ] Sandbox-only enforcement
  - [ ] Confirmation prompt before execution
  - [ ] Companion file integrity checks
- [ ] Error handling
  - [ ] Rollback on errors
  - [ ] Detailed error logging
  - [ ] Report success/failure summary

**Deliverable:** `scripts/tools/apply_automation_decisions.py`  
**Testing:** Test on 10 approved decisions in sandbox (dry-run mode)

---

### 4.3: Documentation

- [ ] Document full pipeline flow
- [ ] Document decision file formats
- [ ] Document safety guarantees
- [ ] Update CURRENT_TODO_LIST.md
- [ ] Mark Phase 4 as COMPLETE

---

## ⏳ PHASE 5: Sandbox Testing & Iteration

**Status:** 📋 Planned  
**Estimated Time:** 1-2 weeks  
**Dependencies:** Phases 2, 3, 4 complete

### 5.1: Initial Sandbox Test (~2-3 days)

- [ ] Prepare sandbox environment
  - [ ] Copy 100-200 groups from mojo2 to sandbox
  - [ ] Verify copies complete with companions
- [ ] Run full pipeline
  - [ ] Generate decisions
  - [ ] Review in UI
  - [ ] Apply approved decisions
- [ ] Measure metrics
  - [ ] Agreement rate (approved vs overridden)
  - [ ] Crop proposal acceptance rate
  - [ ] Time saved estimate
  - [ ] Bug count
- [ ] Collect feedback
  - [ ] What worked well?
  - [ ] What needs improvement?
  - [ ] Any systematic errors?

**Deliverable:** Initial metrics report  
**Success Criteria:** >60% approval rate, no data loss bugs

---

### 5.2: Model Tuning & Iteration (~3-5 days)

- [ ] Analyze overrides
  - [ ] Why were AI picks overridden?
  - [ ] Patterns in rejected crop proposals?
- [ ] Improve models
  - [ ] Add override data as training examples
  - [ ] Retrain with new data
  - [ ] Test improved models
- [ ] Tune thresholds
  - [ ] Confidence thresholds for auto-approval
  - [ ] Crop amount thresholds
  - [ ] Grouping similarity thresholds
- [ ] Re-test in sandbox
- [ ] Repeat until metrics stabilize

**Success Criteria:** >80% approval rate, <5% overrides

---

### 5.3: Large-Scale Sandbox Test (~2-3 days)

- [ ] Process 1000+ groups in sandbox
- [ ] Measure final metrics
- [ ] Validate no data loss
- [ ] Validate companion file integrity
- [ ] Time savings analysis
- [ ] Bug report (goal: zero critical bugs)

**Success Criteria:** >90% approval rate, zero critical bugs

---

### 5.4: Production Readiness Assessment

- [ ] Review all metrics
- [ ] Review all documentation
- [ ] Review all safety checks
- [ ] Team decision: Ready for production pilot?
- [ ] Document decision and reasoning

---

## ⏳ PHASE 6: Production Pilot (Future)

**Status:** Not Yet Planned  
**Prerequisites:** Phase 5 complete with >90% approval rate

- [ ] Define production pilot scope
- [ ] Create production safety plan
- [ ] Execute pilot
- [ ] Evaluate results
- [ ] Decision: Full rollout or more iteration

---

## 📊 Success Metrics (By Phase)

### Phase 2: Model Training
- **Target:** Top-1 accuracy >70%, Crop IoU >0.7
- **Actual:** _TBD_

### Phase 3: Review UI
- **Target:** Can review 50 groups without bugs
- **Actual:** _TBD_

### Phase 4: Pipeline
- **Target:** Generate 100 decisions, execute 10 approved moves (dry-run)
- **Actual:** _TBD_

### Phase 5: Sandbox Testing
- **Target:** >90% approval rate, zero critical bugs, 50%+ time savings
- **Actual:** _TBD_

---

## 🔄 Continuous Improvement Loop

After each phase:
1. **Update this document** with actual results
2. **Update phase status** (⏳ → ✅)
3. **Document lessons learned**
4. **Update TODO list** in CURRENT_TODO_LIST.md
5. **Update related specs** if implementation diverged

---

## 📝 Notes & Lessons Learned

### Phase 1 (Complete)
- ✅ Successfully captured 17k examples from Mojo 1 + Mojo 2
- ✅ CSV format stable and easy to work with
- ✅ Both Web Image Selector and Multi Crop Tool logging correctly

### Phase 2 (In Progress)
- _Add notes as work progresses_

### Phase 3 (Not Started)
- _Add notes as work progresses_

---

## 🚀 Next Actions

**Immediate (This Week):**
1. [ ] Decide: Start Phase 2 now or wait?
2. [ ] If starting: Set up Python environment (Phase 2.1)
3. [ ] If waiting: Continue other project work

**Short-term (Next 2 Weeks):**
1. [ ] Complete Phase 2 (model training)
2. [ ] Begin Phase 3 (review UI)

**Medium-term (Next Month):**
1. [ ] Complete Phase 3 (review UI)
2. [ ] Complete Phase 4 (pipeline)
3. [ ] Begin Phase 5 (sandbox testing)

---

**Last Updated:** October 20, 2025  
**Next Review:** When Phase 2 begins or 1 week, whichever comes first

