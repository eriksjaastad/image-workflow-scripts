# Current TODO List

**Last Updated:** October 5, 2025

## ⚠️ **IMPORTANT WORKFLOW RULE**
**ONE TODO ITEM AT A TIME** - Complete one task, check in with Erik, get approval before moving to the next item. Never complete multiple TODO items in sequence without user input. This prevents issues and ensures quality control.

---

## 🔥 **HIGH PRIORITY**

### **Create Mojo1 Project File (Start/Finish Tracking) — Immediate**
- **Create manifest**: `data/projects/mojo1.project.json` with fields: projectId, status, createdAt, startedAt, finishedAt, steps[], metrics, counts, removeFileOnFinish.
- **Purpose**: Enables end-to-end images/hour KPI and per-step breakdown now, even mid-project.
- **Finish protocol**: On completion, compute metrics, archive summary, and delete or archive the project file.
- **Next**: Later hook into FileTracker to auto-update steps and finish.

### **Project Throughput Metric (Top Priority) — Images/Hour End-to-End**
- **Goal:** Single-project throughput: total images processed ÷ total project time, from batch arrival to final completion.
- **Define project boundary (planning):** What constitutes a project (e.g., a dated batch folder or manifest) and its ID convention.
- **Capture Start event (planning):** Add an explicit “project_start” marker (CLI or small script) that records timestamp, project_id, initial counts.
- **Capture Finish event (planning):** Add “project_finish” marker when last required stage completes; store timestamp and final counts.
- **Per-process breakdown (planning):** Use existing `FileTracker` logs + step tags (selector, sorter, crop, dedupe, etc.) to compute time per step and items/hour per step.
- **Storage (planning):** Persist project session records under `data/timer_data/projects/PROJECT_ID.json` (or consolidate with existing timer data) and/or append to `data/daily_summaries/`.
- **Aggregator (planning):** Build `scripts/dashboard/project_metrics_aggregator.py` to compute: end-to-end images/hour, step-level rates, ahead/behind vs baseline.
- **Dashboard (planning):** Add a top-level KPI card “Images/hour (project)” with live session progress (on track / behind / ahead), sparkline over time, and breakdown table by step.
- **Baseline & tracking (planning):** Establish baseline from recent N projects; flag deviations >±10%.
- **Safety:** Planning-only; no destructive operations. Implement start/finish markers and aggregator first; dashboard UI follows.

### **Historical Throughput Backfill from Timesheets (Planning + Partial)**
- **Goal:** Build historical images/hour baseline from Erik's timesheets to compare current/future projects.
- **Input doc:** Incoming blueprint (timesheet-to-history). Reference when available.
- **Schema (planning):** `project_id`, `date`, `task/step`, `start_time`, `end_time`, `images_processed(optional)`, `notes`.
- **Mapping (planning):** Map timesheet categories to pipeline steps (selector, sorter, crop, dedupe, review).
- **Converter (planning):** `scripts/tools/import_timesheet_history.py` to ingest CSV/JSON and emit project-level aggregates under `data/timer_data/history/`.
- **Baselines (planning):** Compute per-project and rolling baseline images/hour; store for dashboard.
- **Mojo1 partial now:** If starting numbers are available, update `mojo1/Mojo1.project.json` counts.initialImages and, if applicable, adjust `startedAt`.
- **Dashboard integration (planning):** Extend aggregator to include historical comparisons (ahead/behind vs historical baseline).

### **Sandbox Automation Orchestrator (Tomorrow)**
- **Create sandbox copy**: copy remaining images from `mojo1/` into `sandbox/` (read-only source, safe target)
- **Run thinning steps**: pHash near-dup → FAISS semantic → timestamp-window clusters (reports + `_review_dupes/` staging only)
- **Grading metrics**: counts kept/flagged, estimated review saved, thresholds summary
- **Multi-source grouping**: extend similar-image grouper to accept `selected/` + `crop/` in one pass
- **Sidecar visual cues**: generate `.cropped` sidecars and add badge in Character Sorter
- **One-shot script**: single CLI to run all steps with dry-run/stage modes

### **Sidecar crop flag + UI badge + multi-source grouping**
- **Add sidecar flag**: create `.cropped` sidecar files next to PNGs in `crop/` (same-stem). No image changes; travels with file during moves.
- **Character sorter badge**: detect `.cropped` companions and render a small “Cropped” badge on thumbnails.
- **Similar-image grouper support**: add optional multi-source input (e.g., `selected/` + `crop/`) and preserve/propagate sidecars.
- **CLI helper**: simple script/command to generate sidecars for existing `crop/` PNGs.
- **Safety**: read-only marker; never modifies image bytes.

### **Quick UI wins – Web Image Selector header**
- **Add separate counters**: show "Kept approved" vs "Kept to crop" in the header summary
- **Add progress bar**: small inline bar for remaining files (current batch + overall)
- **Accuracy rules**: file-count based; default-delete for untouched; per-image-skip treated consistently with server
- **Style**: follow `Documents/WEB_STYLE_GUIDE.md` (colors, spacing, typography)
- **Scope**: UI only; no behavioral changes to move/delete
- **Why**: faster feedback during long sessions; reduces mental load during tedious passes

### **Cross-timestamp duplicate set pruning (safe staging)**
- **Goal**: detect visually identical sets with same stage lineup (1, 1.5, 2, 3) but different timestamps; keep one set, stage others for review.
- **Approach**: per-stage CLIP embedding similarity within candidate clusters; require same stage names across compared sets; cross-check YAML similarity.
- **Pipeline**: build candidate pairs via similar-image grouper; verify stage-by-stage match; produce report.
- **Action**: move redundant whole-set (image + all companions) into `delete_review/` (not Trash) for manual confirmation.
- **Modes**: dry-run report, conservative thresholds, manual approve step.

---

## 🧠 **Automation Pipeline Plan (Planning Only)**

> Goal: Reduce Erik's manual review time by auto-picking winners in 2–4 near-duplicate sets and staging non-keepers safely for quick review. AI photography issues (blur, exposure) are irrelevant; we focus on AI anomalies (hands/feet, missing belly button, extra fingers, etc.).

### **A. Objectives & Success Criteria**
- **Primary objective**: Auto-select a best image per near-duplicate set with high precision; present only ambiguous sets for review.
- **Safety**: Non-destructive by default; move non-keepers to review folders with companions; never alter image bytes.
- **Companion integrity**: Always move image + `.yaml`/`.caption` and any same-stem companions together.
- **Measurable success**:
  - Manual time reduction ≥ 50% on duplicate thinning passes.
  - Auto-pick agreement with Erik’s choice ≥ 98% on calibrated sets.
  - False-positive staging (wrongly moved a keeper) ≤ 1 per 1,000 images.

### **B. What Tools Do What (Human vs Automation)**
- **Human-in-the-loop tools (existing/updated)**
  - `scripts/01_web_image_selector.py`: stays primary for version selection; may receive a “AI pick” hint later (optional).
  - `scripts/02_web_character_sorter.py`: add optional badges (e.g., “cropped” sidecar indicator already planned) and a future “AI suspect” tag.
  - New: `scripts/07_auto_pick_review.py` (planning only): web UI to review auto-pick reports, show chosen vs staged, approve/override by group.
  - `scripts/06_web_duplicate_finder.py`: remains useful for visual comparisons on tough clusters.
- **Automation scripts (offline, reversible)**
  - `scripts/tools/compute_phash.py`: compute pHash; write to `data/ai_data/hashes/`.
  - `scripts/tools/compute_embeddings.py`: OpenCLIP ViT-B/32; write to `data/ai_data/embeddings/`.
  - `scripts/tools/detect_hands_feet.py`: MediaPipe Hands (+ optional Pose feet presence); output per-image presence/anomaly hints to `data/ai_data/hands/`.
  - `scripts/utils/phash_group_near_dupes.py`: group pairs/sets by Hamming distance (≤ 8–10).
  - `scripts/utils/clip_group_near_dupes.py`: group pairs/sets by CLIP cosine (≥ 0.95; tunable).
  - `scripts/utils/auto_pick_near_dupes.py`: merge groupers, apply stage-aware ranking + anomaly gates, generate report; optional staging to `_review_dupes/`.
  - `scripts/orchestrators/automation_pipeline.py`: one CLI to run the pipeline in dry-run/stage modes, with metrics summary.

### **C. Selection Logic (Stage-aware + Anomaly-aware)**
- Candidate groups = filename stage runs + pHash near-exacts + CLIP semantic near-dupes.
- Rank within group:
  1) Highest stage number wins if it passes anomaly gate.
  2) If tie, prefer fewer detected hands/feet in-frame; if present, require anomaly checks to pass.
  3) Break ties by larger resolution, then latest mtime.
- Anomaly gate (AI-specific):
  - Hands/feet presence signal (fast): presence = higher risk; down-rank unless scores are strong.
  - Simple hand heuristics from keypoints: flag likely extra/fused fingers if geometry inconsistent (see doc below).
  - Optional “missing belly button” is future work; keep placeholder tag.

References: `Documents/hand_foot_anomaly_scripts.md` (approaches and two‑stage pipeline ideas).

### **D. Thresholds (initial; calibrate on sample)**
- pHash Hamming: near-exact ≤ 8; lenient ≤ 10.
- CLIP cosine: ≥ 0.95 to group; raise to 0.97 if too broad.
- Ambiguity margin: if top two scores within 0.05 → send to human review.
- Hands/feet presence: treat as risk booster; always route borderline groups with hands/feet for review.

### **E. Milestones & Tasks (Planning → Calibration → Tools → Review UI)**
1) Planning & calibration set (no code changes)
   - Define 100–200 image calibration subset across multiple characters and stages.
   - Enumerate anomaly tags we care about first (hands/feet, belly button missing, facial stacking).
   - Decide reporting schema fields for auto-pick report (group_id, chosen, losers, scores, reasons).
2) Offline feature tools (standalone; safe)
   - Build: `compute_phash.py`, `compute_embeddings.py`, `detect_hands_feet.py`.
   - Run on calibration subset; store outputs under `data/ai_data/`.
3) Grouping + auto-pick (reports only)
   - Build: `phash_group_near_dupes.py`, `clip_group_near_dupes.py`, `auto_pick_near_dupes.py` (report only).
   - Validate groups and picks on calibration subset; tune thresholds.
4) Staging mode (non-keepers → `_review_dupes/`)
   - Extend `auto_pick_near_dupes.py` with `--stage` flag; default is dry-run.
   - Ensure companion handling via `move_file_with_all_companions`.
   - Metrics: counts kept/staged/ambiguous; false-positive sampling plan.
5) Review UI (human approval loop)
   - Plan `07_auto_pick_review.py` for tabular review of auto-pick decisions with thumbnails and approve/override.
   - Keep it optional; start with CSV/Markdown report + `06_web_duplicate_finder.py` for hard cases.
6) Orchestrator & metrics
   - `automation_pipeline.py` to chain steps and emit a summary (precision/recall estimates on calibration set, time saved).

### **F. Testing & Evaluation (before any wide run)**
- Golden fixtures for grouping/selection rules (small fixture dirs committed to `scripts/tests/fixtures/`).
- Dry-run as default; CI-style check that no file moves occur without `--stage`.
- Companion integrity tests (image + sidecars preserved through moves).
- Calibration evaluation: report agreement with Erik’s picks on the subset; adjust thresholds until ≥98%.

### **G. Risks & Safeguards**
- CLIP over-grouping across pose changes → mitigate with higher cosine threshold + stage-name parity.
- Hands detection false positives in stylized cases → treat presence as a review trigger, not an auto-reject.
- Always stage to review, never hard-delete; centralized logging via FileTracker.

### **H. Deliverables**
- Auto-pick report (CSV/Markdown) with reasons per decision.
- `_review_dupes/` staging structure mirroring sources (dry-run off).
- Optional review UI plan + wireframe before implementation.

### **I. References**
- `Documents/hand_foot_anomaly_scripts.md`
- `image_batch_culling_pipeline.md`, `similar_image_dedup_automation_plan.md`, `stage_aware_image_culling_workflow_v2.md`

> Note: This section is planning-only. No scripts will be created until we review and approve this plan.

### **J. Portrait Fast‑Track via MediaPipe Pose (Planning + Experiment)**
- **Hypothesis**: Images that are clearly head‑and‑shoulders portraits rarely need cropping; selecting the highest stage is typically sufficient.
- **Goal**: Automatically identify portrait (head+shoulders) images and route them into a fast lane, reducing intermixing with images that need crop decisions.
- **Detection approach (planning)**:
  - MediaPipe Pose + Face for head box + shoulder keypoints; simple heuristics: face area fraction, shoulder span band, hips mostly out of frame.
  - Fallback: face‑area‑only heuristic for speed; escalate ambiguous cases to full Pose.
- **Proposed pipeline (planning)**:
  1) Offline classify portrait vs non‑portrait (dry‑run → CSV of flags).
  2) Option A: Pre‑stage portraits into a `portraits_fastpass/` review dir.
  3) Option B: Tag portraits and surface a “Portrait fast pass” filter in the web selector.
  4) In fast pass, default selection rule = highest stage that passes anomaly gate; do not send to `crop/`.
- **Calibration plan (planning)**: Build a 100–200 image labeled subset to tune thresholds and confirm ≥98% agreement with Erik on portrait identification.
- **Safety**: Planning‑only; if approved, first build as dry‑run report with zero file moves.

### **K. Workflow Concepts — Pros/Cons and Throughput Modeling (Planning)**
- Compare workflows to maximize “quick wins” first without harming overall throughput:
  - Portrait fast‑pass first vs after near‑dupe thinning.
  - Stage‑aware auto‑pick before/after character sorting.
  - Routing by hands/feet presence early vs late.
- Define evaluation metrics: items/hour, percent auto‑finalized, review queue size, context‑switch cost.
- Run small A/B sessions and log timings to confirm which sequence actually yields faster end‑to‑end results (avoid “feels fast but is slower”).

## 🤖 **AI Training - Comprehensive Implementation Plan**

> 📖 **Master Reference:** `Documents/image_cropping_ranking_training_plan.md`  
> Full technical details, architecture, and Apple Silicon optimization strategies

### ✅ **Current Status: Basic Logging Complete**

**What We've Built:**
- ✅ Training logging built into 4 tools (no flags needed!)
- ✅ Web image selector → `data/training/selection_only_log.csv`
- ✅ Desktop image selector crop → `data/training/select_crop_log.csv`
- ✅ Multi-crop tool → `data/training/select_crop_log.csv`
- ✅ Web duplicate finder → `data/training/duplicate_detection_log.csv`

**Migration Path:**
- ✅ CSV logging provides foundation
- 🔄 Need to upgrade to JSONL schema with anomaly tags (see master doc)
- 🔄 Add `set_id`, `anomaly_tags`, detailed metadata
- 💡 Keep CSV logs running while we build JSONL alongside

---

## 📊 **CURRENT STATUS: Environment Ready, Week 1 Next**

**✅ Completed Today (Oct 5):** MPS + AI packages installed, all tests passing  
**🔜 Next Step:** Create `data/ai_data/` directory structure (Step 1)  
**⏸️ Paused:** Waiting to begin Week 1 implementation

---

## 🎯 **Training Philosophy: Apprentice Model**

The AI will learn like an apprentice who:
1. **Watches you work** (weeks/months) - learns patterns, no interference
2. **Makes suggestions** (when ready) - you accept/reject/modify to refine it  
3. **Works semi-autonomously** (distant future) - with your review

**Critical:** No automation until AI proves it understands your preferences through extensive observation.

---

## 📊 **PHASE 1: OBSERVATION MODE** (Current Focus - Weeks 1-3+)

**Goal:** Silent logging system that watches your existing workflow without changing anything.

### **Week 1: Foundation - Passive Data Collection**

**1.1 Set Up AI Data Structure** ✓ Safe, no breaking changes
```
data/ai_data/
  ├── embeddings/           # Image embeddings (computed offline)
  ├── hashes/              # Perceptual hashes
  ├── saliency/            # Saliency maps
  ├── hands/               # Hand detection data
  ├── observations/        # Your actual decisions (JSONL logs)
  │   └── sessions/
  └── models/              # Future trained models
```

**1.2 Create Offline Analysis Tools** ✓ Can run anytime, won't break workflow
- `tools/compute_embeddings.py` - Batch process images in `mojo/`, `crop/`, `selected/`
  - Model: OpenCLIP ViT-B/32 (best balance speed/accuracy for M4 Pro)
  - Output: `data/ai_data/embeddings/`
- `tools/compute_phash.py` - Generate perceptual hashes for duplicate detection
  - Output: `data/ai_data/hashes/`
- `tools/compute_hands_saliency.py` - Extract hand keypoints + saliency maps
  - MediaPipe Hands for keypoint detection
  - U²-Net via rembg for saliency (simplest implementation)
  - Output: `data/ai_data/hands/`, `data/ai_data/saliency/`
- **All run independently, no integration needed yet**

**1.3 Design Observation Schema** ✓ Planning only
```jsonl
{
  "timestamp": "2025-10-05T19:12:31Z",
  "session_id": "mojo_batch_001",
  "tool": "web_image_selector",
  "image_group": ["img1.png", "img2.png", "img3.png"],
  "action": "selected",
  "chosen": "img2.png",
  "rejected": ["img1.png", "img3.png"],
  "crop_box": null,  # or {x, y, w, h} if cropped
  "deleted": false,
  "notes": "optional metadata"
}
```

---

### **Week 2: Passive Integration - Watch Without Interfering**

**2.1 Add Silent Logging to Existing Tools** ⚠️ Needs careful integration
- Modify `01_web_image_selector.py` - log selections
- Modify `04_multi_crop_tool.py` - log crop decisions
- Modify `01_desktop_image_selector_crop.py` - log select+crop
- **Critical Requirements:**
  - ✅ Completely silent (no UI changes)
  - ✅ Optional (can disable with env var)
  - ✅ Zero performance impact
  - ✅ Fail-safe (errors don't break tools)
- **Feature Flag:** `export ENABLE_AI_LOGGING=true` to activate
- **Integration Strategy:** Add to ONE tool first, test thoroughly, then expand

**2.2 Build Observation Dashboard** ✓ Separate tool, safe
- Web viewer to see what AI has learned
- Show statistics: "Observed 1,247 decisions across 89 sessions"
- Visualize patterns without making recommendations
- Check data quality
- Location: `scripts/dashboard/ai_observation_viewer.py`

---

### **Week 3: Data Accumulation & Quality Check**

**3.1 Run Batch Feature Extraction**
- Process all images in `mojo/`, `crop/`, `selected/`
- Generate embeddings, hashes, saliency maps
- Build feature database
- Verify outputs for quality

**3.2 Validate Observation Data**
- Check logs are capturing decisions correctly
- Verify file paths resolve
- Ensure data quality for training
- Build validation script: `tools/validate_observations.py`

**3.3 Initial Pattern Analysis** (Optional)
- Simple statistics: "You prefer images with X"
- No recommendations, just showing what it sees
- Sanity check the AI is learning something meaningful
- Output: Human-readable report of patterns detected

---

## 🛠️ **Implementation Strategy - No Breaking Changes**

### **What Can We Do RIGHT NOW (Zero Risk)**

1. ✅ **Create `data/ai_data/` structure** - new directory, can't break anything
2. ✅ **Build standalone tools** (`compute_embeddings.py`, etc.) - run independently
3. ✅ **Write observation schema docs** - planning only
4. ✅ **MPS benchmark test** - verify M4 Pro performance with PyTorch

### **What Needs Careful Integration**

1. ⚠️ **Adding logging to existing tools** - need to test thoroughly
   - Add behind feature flag: `ENABLE_AI_LOGGING=true`
   - Extensive error handling
   - Test on small batch first
   - Integrate ONE tool at a time

### **What We Can Build Incrementally**

1. 🔄 **Observation dashboard** - separate web tool
2. 🔄 **Analysis scripts** - run offline on collected data
3. 🔄 **Model training** - when enough data accumulated

---

## 🗓️ **Implementation Roadmap (Week-by-Week)**

### **✅ COMPLETED: Environment Setup (Oct 5, 2025)**

**What We Built:**
- ✅ MPS benchmark test - verified M4 Pro + PyTorch working (1.76x speedup!)
- ✅ Installed AI packages into `.venv311`:
  - OpenCLIP 3.2.0 (CLIP embeddings)
  - imagehash 4.3.2 (perceptual hashing)
  - MediaPipe 0.10.21 (hand detection)
  - rembg 2.0.67 (saliency detection)
  - pandas 2.3.3 (data handling)
- ✅ Verified no conflicts with existing tools (163 tests passed)
- ✅ Web image selector and all utilities working perfectly

**NumPy Note:** Downgraded from 2.2.6 → 1.26.4 for MediaPipe compatibility. OpenCV and all existing scripts work fine with this version.

---

### **🔜 NEXT: Week 1 - Foundation (Passive Data Collection)**

**Priority Order (each item depends on previous):**

**1️⃣ Create AI Data Structure** ⚠️ **DO THIS FIRST**
- Create `/absolute/path/to/project-root/data/ai_data/` directory
- Subdirectories: `embeddings/`, `hashes/`, `saliency/`, `hands/`, `observations/sessions/`, `models/`
- Safe operation: just creates empty directories, can't break anything
- **Status:** Ready to implement

**2️⃣ Build Standalone Analysis Tools** ⚠️ **DO SECOND**
- Create `scripts/tools/compute_embeddings.py` (OpenCLIP ViT-B/32, use MPS)
- Create `scripts/tools/compute_phash.py` (perceptual hashing)
- Create `scripts/tools/compute_hands_saliency.py` (MediaPipe + U²-Net)
- All run independently on `mojo/`, `crop/`, `selected/` directories
- No integration with existing tools yet - completely safe
- **Status:** Ready to implement
- **Dependencies:** Need step 1 completed first (directories must exist)

**3️⃣ Test on Small Sample** ⚠️ **DO THIRD**
- Run tools on small sample (e.g., 10-20 images from character_group_2/)
- Verify outputs are correct
- Check file sizes and formats
- **Status:** Ready after steps 1-2
- **Dependencies:** Steps 1-2 must work first

**4️⃣ Design & Document Observation Schema**
- Finalize JSONL schema for logging
- Document anomaly tags
- Create example session files
- **Status:** Ready to implement (documentation only)
- **Dependencies:** None (can do anytime)

---

### **🔮 FUTURE: Week 2 - Passive Integration (Watch Mode)**

**NOT STARTED - Don't begin until Week 1 complete**

**5️⃣ Add Silent Logging to ONE Tool**
- Integrate observation logging into `01_web_image_selector.py` first
- Always-on by default (learns from everything you do)
- Emergency kill switch: `export DISABLE_AI_LOGGING=true`
- Extensive testing on small batch
- **Status:** Not started
- **Dependencies:** Must complete Week 1 first + verify schema works

**6️⃣ Build Observation Dashboard**
- Create `scripts/dashboard/ai_observation_viewer.py`
- Show statistics: "Observed X decisions across Y sessions"
- Visualize patterns without making recommendations
- **Status:** Not started
- **Dependencies:** Need logging data from step 5

**7️⃣ Expand to Other Tools**
- Add logging to `04_multi_crop_tool.py`
- Add logging to `01_desktop_image_selector_crop.py`
- Only if step 5 works perfectly
- **Status:** Not started
- **Dependencies:** Step 5 must be rock-solid first

---

### **🔮 FUTURE: Week 3 - Data Accumulation & Validation**

**NOT STARTED - Don't begin until Week 2 complete**

**8️⃣ Run Batch Feature Extraction**
- Process all existing images in mojo/, crop/, selected/
- Generate complete feature database
- **Status:** Not started

**9️⃣ Validate Observation Data**
- Build `scripts/tools/validate_observations.py`
- Check data quality, file paths, completeness
- **Status:** Not started

**🔟 Initial Pattern Analysis**
- Simple statistics showing what AI learned
- No recommendations, just pattern visualization
- **Status:** Not started

---

## 📋 **Model & Technology Choices**

**Documented for future reference:**

- **OpenCLIP ViT-B/32**: Best balance of speed/accuracy for M4 Pro (512D embeddings, fast inference)
- **U²-Net via rembg**: Simplest saliency implementation (Apache 2.0 license)
- **MediaPipe Hands**: Standard hand keypoint detection (Apache 2.0 license)
- **pHash**: Perceptual hashing for near-duplicate detection (simple, fast)
- **Skip YOLO for now**: Add in Phase 2 if needed for face/foot detection

---

## 📊 **PHASE 2: RECOMMENDATION MODE** (Future - Weeks 4+)

**Won't start until Phase 1 data collection is satisfactory**

### **Week 4+: Optional Recommendation Layer**

**4.1 Build Ranking Model v0**
- Create: `train/rank_bt.py` (Bradley-Terry or tiny MLP)
- Features: CLIP embedding + sharpness + subject size + anomaly counts
- Train on accumulated observation data
- Use MPS for Apple Silicon acceleration

**4.2 Build Crop Proposer v0**
- Create: `tools/propose_crop.py` (fixed-aspect, objective-based)
- Constraints: Keep original aspect ratio, head preference, ≥1/3 body
- Score: `α·SaliencyIn – β·AnomalyOverlap – γ·HeadCut – δ·JointCut`
- Output: Proposed crops with confidence scores

**4.3 Add Toggle-able Recommendation UI**
- Add optional recommendation layer to tools (can toggle on/off)
- "AI suggests: img2.png (confidence: 87%)" - you agree/disagree
- Log your corrections to improve model
- **Critical:** Must be completely optional and easy to disable

**4.4 Correction Logging**
- Log when you accept AI suggestions
- Log when you reject and choose differently
- Log when you modify AI crop proposals
- Use corrections to refine models incrementally

---

## 📊 **PHASE 3: SEMI-AUTOMATION** (Distant Future)

**Won't start until Phase 2 shows sustained high accuracy**

### **Requirements for Phase 3 Activation:**
- **Metric goals:** Top-1 agreement ≥99%, Crop acceptance ≥99%, Residual anomalies ≈0
- **Promotion criteria:** Sustained for N consecutive sets (e.g., 300)
- **Manual override:** Easy kill switch to return to Phase 2 or Phase 1

### **Batch Processing Features:**
- Batch processing with review queue
- High confidence = auto-process
- Low confidence = ask you first
- **Safety:** AI-assist flags residual anomalies for your review

### **Active Learning & Routing**
- Route sets with low confidence margin first
- Prioritize: Small ranking margin, crop near threshold, detector disagreement
- Optional: Integrate Cleanlab/ActiveLab for label QA

### **Optional Enhancements**
- Add tiny YOLO face/foot detection
- Improve "don't cut here" penalties (face/foot boxes)
- Refine crop proposals based on accumulated data

---

### 🔧 **Supporting Tasks (Ongoing)**

**A. Data Monitoring & Validation**
- Create verification script: Check logs being written
- Alert if logs stale (>X days without updates)
- Validate CSV/JSONL structure and data quality
- Daily/weekly automated checks

**B. Documentation**
- Document logging schemas in Technical Knowledge Base
- Create training data inventory
- Track model versions and approval rates

**C. Environment Setup**
- Python ≥3.11, PyTorch with MPS (Metal)
- Install: `torch`, `lightning`, `accelerate`, `open_clip_torch`, `opencv-python`, `imagehash`, `mediapipe`, `u2net`
- Verify Apple Silicon acceleration working

---

## 🔐 **Data Backup Plan**

**Goal:** Automated, reliable off-repo backups of CSV/log data.

**Scope:**
- `data/training/*.csv`
- `data/file_operations_logs/*.log`
- Optional: manifests with hashes/counts

**Tasks:**
1. Choose backend (S3, Backblaze B2, or Google Drive)
2. Create `scripts/backup/backup_training_data.py`
   - Package files into timestamped tar.gz
   - Write manifest with sha256 + row counts
   - Upload with lifecycle policy (keep 8 weekly)
3. Create `scripts/backup/restore_training_data.py`
4. Schedule weekly cron (Sun 02:00 local)
5. End-to-end test: backup → delete → restore → verify
6. Document in Knowledge Base

**Defaults:** Weekly backups, 8 weeks retention, server-side encryption

---

## 🔧 **MEDIUM PRIORITY**

### 1. Desktop Selector Selection Toggle Bug
**Issue:** When one image is selected by hotkey and then another image's crop position is modified, the original image doesn't toggle back to delete.
- **Expected:** Only ONE image should be kept at a time
- **Impact:** UX issue during long image processing sessions
- **File:** `scripts/01_desktop_image_selector_crop.py`
- **Note:** Lower priority - not using desktop selector for a few days

### 2. Improve Test Coverage
- Add unsorted input tests (verify pre-sorting requirement)
- Add unknown stage handling tests
- Add min_group_size variation tests
- Add two-runs-in-sequence tests
- Add more edge case tests for companion_file_utils.py (permission errors, malformed data)
- Add cleanup functionality tests for check_companions.py

### 3. Create Missing Test Files
**Need tests for:**
- `04_multi_crop_tool.py`
- `06_web_duplicate_finder.py`
- `utils/character_processor.py`
- `utils/duplicate_checker.py`
- `utils/recursive_file_mover.py`
- `utils/similarity_viewer.py`
- `utils/triplet_deduplicator.py`
- `utils/triplet_mover.py`

### 4. Companion file deletion parity (standardize deletes)
**Issue:** Some delete paths still handle only `.yaml`. We should always delete image + ALL companion files using centralized utilities.

**Files & examples:**
- `scripts/utils/base_desktop_image_tool.py` — YAML-only delete in base method used by inheritors.
  - Example:
    ```
    def safe_delete(self, png_path: Path, yaml_path: Path):
        send2trash(str(png_path))
        if yaml_path.exists():
            send2trash(str(yaml_path))
    ```
  - Impact: Tools calling base `safe_delete` won’t remove `.caption` sidecars.

- `scripts/04_multi_crop_tool.py` — Calls base `safe_delete(png_path, yaml_path)` during delete flow.
  - Example: `self.safe_delete(png_path, yaml_path)`
  - Note: Fixing base method fixes this tool automatically.

- `scripts/utils/triplet_deduplicator.py` — Dry-run and removal reference only `.yaml`.
  - Example dry-run:
    ```
    yaml_file = png_file.parent / f"{png_file.stem}.yaml"
    if yaml_file.exists():
        print(f"    🗑️  {yaml_file.name}")
    ```
  - Expected: Use wildcard companions (YAML and/or caption) for print and removal.

- `scripts/archive/04_batch_crop_tool.py` — Legacy; uses `safe_delete(png_path, yaml_path)`.
  - Note: Mark as legacy; optional to update.

**Actions:**
- Replace base `safe_delete` implementation with centralized utility:
  - `safe_delete_image_and_yaml(png_path, hard_delete=False, tracker=self.tracker)` (already deletes ALL companions)
- Update `triplet_deduplicator.py` to:
  - Use `find_all_companion_files(png_file)` for dry-run printing and actual deletion
  - Use `safe_delete_paths([png] + companions, ...)` or `safe_delete_image_and_yaml`
- Ensure FileTracker logging preserved (source dir, file_count, files list)

**Notes:**
- `01_desktop_image_selector_crop.py` already delegates deletes to the centralized companion delete — no change needed.
- Add tests for caption-delete paths in `utils/triplet_deduplicator.py` after changes.

---

## 📚 **LOW PRIORITY / FUTURE**

### 4. Code Conventions & Patterns Catalog
**Create:** `Documents/CONVENTIONS_REFERENCE.md`
- Analyze all scripts for reusable patterns
- Document Flask structure, CSS, JavaScript patterns
- Document matplotlib setup, event handling
- Create ready-to-use code templates
- Benefits: Consistency, maintainability, easier onboarding

### 5. Create Local Homepage
Build custom homepage in Documents with links to all AI systems and tools

### 6. Web Interface Template System Investigation
Evaluate if template would simplify web tool maintenance vs add complexity

### 7. Experiment: Hand/Foot Anomaly Scripts
General line item to check out/test hand and foot anomaly scripts when time allows. Purpose: see if they catch any of my mistakes or produce signals our AI could use. Not urgent; exploratory—run on recent batches and jot a brief note on usefulness and potential integration.

---

## 📊 **Dashboard Enhancements (Optional)**
1. Historical average overlays
2. Script update correlation with productivity
3. Pie chart time distribution
4. CSV/JSON data export
5. GitHub integration for change tracking

---

## 📝 **Documentation Updates Needed**
1. Add desktop hotkey reference to Knowledge Base (p [ ] \\ and A/S/D/F/B)
2. Document training log flags and schemas
3. Update backup system runbook when implemented
