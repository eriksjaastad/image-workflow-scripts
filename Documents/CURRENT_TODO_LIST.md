# Current TODO List

**Last Updated:** October 5, 2025

## ⚠️ **IMPORTANT WORKFLOW RULE**
**ONE TODO ITEM AT A TIME** - Complete one task, check in with Erik, get approval before moving to the next item. Never complete multiple TODO items in sequence without user input. This prevents issues and ensures quality control.

---

## 🔥 **HIGH PRIORITY**

### 1. Desktop Selector Selection Toggle Bug
**Issue:** When one image is selected by hotkey and then another image's crop position is modified, the original image doesn't toggle back to delete.
- **Expected:** Only ONE image should be kept at a time
- **Impact:** UX issue during long image processing sessions
- **Priority:** HIGH - affects daily workflow
- **File:** `scripts/01_desktop_image_selector_crop.py`

---

## 🧠 **AI Select+Crop Training Plan**

### Phase 1 — Logging (no workflow disruption)
1. Desktop: `--log-training` flag already implemented
   - Writes to `data/training/select_crop_log.csv`
   - Columns: session_id, set_id, directory, image_count, chosen_index, chosen_path, crop coords, etc.
2. Web selector: `--log-training` flag already implemented
   - Writes to `data/training/selection_only_log.csv`
3. **TODO:** Document flags, file locations, and schemas in Knowledge Base

### Phase 2 — Dataset Builder
**Create:** `scripts/datasets/build_select_crop_dataset.py`
- Inputs: the two CSVs; validate file existence
- Split by set_id into train/val/test (80/10/10)
- Outputs: Ranking pairs (JSONL), Crop samples (JSONL), optional COCO JSON

### Phase 3 — Training (two-head model)
**Create:** `scripts/train/train_select_crop.py`
- Backbone: ViT-B/16
- Heads: ranking (embedding+MarginRankingLoss) + bbox (SmoothL1+IoU)
- Loss: total = 1.0*L_rank + 2.0*L_bbox (tunable)
- Optimizer: AdamW; cosine decay; 10–30 epochs

### Phase 4 — Evaluation
**Create:** `scripts/eval/eval_select_crop.py` + notebook
- Metrics: Top-1, IoU@0.5, mean IoU, MAE
- Qualitative side-by-sides

### Phase 5 — Inference Integration
**Add to desktop tool:** `--ai-suggest` flag
- Score images → suggest top-1; predict crop; draw suggested rect
- Hotkeys: T (toggle suggestions), Y (accept suggestion)
- Always user-controlled, no auto-submit

**Dependencies:** torch, torchvision, timm, albumentations, scikit-learn, pycocotools, tqdm, pyyaml, rich

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

### 2. Improve Test Coverage
- Add unsorted input tests (verify pre-sorting requirement)
- Add unknown stage handling tests
- Add min_group_size variation tests
- Add two-runs-in-sequence tests
- Add more edge case tests

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

### 4. Fix Remaining 6 Non-Priority Test Failures
- 3 unique issues (date sorting, dashboard verification)
- All priority hotspots already fixed
- Not blocking work

---

## 📚 **LOW PRIORITY / FUTURE**

### 5. Code Conventions & Patterns Catalog
**Create:** `Documents/CONVENTIONS_REFERENCE.md`
- Analyze all scripts for reusable patterns
- Document Flask structure, CSS, JavaScript patterns
- Document matplotlib setup, event handling
- Create ready-to-use code templates
- Benefits: Consistency, maintainability, easier onboarding

### 6. Root Directory Cleanup
Check files in root directory - clean out or organize into proper directories

### 7. Create Local Homepage
Build custom homepage in Documents with links to all AI systems and tools

### 8. Web Interface Template System Investigation
Evaluate if template would simplify web tool maintenance vs add complexity

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
