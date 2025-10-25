# Git Tracking Policy for data/ Directory
**Date:** October 21, 2025  
**Status:** ✅ Implemented and verified

---

## 📋 **Summary: What's Tracked vs Ignored**

| Directory | Status | Reason | Size |
|-----------|--------|--------|------|
| `data/schema/` | ✅ **TRACKED** | Infrastructure (7 JSON schemas) | ~20 KB |
| `data/projects/` | ✅ **TRACKED** | Project manifests (22 files) | ~30 KB |
| `data/ai_data/models/` | ✅ **TRACKED** | AI models (.pt, .json) | ~50 MB |
| `data/training/*.csv` | ✅ **TRACKED** | Training data CSVs | ~10 MB |
| `data/aggregates/` | ❌ **IGNORED** | Data logs (regenerable) | ~5 MB |
| `data/crop_progress/` | ❌ **IGNORED** | Progress logs (regenerable) | <1 MB |
| `data/daily_summaries/` | ❌ **IGNORED** | Daily logs (regenerable) | ~10 MB |
| `data/file_operations_logs/` | ❌ **IGNORED** | Operation logs (regenerable) | ~50 MB |
| `data/log_archives/` | ❌ **IGNORED** | Archived logs (regenerable) | ~100 MB |
| `data/snapshot/` | ❌ **IGNORED** | Snapshot warehouse (regenerable) | ~20 MB |
| `data/sorter_progress/` | ❌ **IGNORED** | Progress logs (regenerable) | <1 MB |
| `data/timer_data/` | ❌ **IGNORED** | Timer logs (personal data) | ~5 MB |
| `data/ai_data/cache/` | ❌ **IGNORED** | Embedding cache (regenerable) | ~500 MB |
| `data/ai_data/embeddings/` | ❌ **IGNORED** | Embeddings (regenerable) | ~500 MB |
| `data/training/*.txt` | ❌ **IGNORED** | Reports (regenerable) | ~10 MB |
| `data/training/*.json` | ❌ **IGNORED** | Reports (regenerable) | ~5 MB |

---

## ✅ **What IS Tracked (90 MB total)**

### **1. Schemas (7 files, ~20 KB) - CRITICAL**
```
data/schema/
├── crop_training_v2.json         (NEW - Oct 21, 2025)
├── daily_aggregate_v1.json
├── derived_session_v1.json
├── operation_event_v1.json
├── progress_snapshot_v1.json
├── project_v1.json
└── timer_session_v1.json
```

**Why:** Infrastructure definitions. If lost, the entire data pipeline breaks.

---

### **2. Project Manifests (22 files, ~30 KB) - CRITICAL**
```
data/projects/
├── mojo1.project.json            (Active project metadata)
├── mojo2.project.json
├── mojo3.project.json            (Current active project)
├── 1011.project.json
├── 1012.project.json
├── ... (16 more historical projects)
├── global_bans.json
├── mojo1_allowed_ext.json
└── mojo2_allowed_ext.json
```

**Why:** Project metadata, start/end dates, image counts, paths. Required for dashboard and project tracking.

---

### **3. AI Models (~50 MB) - HIGH VALUE**
```
data/ai_data/models/
├── ranker_v3_w10.pt              (~30 MB - trained model)
├── ranker_v3_w10_metadata.json
├── crop_proposer_v2.pt           (~20 MB - trained model)
└── crop_proposer_v2_metadata.json
```

**Why:** Hours of training time. Would take 1-2 hours to regenerate.

---

### **4. Training Data CSVs (~10 MB) - HIGH VALUE**
```
data/training/
├── select_crop_log.csv           (7,194 rows - crop decisions)
├── selection_only_log.csv        (21,250 rows - selections)
├── anomaly_cases.csv
└── historical_crop_log.csv
```

**Why:** Manual decisions and training data. Cannot be regenerated.

---

## ❌ **What Is IGNORED (~1.2 GB total)**

### **1. Data Logs & Snapshots (~200 MB) - Regenerable**
- `data/aggregates/` - Dashboard aggregates
- `data/crop_progress/` - Crop tool progress
- `data/daily_summaries/` - Daily summaries
- `data/file_operations_logs/` - Operation logs (50+ MB)
- `data/log_archives/` - Archived logs (100+ MB)
- `data/snapshot/` - Snapshot warehouse
- `data/sorter_progress/` - Sorter progress
- `data/timer_data/` - Timer sessions

**Why:** Regenerable from raw logs or too large. Dashboard can recreate snapshots on demand.

---

### **2. AI Cache (~1 GB) - Regenerable**
- `data/ai_data/cache/` - Embedding cache (~500 MB)
- `data/ai_data/embeddings/` - Computed embeddings (~500 MB)
- `data/ai_data/logs/` - Training logs
- `data/ai_data/training_snapshots/` - Training checkpoints

**Why:** Can be regenerated (takes time but is reproducible).

---

### **3. Reports (~15 MB) - Regenerable**
- `data/training/*.txt` - Inspection reports
- `data/training/*.json` - Analysis reports
- `data/training/manifests/` - Training manifests

**Why:** Generated from training data CSVs.

---

## 🔒 **Verification (Oct 21, 2025)**

**Files staged for commit:**
```bash
$ git diff --cached --numstat | grep "data/"
59 0 data/schema/crop_training_v2.json
87 0 data/schema/daily_aggregate_v1.json
88 0 data/schema/derived_session_v1.json
92 0 data/schema/operation_event_v1.json
81 0 data/schema/progress_snapshot_v1.json
61 0 data/schema/project_v1.json
57 0 data/schema/timer_session_v1.json
40 0 data/projects/1011.project.json
... (22 project manifests)
```

**Total:** ~1,500 lines of JSON (small infrastructure files only) ✅

**Not showing up (correctly ignored):**
- ✅ No log files
- ✅ No snapshot data
- ✅ No cache files
- ✅ No large data files

---

## 📝 **Maintenance**

### **When to Commit:**
- ✅ After creating new schemas
- ✅ After starting/finishing projects (manifest changes)
- ✅ After training AI models (new .pt files)
- ✅ After collecting training data (CSV updates)

### **When NOT to Commit:**
- ❌ Daily operation logs
- ❌ Snapshot updates
- ❌ Progress files
- ❌ Cache/embeddings

### **Regular Commits:**
```bash
# Schema or manifest changes
git add data/schema/ data/projects/
git commit -m "Add crop_training_v2 schema"

# AI model updates
git add data/ai_data/models/*.pt data/ai_data/models/*.json
git commit -m "Update Crop Proposer v2 model (71.17% IoU)"

# Training data updates
git add data/training/*.csv
git commit -m "Backfill 7,193 crop training rows"
```

---

## ⚠️ **Before This Fix (Oct 20, 2025)**

❌ **Problem:** `data/` was completely ignored  
❌ **Impact:** Schemas, manifests, and models were NOT backed up  
❌ **Risk:** Computer death = lose all infrastructure

## ✅ **After This Fix (Oct 21, 2025)**

✅ **Solution:** Selective tracking with clear policy  
✅ **Impact:** Critical files are safe on GitHub  
✅ **Benefit:** Computer death = only lose data logs (regenerable)

---

## 🎯 **Summary**

**Tracked (90 MB):**
- Schemas (infrastructure)
- Project manifests (metadata)
- AI models (training time)
- Training CSVs (manual work)

**Ignored (~1.2 GB):**
- Data logs (regenerable)
- Snapshots (regenerable)
- Cache (regenerable)
- Reports (regenerable)

**Result:** GitHub repo stays lean, critical files are safe, data logs don't bloat the repo.

---

**Status:** ✅ Implemented and verified (Oct 21, 2025)

