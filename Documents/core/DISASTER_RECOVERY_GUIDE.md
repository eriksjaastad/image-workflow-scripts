# Disaster Recovery Guide
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26

**Date:** October 21, 2025  
**Purpose:** What you get back from GitHub if your computer dies

---

## 🚨 **Critical Question: "What if my computer falls in the toilet?"**

If you lose your machine and need to recover from GitHub, here's **exactly** what you'll get back:

---

## ✅ **What IS Backed Up on GitHub**

### **1. All Code & Scripts**
- ✅ `scripts/` - All tools (image selector, cropper, sorter, etc.)
- ✅ `Documents/` - All documentation
- ✅ `.gitignore`, `requirements.txt`, etc.

### **2. Critical Infrastructure Files** (as of Oct 21, 2025)
- ✅ `data/schema/` - **7 JSON schemas** (crop_training_v2.json, project_v1.json, etc.)
- ✅ `data/projects/` - **All 19 project manifests** (mojo1.project.json, mojo2.project.json, etc.)
- ✅ `data/ai_data/models/` - **Trained AI models** (ranker_v3_w10.pt, crop_proposer_v2.pt, metadata files)
- ✅ `data/training/` - **Training data CSVs** (select_crop_log.csv, selection_only_log.csv)

**Note:** These were just added to git tracking on Oct 21, 2025. Before today, they were NOT backed up!

---

## ❌ **What Is NOT Backed Up**

### **1. Raw Image Files** (By Design - Too Large)
- ❌ `mojo1/`, `mojo2/`, `mojo3/` - Raw project images (~100 GB+)
- ❌ `training data/` - Training images (~90 GB)
- ❌ `__selected/`, `__crop/`, `__character_group_*/` - Working directories
- ❌ `exports/` - Final deliverables

**Why:** GitHub has a 100 GB repo limit. Image files are huge.

**Solution:** Keep image backups on external drive or cloud storage.

---

### **2. Personal/Productivity Data**
- ❌ `data/timesheet.csv` - Billing data (private)
- ❌ `data/file_operations_logs/` - Daily operation logs
- ❌ `data/log_archives/` - Historical logs
- ❌ `data/snapshot/` - Snapshot data warehouse
- ❌ `data/timer_data/` - Timer session data
- ❌ `data/daily_summaries/` - Daily productivity summaries

**Why:** Personal data, regenerable from logs, or too large.

**Solution:** 
- Timesheet: Back up separately (Dropbox, Google Drive)
- Logs/snapshots: Regenerable from raw logs if needed

---

### **3. Temporary/Cache Files**
- ❌ `data/ai_data/cache/` - Embedding cache
- ❌ `data/ai_data/embeddings/` - Generated embeddings
- ❌ `thumbnails/`, `temp/`, `sandbox/` - Working files
- ❌ `__pycache__/`, `.pytest_cache/` - Python cache

**Why:** Regenerable, temporary, or cache files.

**Solution:** Regenerate after recovery (embeddings take time but are reproducible).

---

## 🔧 **Recovery Process**

### **Step 1: Clone Repository**
```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
```

**You Get:**
- ✅ All scripts
- ✅ All docs
- ✅ Schemas
- ✅ Project manifests
- ✅ AI models
- ✅ Training data CSVs

---

### **Step 2: Restore Image Files** (From Backup)
```bash
# Restore from external drive or cloud storage
# Example:
cp -r /Volumes/Backup/mojo1/ ./mojo1/
cp -r /Volumes/Backup/mojo2/ ./mojo2/
cp -r "/Volumes/Backup/training data/" "./training data/"
```

**You Need:**
- Your external drive with image backups
- Or cloud storage (Dropbox, Google Drive, iCloud)

---

### **Step 3: Recreate Environment**
```bash
# Create virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -r requirements_ai_training.txt
pip install flask pillow send2trash PyYAML
```

---

### **Step 4: Regenerate Embeddings** (Optional)
```bash
# If you need AI features, regenerate embeddings
# This takes time but is fully reproducible
python scripts/ai/compute_embeddings.py
```

---

### **Step 5: Restore Personal Data** (From Backup)
```bash
# Restore timesheet (if backed up separately)
cp ~/Dropbox/Backup/timesheet.csv data/timesheet.csv

# Restore logs (if backed up)
cp -r ~/Dropbox/Backup/file_operations_logs/ data/file_operations_logs/
```

---

## 📋 **Current Backup Status** (Oct 21, 2025)

| Item | Backed Up? | Location | Size | Recovery Priority |
|------|-----------|----------|------|-------------------|
| **Scripts** | ✅ GitHub | `scripts/` | <1 MB | **CRITICAL** |
| **Docs** | ✅ GitHub | `Documents/` | <5 MB | **HIGH** |
| **Schemas** | ✅ GitHub | `data/schema/` | <20 KB | **CRITICAL** |
| **Project Manifests** | ✅ GitHub | `data/projects/` | <100 KB | **CRITICAL** |
| **AI Models** | ✅ GitHub | `data/ai_data/models/` | ~50 MB | **HIGH** |
| **Training CSVs** | ✅ GitHub | `data/training/*.csv` | <10 MB | **HIGH** |
| **Raw Images (mojo1-3)** | ❌ External | N/A | ~100 GB | **MEDIUM** |
| **Training Images** | ❌ External | `training data/` | ~90 GB | **MEDIUM** |
| **Timesheet** | ❌ Manual | `data/timesheet.csv` | <1 MB | **HIGH** |
| **Logs** | ❌ Regenerable | `data/file_operations_logs/` | ~50 MB | **LOW** |
| **Embeddings** | ❌ Regenerable | `data/ai_data/embeddings/` | ~500 MB | **LOW** |
| **Snapshots** | ❌ Regenerable | `data/snapshot/` | ~20 MB | **LOW** |

---

## 🎯 **Recommended Backup Strategy**

### **1. GitHub (Automatic)**
- ✅ Code, docs, schemas, manifests, models (already set up)
- Commit regularly: `git push`

### **2. External Drive (Weekly)**
- Raw project images (`mojo1/`, `mojo2/`, etc.)
- Training images (`training data/`)
- Final deliverables (`exports/`)

### **3. Cloud Backup (Daily/Weekly)**
- Timesheet (`data/timesheet.csv`)
- Optional: Project manifests (redundant with GitHub)
- Optional: Current project raw images (for active work)

### **4. Don't Back Up (Regenerable)**
- Logs (`data/file_operations_logs/`, `data/snapshot/`)
- Cache (`data/ai_data/cache/`, embeddings)
- Temporary files

---

## ⚠️ **What Was Fixed Today** (Oct 21, 2025)

**Before:** `data/schema/` and `data/projects/` were NOT tracked by git!

**After:** Added to `.gitignore` as exceptions:
```gitignore
data/*                  # Ignore most of data/
!data/schema/           # But allow schemas
!data/projects/         # But allow project manifests
!data/ai_data/models/   # But allow AI models
!data/training/         # But allow training CSVs
```

**Impact:** If your computer died yesterday (Oct 20), you would have lost:
- ❌ All 7 schema definitions
- ❌ All 19 project manifests
- ❌ Would need to recreate from memory

**Now (Oct 21):** These are safe on GitHub!

---

## 🧪 **Test Your Recovery** (Recommended)

1. Clone repo to a temp directory (simulate fresh machine)
2. Verify all critical files are present
3. Check you can run basic scripts
4. Confirm AI models load correctly

```bash
# Test recovery in temp location
cd /tmp
git clone [your-repo] test-recovery
cd test-recovery
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements_ai_training.txt
python scripts/01_ai_assisted_reviewer.py --help  # Should work
```

---

## 📝 **Maintenance Checklist**

- [ ] Commit code changes regularly (`git push`)
- [ ] Weekly: Back up raw images to external drive
- [ ] Weekly: Export timesheet to Dropbox/Google Drive
- [ ] Monthly: Test recovery process
- [ ] After each project: Back up final deliverables

---

**Bottom Line:** As of Oct 21, 2025, your **infrastructure** (code, schemas, manifests, models) is safe on GitHub. Your **images and personal data** need separate backup strategy (external drive + cloud).

**Computer falls in toilet? You're OK!** (As long as images are backed up externally)

