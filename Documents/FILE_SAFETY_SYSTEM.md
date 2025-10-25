# File Safety System
**Last Updated:** October 20, 2025

## 🎯 Goal
Ensure NO scripts accidentally modify production image or companion files (except the crop tool).

---

## 🛡️ Protection Layers

### **Layer 1: Cursor AI Rules** (`.cursorrules`)
- Explicit rules that Cursor AI must follow when writing code
- Prevents accidental file modifications during development
- Located: `.cursorrules` in project root

**Key Rules:**
- ✅ Move/delete operations allowed
- ✅ Creating NEW files in safe zones allowed
- ✅ Crop tool can write cropped images
- ❌ NO modifications to existing production files
- ❌ NO overwrites of existing images/YAML/captions

### **Layer 2: Code Audit Script** (`scripts/tools/audit_file_safety.py`)
- Scans all Python scripts for dangerous patterns
- Identifies potential file modification operations
- Run regularly to catch violations

**Usage:**
```bash
# Quick scan
python scripts/tools/audit_file_safety.py

# Verbose mode (shows all matches, even safe ones)
python scripts/tools/audit_file_safety.py --verbose
```

**What it detects:**
- `open()` with write modes ('w', 'wb', 'a')
- `Image.save()`, `cv2.imwrite()`
- `shutil.copy()` that might overwrite
- `yaml.dump()`, `json.dump()` to files
- `.write()` calls on file objects

### **Layer 3: FileTracker Audit Trail**
- All file operations logged to `data/file_operations_logs/`
- Can verify what actually happened to files
- Provides forensic evidence if something goes wrong

**Check logs:**
```bash
# Find all operations on a specific file
grep "filename.png" data/file_operations_logs/file_operations_20251020.log

# See all crop operations today
grep '"operation": "crop"' data/file_operations_logs/file_operations_20251020.log
```

### **Layer 4: Design Philosophy**
- **Move, Don't Modify:** Scripts move files between directories, they don't change contents
- **Read-Only by Default:** All production files treated as immutable
- **New Versions, Not Updates:** Create new files instead of modifying existing ones
- **Companion Integrity:** Always move image + companions together

---

## 📁 Safe Zones (Where NEW Files CAN Be Created)

```
✅ data/ai_data/            - AI training data, embeddings
✅ data/file_operations_logs/ - Operation logs
✅ data/daily_summaries/    - Report files
✅ data/dashboard_archives/ - Dashboard data
✅ sandbox/                 - Testing area
✅ Temp directories created by scripts
```

## 🚫 Protected Zones (NO Modifications)

```
❌ mojo1/, mojo2/     - Production images
❌ selected/          - Selected images
❌ crop/              - Cropped images (except crop tool creating them)
❌ Any directory with production images
```

---

## 🔧 Allowed File Operations

### ✅ **SAFE Operations**

**Moving Files:**
```python
from utils.companion_file_utils import move_file_with_all_companions
move_file_with_all_companions(image_path, target_dir)
```

**Deleting Files:**
```python
from send2trash import send2trash
send2trash(file_path)  # Goes to macOS Trash (recoverable)
```

**Reading Files:**
```python
# YAML
data = yaml.safe_load(open(yaml_path, 'r'))

# Text
content = Path(file_path).read_text()

# Images
image = PIL.Image.open(image_path)
```

**Creating NEW Files (in safe zones):**
```python
# Reports/logs in data/
with open('data/daily_summaries/summary.json', 'w') as f:
    json.dump(data, f)

# Thumbnails in temp
thumbnail.save(f'data/ai_data/thumbnails/{name}_thumb.png')
```

### ❌ **FORBIDDEN Operations**

```python
# ❌ Overwriting existing production files
with open('mojo2/_asian/image.png', 'wb') as f:
    f.write(data)

# ❌ Modifying images in-place
image = PIL.Image.open('selected/photo.png')
image.save('selected/photo.png')  # Overwrites!

# ❌ Overwriting companion files
with open('crop/image.yaml', 'w') as f:
    yaml.dump(data, f)
```

**Exception:** `02_ai_desktop_multi_crop.py` CAN write cropped images (this is its purpose).

---

## 🔍 How to Use This System

### **During Development:**
1. Cursor AI automatically follows `.cursorrules`
2. Any file write operations require justification
3. Claude will warn if operation looks dangerous

### **Before Committing Code:**
```bash
# Run safety audit
python scripts/tools/audit_file_safety.py

# Review any flagged issues
# Verify they're either:
#   - In safe zones (creating NEW files)
#   - In crop tool (allowed exception)
#   - False positives (logging, etc.)
```

### **After Running Scripts:**
```bash
# Check what actually happened
tail -20 data/file_operations_logs/file_operations.log

# Verify specific file
grep "suspicious_file.png" data/file_operations_logs/*.log
```

### **If Something Goes Wrong:**
1. Check FileTracker logs for timeline
2. Look for unexpected 'crop' or 'modify' operations
3. Use git to recover if needed: `git checkout filename`
4. Check macOS Trash (~/.Trash/) for deleted files

---

## 📊 Regular Maintenance

**Weekly:**
```bash
# Run safety audit
python scripts/tools/audit_file_safety.py
```

**Monthly:**
```bash
# Review all file operations
grep -E '"operation": "(crop|modify|delete)"' data/file_operations_logs/*.log | wc -l

# Should see:
# - Lots of 'move' operations ✅
# - Some 'crop' operations (from crop tool) ✅
# - Some 'delete' operations (to Trash) ✅
# - NO 'modify' operations (except crop) ❌
```

**After Adding New Scripts:**
```bash
# Audit new script specifically
python scripts/tools/audit_file_safety.py --verbose | grep "new_script.py"
```

---

## 🚨 When to Be Concerned

**Red Flags:**
- ❌ File modification operations outside crop tool
- ❌ Overwrites without logging
- ❌ Image writes to production directories
- ❌ YAML/caption modifications without justification
- ❌ Missing FileTracker logs for operations

**Safe Patterns:**
- ✅ 'move' operations with companion files
- ✅ 'delete' operations via send2trash
- ✅ 'crop' operations from multi_crop_tool
- ✅ Creating NEW files in data/ directories
- ✅ Reading files (no modification)

---

## 💡 Philosophy

**"Data is permanent, code is temporary"**

We can always fix code, but we can't recover lost or corrupted data. When in doubt:
1. Don't modify the file
2. Create a new version
3. Log the operation
4. Test in sandbox first

**Remember:** One accidental overwrite can destroy hours of work. These protections exist to prevent that!

---

## 🔗 Related Documentation

- `.cursorrules` - AI coding rules
- `scripts/tools/audit_file_safety.py` - Safety audit tool
- `scripts/utils/companion_file_utils.py` - Safe file operations
- `Documents/TECHNICAL_KNOWLEDGE_BASE.md` - File operation patterns

