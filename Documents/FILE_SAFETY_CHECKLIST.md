# File Safety Reminder Checklist
**For AI Sessions & Code Reviews**

---

## 🚨 **START HERE - Read Every Session**

When starting a new AI session or code review, verify file safety:

### **Quick Checks:**
```bash
# 1. Run safety audit (takes ~5 seconds)
python scripts/tools/audit_file_safety.py

# 2. Review Cursor rules
cat .cursorrules

# 3. Check recent file operations
tail -20 data/file_operations_logs/file_operations.log
```

---

## 📋 **File Safety Checklist**

### **Before Writing Code:**
- [ ] Reviewed `.cursorrules` file safety rules
- [ ] Understand: Only crop tool modifies images
- [ ] Know safe zones: `data/`, `sandbox/`
- [ ] Know protected zones: `mojo1/`, `mojo2/`, `selected/`, `crop/`

### **While Writing Code:**
- [ ] All file writes go to safe zones
- [ ] No overwrites of existing production files
- [ ] Using `move_file_with_all_companions()` for moves
- [ ] Using `send2trash()` for deletes
- [ ] Logging operations via FileTracker

### **Before Committing:**
- [ ] Run: `python scripts/tools/audit_file_safety.py`
- [ ] Review flagged issues
- [ ] Verify writes are in safe zones
- [ ] Run test suite: `python scripts/tests/test_file_safety_audit.py`
- [ ] Check FileTracker logs for unexpected operations

### **After Running Scripts:**
- [ ] Check logs: `tail data/file_operations_logs/file_operations.log`
- [ ] Verify operations match expectations
- [ ] No unexpected 'modify' or 'crop' operations
- [ ] All moves include companion files

---

## 🚫 **NEVER DO THESE:**

```python
# ❌ Overwrite existing production images
with open('mojo2/_asian/image.png', 'wb') as f:
    f.write(data)

# ❌ Modify images in-place (except crop tool)
image = PIL.Image.open('selected/photo.png')
image.save('selected/photo.png')

# ❌ Overwrite companion files
with open('crop/image.yaml', 'w') as f:
    yaml.dump(data, f)

# ❌ Move without companions
shutil.move(image_path, target_dir)  # Missing YAML/caption!
```

---

## ✅ **ALWAYS DO THESE:**

```python
# ✅ Move with all companions
from utils.companion_file_utils import move_file_with_all_companions
move_file_with_all_companions(image_path, target_dir)

# ✅ Delete to Trash (recoverable)
from send2trash import send2trash
send2trash(file_path)

# ✅ Create NEW files in safe zones
with open('data/daily_summaries/report.json', 'w') as f:
    json.dump(data, f)

# ✅ Read files (no modification)
data = yaml.safe_load(open(yaml_path, 'r'))
image = PIL.Image.open(image_path)
```

---

## 🔍 **Quick Commands**

```bash
# Run safety audit
python scripts/tools/audit_file_safety.py

# Run safety test
python scripts/tests/test_file_safety_audit.py

# Check recent operations
tail -50 data/file_operations_logs/file_operations.log

# Find operations on specific file
grep "filename.png" data/file_operations_logs/*.log

# See all crop operations
grep '"operation": "crop"' data/file_operations_logs/*.log
```

---

## 📚 **Documentation**

- **`.cursorrules`** - AI coding rules (in project root)
- **`Documents/FILE_SAFETY_SYSTEM.md`** - Complete guide
- **`Documents/TECHNICAL_KNOWLEDGE_BASE.md`** - Quick reference (top of file)
- **`scripts/tools/audit_file_safety.py`** - Safety audit tool
- **`scripts/tests/test_file_safety_audit.py`** - Safety test

---

## 💡 **Remember:**

> **"Data is permanent, code is temporary"**
> 
> We can always fix code, but we can't recover corrupted data.
> When in doubt, DON'T modify the file.

---

**Last Updated:** October 20, 2025

