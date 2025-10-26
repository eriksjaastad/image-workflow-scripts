# Code Review: Orphan .decision Cleanup Feature

## Summary
Added orphan .decision file cleanup to character sorter with preview and staging functionality.

**Files Changed:** `scripts/03_web_character_sorter.py` (+62 lines)

---

## Issues Found

### 🔴 CRITICAL: File Safety Violation

**Location:** `scripts/03_web_character_sorter.py:934`
```python
shutil.move(str(d), str(dest))
```

**Issue:** Uses `shutil.move()` directly instead of blessed companion-aware utilities.

**From `.cursorrules` Rule #2:**
> ✅ **SAFE Operations:**
> - Moving files (with companions): `move_file_with_all_companions()`

**Required Fix:**
```python
from utils.companion_file_utils import move_file_with_all_companions

# In stage_orphan_decisions():
moved_files = move_file_with_all_companions(d, staging, dry_run=False)
moved.append(d.name)
```

---

### 🟡 MEDIUM: Performance - O(n*m) Glob Pattern

**Location:** `scripts/03_web_character_sorter.py:905, 930`
```python
matches = list(repo_root.rglob(f"**/{stem}.png"))
```

**Issue:** Runs full recursive glob for EACH .decision file. With 100 orphans, that's 100 full repo scans.

**Optimization:**
```python
# Build PNG stem set once (O(n))
png_stems = {p.stem for p in repo_root.rglob("**/*.png")}

# Check orphans (O(1) per decision)
for d in decisions:
    if d.stem not in png_stems:
        orphans.append(str(d.resolve()))
```

**Impact:** Could reduce runtime from minutes to seconds on large repos.

---

### 🟡 MEDIUM: Undocumented Safe Zone

**Location:** `scripts/03_web_character_sorter.py:922`
```python
staging = repo_root / "__delete_staging"
```

**Issue:** Creates `__delete_staging/` directory, but it's not listed in `.cursorrules` Rule #3 designated safe zones.

**Recommendation:** Add to `.cursorrules`:
```markdown
### **Rule #3: Designated Safe Zones**
**Where NEW files CAN be created:**
- `data/ai_data/` - AI training data, embeddings, caches
- `data/file_operations_logs/` - Operation logs
- `sandbox/` - Testing and experiments
- `__delete_staging/` - Orphaned files awaiting manual review before deletion
```

---

### 🟢 MINOR: Truncated FileTracker Logging

**Location:** `scripts/03_web_character_sorter.py:945`
```python
files=moved[:10],
notes="orphan .decision staged"
```

**Issue:** Only logs first 10 files but doesn't indicate truncation.

**Better:**
```python
notes=f"orphan .decision staged ({len(moved)} total)"
```

---

## Verdict

**❌ NEEDS FIXES** - Critical file safety violation must be addressed.

**Required before merge:**
1. Replace `shutil.move()` with `move_file_with_all_companions()`
2. Optimize glob performance (build PNG set once)
3. Document `__delete_staging/` in `.cursorrules`

**Nice to have:**
4. Show total count in FileTracker notes

---

**Reviewed by:** Claude
**Date:** 2025-10-26
