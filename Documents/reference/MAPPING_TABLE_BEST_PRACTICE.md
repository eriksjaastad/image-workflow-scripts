# Best Practice: Project Mapping Tables for Batch Operations

**Date:** 2025-11-01  
**Author:** Claude (at Erik's direction)  
**Status:** Recommended Best Practice

---

## 📋 The Problem

When running batch operations across multiple projects with:
- Different naming conventions (spaces, underscores, suffixes)
- Multiple source locations (zips, directories)
- Existing databases with various schemas
- Potential mismatches or missing items

**It's easy to:**
- ❌ Miss projects
- ❌ Use wrong file paths
- ❌ Create errors from name mismatches
- ❌ Not realize databases are missing until mid-run
- ❌ Waste hours troubleshooting

---

## ✅ The Solution: Create a Mapping Table FIRST

**Before running any batch operation:**

1. **List ALL source files** (zips, directories, etc.)
2. **List ALL target files** (databases, output directories, etc.)
3. **Create a markdown table** showing the mapping
4. **Identify gaps, mismatches, and issues** BEFORE running anything
5. **Resolve questions** with the human
6. **Update the script** with verified mappings
7. **Run with confidence**

---

## 📊 Example: Batch AI Predictions

See: `data/ai_data/ZIP_DATABASE_MAPPING.md`

**What it showed:**
- ✅ 15 projects ready to process
- ❌ 5 projects missing databases
- ⚠️ 3 name mismatches (Eleni, Kiara, 1101)
- ✅ All issues resolved before running

**Result:**
- Zero surprises during execution
- Clear plan for handling edge cases
- Human approval before proceeding
- Saved hours of debugging

---

## 🎯 When to Use This

**Always create a mapping table for:**

1. **Batch file operations**
   - Moving/copying files across projects
   - Renaming files in bulk
   - Archiving completed work

2. **Database operations**
   - Backfilling data across multiple databases
   - Schema migrations
   - Data exports/imports

3. **Training data preparation**
   - Matching images to labels
   - Verifying dataset completeness
   - Cross-referencing multiple sources

4. **Project cleanup**
   - Identifying orphaned files
   - Finding missing companions
   - Verifying project structure

---

## 📝 Mapping Table Template

```markdown
| Source | Target | Status | Notes |
|--------|--------|--------|-------|
| file1.zip | db1.db | ✅ Match | |
| file2.zip | db2.db | ⚠️ Mismatch | Name differs |
| file3.zip | MISSING | ❌ No target | Create first |
```

**Status codes:**
- ✅ Match - Ready to process
- ⚠️ Mismatch - Name differs but same project
- ❌ Missing - Target doesn't exist

---

## 🔍 Key Benefits

1. **Catches issues early** - Before wasting compute time
2. **Provides clarity** - Shows exactly what will happen
3. **Enables review** - Human can spot problems
4. **Documents decisions** - Why certain choices were made
5. **Saves time** - No surprises mid-batch
6. **Builds confidence** - Know what you're running

---

## 💡 Pro Tips

1. **Generate automatically** - Use scripts to list files, then create table
2. **Include counts** - Number of images, records, etc.
3. **Add notes** - Explain naming differences
4. **Version control** - Commit the mapping document
5. **Update after changes** - Keep it current
6. **Reference in scripts** - Add comment linking to mapping doc

---

## 🎓 Real Example

**Erik's reaction to the mapping table:**
> "I am shocked that we are missing databases. I also have to say nobody else has made a table like this. This is a brilliant idea."

**Why it worked:**
- Showed gaps clearly (5 missing databases)
- Resolved naming confusion (Eleni, Kiara, Patricia)
- Provided confidence to proceed
- Created reusable reference document

---

## 📁 Where to Store Mapping Tables

```
data/ai_data/ZIP_DATABASE_MAPPING.md
data/ai_data/PROJECT_INVENTORY_MAPPING.md
data/backups/BACKUP_RESTORE_MAPPING.md
Documents/reference/DATASET_MAPPINGS/
```

---

## ✅ Checklist Before Batch Operations

Before running any batch script:

- [ ] Created mapping table for all sources → targets
- [ ] Identified all gaps and mismatches
- [ ] Resolved questions with human
- [ ] Updated script with verified mappings
- [ ] Added mapping document path to script comments
- [ ] Reviewed table with human for approval
- [ ] Documented any special cases in Notes column

---

## 🎯 Summary

**Instead of:**
```python
# Hope these match up!
projects = ["1011", "1012", "Eleni", ...]
```

**Do this:**
```python
# See data/ai_data/ZIP_DATABASE_MAPPING.md for verified mappings
projects = [
    ("1011", "1011.zip"),        # ✅ Verified
    ("Eleni", "Eleni_raw.zip"),  # ✅ Name mismatch resolved
    ...
]
```

**Result:** Confidence, clarity, and no surprises. 🎯

---

**Remember:** A few minutes creating a mapping table can save hours of debugging. Always map first, run second.

