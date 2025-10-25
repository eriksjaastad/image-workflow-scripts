# 🚨 GIT REPOSITORY DISASTER RECOVERY PLAN

## Problem Discovered

Date: October 25, 2025
By: System Diagnostic Script

### Critical Issues Found

1. **Local `main` branch is CORRUPTED** (commit 78710b091)
   - Contains 50,466 files (should be ~500)
   - Has entire `.venv311/` directory committed (648 MB of binaries)
   - Has `.coverage` file committed
   - Diverged from `origin/main` by 1 commit ahead

2. **Remote `origin/main` is CLEAN** (commit 733ded40f)
   - Zero venv files
   - Proper .gitignore in place
   - This is the good version!

3. **Current working branch** (`claude/initial-setup-011CUQcxYK5MCd28FQqkSZrL`)
   - Is clean
   - Only added `current_project_dashboard_v2.py` (673 lines)
   - Based on the good commit (733ded40f)

## Recovery Strategy

### Option 1: HARD RESET Local Main (RECOMMENDED)

This will discard the bad commit and sync with clean origin/main.

```bash
# Backup current state first
git branch backup/main-before-reset-$(date +%Y%m%d-%H%M%S) main

# Reset local main to match origin/main
git checkout main
git reset --hard origin/main

# Verify clean state
git log --oneline -5
git ls-tree -r --long main | awk '$4 > 5000000 {print $4, $5}'  # Should show nothing huge
```

### Option 2: Cherry-pick Good Changes

If there were any good changes in the bad commit:

```bash
# Extract what changed (code only)
git diff origin/main main -- scripts/ Documents/ configs/ > good_changes.patch

# Review the patch
less good_changes.patch

# Reset main
git checkout main
git reset --hard origin/main

# Apply only good changes
git apply good_changes.patch
git add -p  # Selectively stage only code changes
git commit -m "Recover code changes from diverged main"
```

### Option 3: Start Fresh from origin/main

```bash
# Delete local main entirely
git branch -D main

# Recreate from origin
git checkout -b main origin/main

# Set up tracking
git branch --set-upstream-to=origin/main main
```

## Branch Cleanup Plan

### Branches to Keep
- `main` (after reset to origin/main)
- `claude/initial-setup-011CUQcxYK5MCd28FQqkSZrL` (current working branch)

### Branches to Delete
- `backup/before-cleanup-20251025-141128` (temporary backup, keep for now)
- Old codex branches on remote (if not needed):
  - `origin/codex/fix-comments-in-main.py`
  - `origin/codex/fix-face-detection-code`
  - `origin/codex/fix-face-detection-for-clustering`

## Step-by-Step Execution

### Phase 1: Backup Everything
```bash
cd "/Users/eriksjaastad/projects/Eros Mate"

# Create safety backup
git branch backup/main-corrupted-$(date +%Y%m%d-%H%M%S) main

# Create backup of current working state
git stash push -m "Recovery session backup - $(date +%Y%m%d-%H%M%S)"
```

### Phase 2: Fix Local Main Branch
```bash
# Switch to main
git checkout main

# Verify we're on the bad commit
git log --oneline -1
# Should show: 78710b091 Backfill v3 complete...

# HARD RESET to clean origin/main
git reset --hard origin/main

# Verify clean
git log --oneline -1
# Should show: 733ded40f CRITICAL: Remove accidentally committed production data

# Verify no large files
git ls-tree -r --long main | awk '$4 > 10000000 {print $4/1000000 "MB", $5}'
# Should be empty or show only legitimate large files
```

### Phase 3: Clean Working Directory
```bash
# Make sure .gitignore is correct (already updated)
git status

# Should show:
#   M .gitignore  (we improved it)
#   ?? scripts/tools/system_diagnostic.py  (new tool)
#   ?? crop_auto/  (should be ignored - FIX NEEDED)
#   ?? crop_cropped/  (should be ignored - FIX NEEDED)
#   ?? mojo3/  (should be ignored - FIX NEEDED)
```

### Phase 4: Test and Verify
```bash
# Run diagnostic
python scripts/tools/system_diagnostic.py

# Should show:
#   ✓ No large files in main
#   ✓ No .venv files in main

# Check repository size
du -sh .git
# Should be reasonable (< 100 MB)
```

### Phase 5: Push Clean State (CAREFUL!)
```bash
# Verify main is now synced with origin
git status
# Should say: "Your branch is up to date with 'origin/main'"

# If main is now clean and matches origin, we're good
# NO PUSH NEEDED - we're already synced with origin
```

### Phase 6: Merge Claude Branch
```bash
# Now we can safely merge the claude branch changes
git checkout main
git merge --no-ff claude/initial-setup-011CUQcxYK5MCd28FQqkSZrL

# Or create a PR on GitHub for review
```

## Prevention

### Update .gitignore (DONE)
Already added:
```
crop_auto/
crop_cropped/
mojo3/
```

### Add Pre-commit Size Check
Create `.git/hooks/pre-commit-size-check`:
```bash
#!/bin/bash
# Check for accidentally large commits

MAX_SIZE=5000000  # 5 MB

large_files=$(git diff --cached --name-only | while read file; do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ $size -gt $MAX_SIZE ]; then
            echo "$file ($size bytes)"
        fi
    fi
done)

if [ -n "$large_files" ]; then
    echo "❌ Commit blocked: Large files detected"
    echo "$large_files"
    echo ""
    echo "Large files should not be committed. Use .gitignore or Git LFS."
    exit 1
fi
```

## Timeline

- **2025-10-23**: Bad commit 78710b091 created on main
- **2025-10-25**: Claude branch created (clean, based on good commit)
- **2025-10-25**: Diagnostic discovered the issue
- **2025-10-25**: Recovery plan created

## Risk Assessment

- **Low Risk**: origin/main is clean, we can always pull from there
- **No Data Loss**: All code is safe in origin/main and claude branch
- **Easy Recovery**: Simple hard reset will fix everything
- **Backup Exists**: Backup branches created for safety

## Questions for Erik

1. Is there anything in commit 78710b091 that you want to keep?
2. Are the old codex/* branches on origin needed, or can we delete them?
3. Should we merge the claude branch to main now, or keep working on it?

## Execution Status

- [ ] Phase 1: Backup Everything
- [ ] Phase 2: Fix Local Main Branch  
- [ ] Phase 3: Clean Working Directory
- [ ] Phase 4: Test and Verify
- [ ] Phase 5: Verify Sync with Origin
- [ ] Phase 6: Consider Merging Claude Branch

