#!/usr/bin/env python3
"""
Data Recovery Script
Recovers all data files from backup branch that were lost during git reset.

This script extracts files from the backup/main-corrupted-20251025-144705 branch
that are legitimate data files (not venv files) and restores them to the working directory.
"""

import subprocess
import os
from pathlib import Path

BACKUP_BRANCH = "backup/main-corrupted-20251025-144705"
REPO_ROOT = Path("/Users/eriksjaastad/projects/Eros Mate")

# Files and directories to recover
FILES_TO_RECOVER = [
    "data/timesheet.csv",
    "data/character_analysis.json",
    "data/character_analysis_fixed.json",
]

DIRS_TO_RECOVER = [
    "data/aggregates/",
    "data/daily_summaries/",
    "data/crop_progress/",
    "data/sorter_progress/",
    "data/data/snapshot/",  # nested data/data directory
]

def run_command(cmd):
    """Run a command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO_ROOT)
    return result.returncode, result.stdout, result.stderr

def extract_file_from_branch(file_path, branch):
    """Extract a single file from a git branch."""
    print(f"  Recovering: {file_path}")
    
    # Create parent directory if needed
    full_path = REPO_ROOT / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract file from git
    cmd = f'git show {branch}:"{file_path}" > "{full_path}"'
    code, stdout, stderr = run_command(cmd)
    
    if code == 0:
        print(f"    ✓ Recovered")
        return True
    else:
        print(f"    ✗ Failed: {stderr}")
        return False

def extract_directory_from_branch(dir_path, branch):
    """Extract all files in a directory from a git branch."""
    print(f"\n  Recovering directory: {dir_path}")
    
    # Get list of all files in this directory from the branch
    cmd = f'git ls-tree -r --name-only {branch} "{dir_path}"'
    code, stdout, stderr = run_command(cmd)
    
    if code != 0:
        print(f"    ⚠ Directory not found in backup")
        return 0
    
    files = [f for f in stdout.strip().split('\n') if f]
    print(f"    Found {len(files)} files")
    
    recovered = 0
    for file_path in files:
        if extract_file_from_branch(file_path, branch):
            recovered += 1
    
    print(f"    ✓ Recovered {recovered}/{len(files)} files")
    return recovered

def main():
    print("="*80)
    print("DATA RECOVERY FROM BACKUP BRANCH")
    print("="*80)
    print(f"\nBackup branch: {BACKUP_BRANCH}")
    print(f"Repository: {REPO_ROOT}")
    
    # Verify we're in the right place
    if not (REPO_ROOT / ".git").exists():
        print(f"\n✗ ERROR: Not a git repository: {REPO_ROOT}")
        return 1
    
    # Verify backup branch exists
    cmd = f"git rev-parse --verify {BACKUP_BRANCH}"
    code, stdout, stderr = run_command(cmd)
    if code != 0:
        print(f"\n✗ ERROR: Backup branch not found: {BACKUP_BRANCH}")
        return 1
    
    print(f"\n✓ Backup branch found")
    
    # Recover individual files
    print("\n" + "-"*80)
    print("RECOVERING INDIVIDUAL FILES")
    print("-"*80)
    
    recovered_files = 0
    for file_path in FILES_TO_RECOVER:
        if extract_file_from_branch(file_path, BACKUP_BRANCH):
            recovered_files += 1
    
    print(f"\n✓ Recovered {recovered_files}/{len(FILES_TO_RECOVER)} individual files")
    
    # Recover directories
    print("\n" + "-"*80)
    print("RECOVERING DIRECTORIES")
    print("-"*80)
    
    total_dir_files = 0
    for dir_path in DIRS_TO_RECOVER:
        total_dir_files += extract_directory_from_branch(dir_path, BACKUP_BRANCH)
    
    print(f"\n✓ Recovered {total_dir_files} files from directories")
    
    # Summary
    print("\n" + "="*80)
    print("RECOVERY COMPLETE")
    print("="*80)
    print(f"  Individual files: {recovered_files}")
    print(f"  Directory files:  {total_dir_files}")
    print(f"  Total recovered:  {recovered_files + total_dir_files}")
    
    # Verify critical files
    print("\n" + "-"*80)
    print("VERIFICATION")
    print("-"*80)
    
    critical_files = [
        "data/timesheet.csv",
        "data/aggregates/overall/agg_15m_cumulative.jsonl",
        "data/crop_progress/crop_progress.json",
    ]
    
    for file_path in critical_files:
        full_path = REPO_ROOT / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✓ {file_path} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_path} (MISSING)")
    
    print("\n✓ Data recovery complete!")
    print("\nNote: These files are now in your working directory but NOT committed to git.")
    print("This is correct - they should remain ignored by .gitignore for data safety.")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

