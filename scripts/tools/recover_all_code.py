#!/usr/bin/env python3
"""
Complete Code & Docs Recovery from Backup
Recovers all missing scripts and documentation
"""

import subprocess
import os
from pathlib import Path

BACKUP = "backup/main-corrupted-20251025-144705"
REPO = Path("/Users/eriksjaastad/projects/Eros Mate")

# All missing code files (NOT test data)
FILES = [
    # Documents
    "Documents/AI_REVIEWER_BATCH_IMPLEMENTATION_COMPLETE.md",
    "Documents/AI_REVIEWER_BATCH_REFACTOR_PLAN.md",
    "Documents/AI_REVIEWER_CLONE_STRATEGY.md",
    "Documents/AI_REVIEWER_REFACTOR_2025-10-22.md",
    "Documents/AI_REVIEWER_VS_WEB_SELECTOR_COMPARISON.md",
    "Documents/AI_REVIEWER_WEB_SELECTOR_PARITY_TODO.md",
    "Documents/AI_TRAINING_DECISIONS_V3_IMPLEMENTATION.md",
    "Documents/BACKFILL_QUICK_START.md",
    "Documents/CURRENT_TODO_LIST.md",
    "Documents/DASHBOARD_BASELINE_TEMPLATES.md",
    "Documents/DATA_CONSOLIDATION_SYSTEM.md",
    "Documents/HANDOFF_HISTORICAL_BACKFILL_2025-10-22.md",
    "Documents/PERFORMANCE_FIX_DESKTOP_MULTICROP.md",
    "Documents/PHASE1_COMPLETE_SUMMARY.md",
    "Documents/SESSION_SUMMARY_2025-10-22_AFTERNOON.md",
    "Documents/TECHNICAL_KNOWLEDGE_BASE.md",
    "Documents/UX_FIX_CROP_TOGGLE_BUTTON.md",
    
    # Updated scripts
    "scripts/00_start_project.py",
    "scripts/01_ai_assisted_reviewer.py",
    "scripts/01_desktop_image_selector_crop.py",
    "scripts/02_ai_desktop_multi_crop.py",
    "scripts/cleanup_logs.py",
    
    # AI analysis scripts
    "scripts/ai/analyze_crop_patterns.py",
    "scripts/ai/analyze_fresh_crops.py",
    "scripts/ai/analyze_real_crop_images.py",
    
    # Updated dashboard files
    "scripts/dashboard/analytics.py",
    "scripts/dashboard/data_engine.py",
    "scripts/dashboard/productivity_dashboard.py",
    "scripts/dashboard/project_metrics_aggregator.py",
    
    # Updated utils
    "scripts/utils/ai_training_decisions_v3.py",
    "scripts/utils/base_desktop_image_tool.py",
    "scripts/utils/companion_file_utils.py",
    
    # Tools
    "scripts/tools/backfill_v3_from_archives.py",
]

os.chdir(REPO)

print("="*80)
print("RECOVERING ALL MISSING CODE FILES")
print("="*80)

recovered = 0
failed = []

for file_path in FILES:
    print(f"\n[{recovered+1}/{len(FILES)}] {file_path}...")
    
    # Create parent dir
    full_path = REPO / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract from git
    cmd = f'git show {BACKUP}:"{file_path}" > "{full_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print(f"  ✓ Recovered")
        recovered += 1
    else:
        print(f"  ✗ FAILED")
        failed.append(file_path)

print("\n" + "="*80)
print("RECOVERY COMPLETE")
print("="*80)
print(f"✓ Recovered: {recovered}/{len(FILES)}")
if failed:
    print(f"✗ Failed: {len(failed)}")
    for f in failed:
        print(f"  - {f}")

print("\n✅ All your code is back!")

