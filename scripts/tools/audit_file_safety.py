#!/usr/bin/env python3
"""
File Safety Audit - Scan scripts for dangerous file modification patterns
=========================================================================

Scans all Python scripts to find potential file modification operations
that could violate our "read-only except crop" policy.

Usage:
    python scripts/tools/audit_file_safety.py
    python scripts/tools/audit_file_safety.py --verbose
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Dangerous patterns that could modify files
DANGEROUS_PATTERNS = [
    # Writing to files
    (r'open\([^)]+,\s*["\']w', 'open() with write mode'),
    (r'open\([^)]+,\s*["\']wb', 'open() with binary write mode'),
    (r'open\([^)]+,\s*["\']a', 'open() with append mode'),
    (r'\.write\(', 'file.write() call'),
    
    # Image modifications
    (r'Image\.save\(', 'PIL Image.save()'),
    (r'cv2\.imwrite\(', 'OpenCV imwrite()'),
    
    # In-place file operations
    (r'shutil\.copy\([^)]+,\s*[^)]+\)', 'shutil.copy() - check if overwriting'),
    (r'\.rename\(', 'Path.rename() - check if overwriting'),
    
    # YAML/JSON dumps to files
    (r'yaml\.dump\([^)]+,\s*open', 'yaml.dump() to file'),
    (r'json\.dump\([^)]+,\s*open', 'json.dump() to file'),
]

# Files that are ALLOWED to modify files (exceptions)
ALLOWED_FILES = {
    '02_ai_desktop_multi_crop.py',  # Crop tool
    'audit_file_safety.py',  # This script
}

# Safe directories where NEW files can be created
SAFE_DIRECTORIES = {
    'data/ai_data/',
    'data/file_operations_logs/',
    'data/daily_summaries/',
    'data/dashboard_archives/',
    'sandbox/',
}


def scan_file(file_path: Path, verbose: bool = False) -> List[Tuple[int, str, str]]:
    """
    Scan a Python file for dangerous patterns.
    
    Returns:
        List of (line_number, pattern_description, line_content) tuples
    """
    issues = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, start=1):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            # Check each dangerous pattern
            for pattern, description in DANGEROUS_PATTERNS:
                if re.search(pattern, line):
                    # Check if it's in a safe directory context
                    is_safe = any(safe_dir in line for safe_dir in SAFE_DIRECTORIES)
                    
                    if not is_safe or verbose:
                        issues.append((line_num, description, line.strip()))
    
    except Exception as e:
        print(f"[!] Error reading {file_path}: {e}")
    
    return issues


def main():
    verbose = '--verbose' in sys.argv
    
    print("🔍 File Safety Audit")
    print("=" * 70)
    print("Scanning for dangerous file modification patterns...\n")
    
    # Find all Python scripts
    project_root = Path(__file__).parent.parent.parent
    scripts_dir = project_root / 'scripts'
    
    all_issues = {}
    total_files = 0
    files_with_issues = 0
    
    # Scan all Python files
    for py_file in scripts_dir.rglob('*.py'):
        # Skip __pycache__ and test fixtures
        if '__pycache__' in str(py_file) or 'fixtures' in str(py_file):
            continue
        
        total_files += 1
        relative_path = py_file.relative_to(project_root)
        
        # Skip allowed files
        if py_file.name in ALLOWED_FILES:
            if verbose:
                print(f"✅ {relative_path} (ALLOWED)")
            continue
        
        issues = scan_file(py_file, verbose)
        
        if issues:
            files_with_issues += 1
            all_issues[relative_path] = issues
    
    # Print results
    if all_issues:
        print(f"\n⚠️  Found {files_with_issues} files with potential issues:\n")
        
        for file_path, issues in sorted(all_issues.items()):
            print(f"\n📄 {file_path}")
            for line_num, description, line_content in issues:
                print(f"   Line {line_num}: {description}")
                print(f"      {line_content[:80]}...")
        
        print(f"\n" + "=" * 70)
        print(f"Summary:")
        print(f"  • Total files scanned: {total_files}")
        print(f"  • Files with issues: {files_with_issues}")
        print(f"  • Total issues found: {sum(len(issues) for issues in all_issues.values())}")
        print(f"\n⚠️  Review these files to ensure they don't modify production files!")
        print(f"✅ Safe zones: {', '.join(SAFE_DIRECTORIES)}")
    else:
        print(f"✅ No issues found!")
        print(f"   • Scanned {total_files} files")
        print(f"   • All file operations appear safe")
    
    print(f"\n" + "=" * 70)
    
    # Return exit code
    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())

