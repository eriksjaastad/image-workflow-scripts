#!/usr/bin/env python3
"""
System Diagnostic Script
Checks the health of the repository, Python environment, and key components.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, capture=True):
    """Run a shell command and return output."""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        else:
            result = subprocess.run(cmd, shell=True, timeout=10)
            return result.returncode, "", ""
    except Exception as e:
        return -1, "", str(e)

def check_python_environment():
    """Check Python environment health."""
    print("\n" + "="*80)
    print("PYTHON ENVIRONMENT")
    print("="*80)
    
    # Python version
    print(f"✓ Python Version: {sys.version}")
    print(f"✓ Python Executable: {sys.executable}")
    
    # Virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"✓ In Virtual Environment: {in_venv}")
    if in_venv:
        print(f"  Location: {sys.prefix}")
    
    # Installed packages
    code, stdout, _ = run_command("pip list --format=freeze | wc -l")
    if code == 0:
        print(f"✓ Installed Packages: {stdout}")
    
    # Key packages
    key_packages = ['pyyaml', 'fastapi', 'uvicorn', 'flask', 'send2trash', 'numpy', 'pandas']
    print("\nKey Packages:")
    for pkg in key_packages:
        code, stdout, _ = run_command(f"pip show {pkg} 2>/dev/null | grep Version")
        if code == 0 and stdout:
            version = stdout.split(': ')[1] if ': ' in stdout else 'installed'
            print(f"  ✓ {pkg}: {version}")
        else:
            print(f"  ✗ {pkg}: NOT INSTALLED")
    
    return True

def check_git_status():
    """Check Git repository health."""
    print("\n" + "="*80)
    print("GIT REPOSITORY")
    print("="*80)
    
    # Current branch
    code, branch, _ = run_command("git branch --show-current")
    if code == 0:
        print(f"✓ Current Branch: {branch}")
    
    # Tracking status
    code, tracking, _ = run_command("git status -sb")
    if code == 0:
        first_line = tracking.split('\n')[0]
        print(f"✓ Status: {first_line}")
    
    # Uncommitted changes
    code, status, _ = run_command("git status --short")
    if code == 0:
        if status:
            print("⚠ Uncommitted Changes:")
            for line in status.split('\n')[:10]:
                print(f"  {line}")
        else:
            print("✓ No Uncommitted Changes")
    
    # Branches
    code, branches, _ = run_command("git branch -a | wc -l")
    if code == 0:
        print(f"✓ Total Branches: {branches}")
    
    # Untracked files
    code, untracked, _ = run_command("git ls-files --others --exclude-standard | wc -l")
    if code == 0:
        count = int(untracked)
        if count > 0:
            print(f"⚠ Untracked Files: {count}")
            code, dirs, _ = run_command("git status --short | grep '^??' | head -10")
            if dirs:
                print("  Sample:")
                for line in dirs.split('\n')[:5]:
                    print(f"    {line}")
        else:
            print("✓ No Untracked Files")
    
    return True

def check_file_system():
    """Check file system and directory structure."""
    print("\n" + "="*80)
    print("FILE SYSTEM")
    print("="*80)
    
    repo_root = Path.cwd()
    print(f"✓ Repository Root: {repo_root}")
    
    # Check key directories
    key_dirs = [
        'scripts',
        'scripts/dashboard',
        'scripts/ai',
        'data',
        'data/training',
        'Documents',
        'sandbox'
    ]
    
    print("\nKey Directories:")
    for dir_path in key_dirs:
        full_path = repo_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
    
    # Check large data directories
    print("\nData Directories (checking .gitignore):")
    data_dirs = ['__crop_auto', '__cropped', 'mojo1', 'mojo2', 'mojo3', 'training data']
    for dir_name in data_dirs:
        full_path = repo_root / dir_name
        if full_path.exists():
            code, result, _ = run_command(f"git check-ignore {dir_name}")
            ignored = "✓ ignored" if code == 0 else "⚠ NOT IGNORED"
            
            # Get size
            code2, size, _ = run_command(f"du -sh '{full_path}' 2>/dev/null | cut -f1")
            size_str = f"({size})" if code2 == 0 else ""
            
            print(f"  {dir_name}: exists {size_str} - {ignored}")
        else:
            print(f"  {dir_name}: not present")
    
    return True

def check_scripts():
    """Check that key scripts are present and executable."""
    print("\n" + "="*80)
    print("KEY SCRIPTS")
    print("="*80)
    
    repo_root = Path.cwd()
    scripts = [
        'scripts/00_start_project.py',
        'scripts/01_ai_assisted_reviewer.py',
        'scripts/01_ai_assisted_reviewer.py',
        'scripts/02_character_processor.py',
        'scripts/03_web_character_sorter.py',
        'scripts/02_ai_desktop_multi_crop.py',
        'scripts/dashboard/current_project_dashboard_v2.py',
    ]
    
    for script in scripts:
        script_path = repo_root / script
        if script_path.exists():
            # Check if it's a Python file and can be parsed
            try:
                with open(script_path, 'r') as f:
                    first_line = f.readline()
                    if first_line.startswith('#!'):
                        print(f"  ✓ {script}")
                    else:
                        print(f"  ✓ {script} (no shebang)")
            except Exception as e:
                print(f"  ⚠ {script} (error reading: {e})")
        else:
            print(f"  ✗ {script} (missing)")
    
    return True

def check_git_branches():
    """Check all git branches for potential issues."""
    print("\n" + "="*80)
    print("BRANCH ANALYSIS")
    print("="*80)
    
    # Local branches
    code, local_branches, _ = run_command(
        "git for-each-ref --sort=-committerdate refs/heads/ --format='%(refname:short)|%(committerdate:short)|%(upstream:short)'"
    )
    
    if code == 0 and local_branches:
        print("\nLocal Branches:")
        for branch in local_branches.split('\n'):
            if branch:
                parts = branch.split('|')
                name = parts[0]
                date = parts[1] if len(parts) > 1 else 'unknown'
                upstream = parts[2] if len(parts) > 2 and parts[2] else 'no upstream'
                print(f"  {name:50s} {date:15s} -> {upstream}")
    
    # Check for large files in branches
    print("\nChecking branches for large committed files...")
    branches_to_check = ['main', 'claude/initial-setup-011CUQcxYK5MCd28FQqkSZrL']
    
    for branch in branches_to_check:
        code, result, _ = run_command(
            f"git ls-tree -r --long {branch} 2>/dev/null | awk '$4 > 5000000 {{print $4, $5}}' | sort -rn | head -5"
        )
        if code == 0 and result:
            print(f"\n  ⚠ Large files in {branch}:")
            for line in result.split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        size_mb = int(parts[0]) / 1_000_000
                        file_path = ' '.join(parts[1:])
                        print(f"    {size_mb:.1f} MB - {file_path}")
        else:
            print(f"  ✓ No large files in {branch}")
    
    return True

def check_pre_commit_hook():
    """Check the pre-commit hook configuration."""
    print("\n" + "="*80)
    print("GIT HOOKS")
    print("="*80)
    
    hook_path = Path('.git/hooks/pre-commit')
    if hook_path.exists():
        print("✓ Pre-commit hook exists")
        with open(hook_path, 'r') as f:
            content = f.read()
            if 'forbidden' in content.lower() or 'banned' in content.lower():
                print("  ⚠ Contains content filtering (this is expected)")
                # Extract forbidden patterns
                if 'forbidden_patterns' in content:
                    print("  Forbidden patterns configured:")
                    for line in content.split('\n'):
                        if 'eros' in line.lower() or '"' in line:
                            print(f"    {line.strip()}")
    else:
        print("✓ No pre-commit hook")
    
    return True

def main():
    """Run all diagnostic checks."""
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "SYSTEM DIAGNOSTIC" + " "*36 + "║")
    print("╚" + "="*78 + "╝")
    
    checks = [
        ("Python Environment", check_python_environment),
        ("Git Repository", check_git_status),
        ("File System", check_file_system),
        ("Scripts", check_scripts),
        ("Git Branches", check_git_branches),
        ("Git Hooks", check_pre_commit_hook),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} {name}")
    
    if all_passed:
        print("\n✓ All diagnostic checks passed!")
        return 0
    else:
        print("\n⚠ Some diagnostic checks failed or have warnings")
        return 1

if __name__ == '__main__':
    sys.exit(main())

