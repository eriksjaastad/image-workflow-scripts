#!/usr/bin/env python3
"""
Test File Safety Audit - Ensure no dangerous file modification patterns exist
==============================================================================

This test runs the file safety audit and fails if dangerous patterns are found
outside of approved safe zones.

Usage:
    pytest scripts/tests/test_file_safety_audit.py
    python scripts/tests/test_file_safety_audit.py
"""

import sys
import unittest
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the audit script
from tools import audit_file_safety


class TestFileSafetyAudit(unittest.TestCase):
    """Test that no dangerous file modification patterns exist."""
    
    def test_no_dangerous_file_modifications(self):
        """
        Run file safety audit and ensure all file modifications are in safe zones.
        
        This test scans all Python scripts for dangerous patterns like:
        - Writing to existing files outside safe zones
        - Image modifications outside crop tool
        - YAML/caption overwrites
        """
        print("\n🔍 Running File Safety Audit...")
        
        # Run the audit (captures issues)
        project_root = Path(__file__).parent.parent.parent
        scripts_dir = project_root / 'scripts'
        
        all_issues = {}
        files_with_issues = 0
        
        # Scan all Python files
        for py_file in scripts_dir.rglob('*.py'):
            # Skip __pycache__ and test fixtures
            if '__pycache__' in str(py_file) or 'fixtures' in str(py_file):
                continue
            
            relative_path = py_file.relative_to(project_root)
            
            # Skip allowed files
            if py_file.name in audit_file_safety.ALLOWED_FILES:
                continue
            
            issues = audit_file_safety.scan_file(py_file, verbose=False)
            
            if issues:
                files_with_issues += 1
                all_issues[relative_path] = issues
        
        # Check if any issues are OUTSIDE safe zones
        dangerous_issues = {}
        
        for file_path, issues in all_issues.items():
            dangerous = []
            for line_num, description, line_content in issues:
                # Check if it's in a safe zone
                is_safe = any(safe_dir in line_content for safe_dir in audit_file_safety.SAFE_DIRECTORIES)
                
                # Also check if it's a progress file or logging
                is_logging = 'log' in line_content.lower() or 'progress' in line_content.lower()
                is_data_dir = 'data/' in line_content
                
                # If not safe, it's dangerous
                if not (is_safe or is_logging or is_data_dir):
                    dangerous.append((line_num, description, line_content))
            
            if dangerous:
                dangerous_issues[file_path] = dangerous
        
        # Print results
        if dangerous_issues:
            print("\n⚠️  DANGEROUS FILE MODIFICATIONS FOUND:")
            for file_path, issues in dangerous_issues.items():
                print(f"\n📄 {file_path}")
                for line_num, description, line_content in issues:
                    print(f"   Line {line_num}: {description}")
                    print(f"      {line_content[:80]}...")
            
            print("\n" + "="*70)
            print("❌ File safety test FAILED!")
            print(f"   Found {len(dangerous_issues)} files with dangerous patterns")
            print(f"   Review these files to ensure they don't modify production files")
            print("\nSafe zones:")
            for safe_dir in audit_file_safety.SAFE_DIRECTORIES:
                print(f"   • {safe_dir}")
            
            self.fail(f"Found {len(dangerous_issues)} files with potentially dangerous file modifications")
        
        # Test passes - print success message
        print(f"\n✅ File safety audit PASSED!")
        print(f"   • Scanned all scripts in {scripts_dir}")
        print(f"   • All file operations appear safe")
        print(f"   • No dangerous patterns found outside safe zones")


def main():
    """Run test standalone."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFileSafetyAudit)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

