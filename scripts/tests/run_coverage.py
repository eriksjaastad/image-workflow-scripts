#!/usr/bin/env python3
"""
Code Coverage Test Runner
Runs all tests with coverage analysis and generates reports.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_coverage():
    """Run tests with coverage and generate reports."""
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    scripts_dir = project_root / "scripts"
    
    print("🔍 Running tests with code coverage...")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Scripts directory: {scripts_dir}")
    
    # Change to project root
    os.chdir(project_root)
    
    # Run coverage on all tests
    print("\n🧪 Running test suite with coverage...")
    result = subprocess.run([
        sys.executable, "-m", "coverage", "run", 
        "--source=scripts",
        "-m", "unittest", "discover", 
        "scripts/tests", "-v"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Tests failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    print("✅ Tests completed successfully!")
    
    # Generate text report
    print("\n📊 Generating coverage report...")
    report_result = subprocess.run([
        sys.executable, "-m", "coverage", "report", "-m"
    ], capture_output=True, text=True)
    
    if report_result.returncode == 0:
        print("📈 Coverage Report:")
        print(report_result.stdout)
    else:
        print("❌ Failed to generate coverage report")
        print(report_result.stderr)
    
    # Generate HTML report
    print("\n🌐 Generating HTML coverage report...")
    html_result = subprocess.run([
        sys.executable, "-m", "coverage", "html"
    ], capture_output=True, text=True)
    
    if html_result.returncode == 0:
        html_dir = project_root / "scripts" / "tests" / "htmlcov"
        print(f"✅ HTML report generated: {html_dir}/index.html")
        print("   Open this file in your browser to view detailed coverage")
    else:
        print("❌ Failed to generate HTML report")
        print(html_result.stderr)
    
    return True

if __name__ == "__main__":
    success = run_coverage()
    sys.exit(0 if success else 1)
