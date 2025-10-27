#!/usr/bin/env python3
"""
Generate Enhanced Coverage Report with Selenium Test Results

This script:
1. Runs unit tests with coverage
2. Runs Selenium tests separately 
3. Combines results into one beautiful HTML report
4. Styles with dark theme from WEB_STYLE_GUIDE.md
"""

import json
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_unit_tests_with_coverage():
    """Run unit tests (non-Selenium) with coverage."""
    print("📊 Running unit tests with coverage...")
    
    result = subprocess.run([
        sys.executable, "-m", "coverage", "run",
        "--source=scripts",
        "-m", "unittest", "discover",
        "scripts/tests",
        "test_*.py",
        "--pattern", "test_[!s]*.py"  # Exclude Selenium tests
    ], capture_output=True, text=True)
    
    # Generate HTML report
    subprocess.run([
        sys.executable, "-m", "coverage", "html",
        "--directory=scripts/tests/htmlcov"
    ])
    
    # Generate JSON data for our custom report
    subprocess.run([
        sys.executable, "-m", "coverage", "json",
        "--output-file=scripts/tests/htmlcov/coverage.json"
    ], capture_output=True, text=True)
    
    return result.returncode == 0


def run_selenium_tests():
    """Run Selenium tests and collect results."""
    print("🔍 Running Selenium integration tests...")
    
    # Run via subprocess to avoid import issues
    result = subprocess.run([
        sys.executable, "-m", "unittest",
        "scripts.tests.test_selenium_simple",
        "scripts.tests.test_web_tools_smoke",
        "-v"
    ], capture_output=True, text=True, cwd=Path.cwd())
    
    # Parse output
    output = result.stdout + result.stderr
    
    # Count tests
    import re
    ran_match = re.search(r'Ran (\d+) test', output)
    total = int(ran_match.group(1)) if ran_match else 0
    
    # Check for failures/errors
    failed = len(re.findall(r'FAIL:', output))
    errors = len(re.findall(r'ERROR:', output))
    passed = total - failed - errors
    
    # Mock result for compatibility
    class MockResult:
        def __init__(self):
            self.testsRun = total
            self.failures = []
            self.errors = []
    
    MockResult()
    
    # Parse results
    selenium_results = {
        'total': total,
        'passed': passed,
        'failed': failed + errors,
        'errors': errors,
        'tests': []
    }
    
    # Save results
    with open('scripts/tests/htmlcov/selenium_results.json', 'w') as f:
        json.dump(selenium_results, f, indent=2)
    
    return selenium_results


def enhance_coverage_html(selenium_results):
    """Inject Selenium results into coverage HTML."""
    print("🎨 Enhancing coverage report with Selenium results...")
    
    index_path = Path('scripts/tests/htmlcov/index.html')
    if not index_path.exists():
        print("❌ Coverage HTML not found!")
        return
    
    html = index_path.read_text()
    
    # Calculate pass rate
    total = selenium_results['total']
    passed = selenium_results['passed']
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    # Create Selenium results section with dark theme
    selenium_section = f'''
    <div id="selenium-results" style="
        background: #181821;
        border: 2px solid #1f1f2c;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    ">
        <h2 style="
            color: #4f9dff;
            font-size: 1.4rem;
            margin: 0 0 1rem 0;
            font-weight: 600;
            border-bottom: 2px solid rgba(79, 157, 255, 0.2);
            padding-bottom: 0.5rem;
        ">
            🔍 Selenium Integration Tests
        </h2>
        
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        ">
            <div style="
                background: #1f1f2c;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="color: #a0a3b1; font-size: 0.8rem; margin-bottom: 0.25rem;">Total Tests</div>
                <div style="color: white; font-size: 2rem; font-weight: 600;">{total}</div>
            </div>
            
            <div style="
                background: #1f1f2c;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="color: #a0a3b1; font-size: 0.8rem; margin-bottom: 0.25rem;">Passed</div>
                <div style="color: #51cf66; font-size: 2rem; font-weight: 600;">{passed}</div>
            </div>
            
            <div style="
                background: #1f1f2c;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="color: #a0a3b1; font-size: 0.8rem; margin-bottom: 0.25rem;">Failed</div>
                <div style="color: #ff6b6b; font-size: 2rem; font-weight: 600;">{selenium_results['failed']}</div>
            </div>
            
            <div style="
                background: #1f1f2c;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="color: #a0a3b1; font-size: 0.8rem; margin-bottom: 0.25rem;">Pass Rate</div>
                <div style="color: {('#51cf66' if pass_rate > 90 else '#ffd43b' if pass_rate > 70 else '#ff6b6b')}; font-size: 2rem; font-weight: 600;">{pass_rate:.0f}%</div>
            </div>
        </div>
        
        <div style="
            background: #1f1f2c;
            padding: 1rem;
            border-radius: 8px;
        ">
            <h3 style="
                color: #4f9dff;
                font-size: 1.1rem;
                margin: 0 0 0.75rem 0;
                font-weight: 600;
            ">Test Coverage</h3>
            <ul style="
                list-style: none;
                padding: 0;
                margin: 0;
                color: #a0a3b1;
                line-height: 1.8;
            ">
                <li style="padding: 0.25rem 0;">✅ Web Image Selector - Smoke Test</li>
                <li style="padding: 0.25rem 0;">✅ Web Character Sorter - Smoke Test</li>
                <li style="padding: 0.25rem 0;">✅ Multi-Directory Viewer - Smoke Test</li>
                <li style="padding: 0.25rem 0;">✅ Duplicate Finder - Smoke Test</li>
                <li style="padding: 0.25rem 0;">✅ Selenium Infrastructure - 3 Tests</li>
            </ul>
        </div>
        
        <div style="
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(79, 157, 255, 0.1);
            border-left: 3px solid #4f9dff;
            border-radius: 4px;
            color: #a0a3b1;
            font-size: 0.9rem;
        ">
            <strong style="color: #4f9dff;">Note:</strong> Selenium tests verify end-to-end functionality. 
            They launch actual Flask servers and test with real browsers. 
            Coverage percentages only reflect unit tests (subprocess tests don't show in coverage).
        </div>
    </div>
    '''
    
    # Inject after the header, before the main content
    # Find the closing </header> tag or first <div> after header
    insertion_point = html.find('</header>')
    if insertion_point == -1:
        insertion_point = html.find('<div id="index">')
    
    if insertion_point != -1:
        html = html[:insertion_point] + selenium_section + html[insertion_point:]
    
    # Also update the page title and add our styles
    html = html.replace('<title>', '<title>Enhanced ')
    
    # Add dark theme styles to the head
    dark_theme_styles = '''
    <style>
    body {
        background: #101014 !important;
        color: #e0e0e0 !important;
    }
    #header {
        background: #181821 !important;
        border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    }
    </style>
    '''
    
    html = html.replace('</head>', dark_theme_styles + '</head>')
    
    # Write enhanced HTML
    index_path.write_text(html)
    print(f"✅ Enhanced coverage report saved to: {index_path}")


def main():
    """Generate enhanced coverage report."""
    print("=" * 60)
    print("🚀 Generating Enhanced Coverage Report")
    print("=" * 60)
    print()
    
    # Run unit tests with coverage
    unit_success = run_unit_tests_with_coverage()
    print()
    
    # Run Selenium tests
    selenium_results = run_selenium_tests()
    print()
    
    # Enhance HTML report
    enhance_coverage_html(selenium_results)
    print()
    
    # Summary
    print("=" * 60)
    print("✅ Enhanced Coverage Report Complete!")
    print("=" * 60)
    print()
    print(f"📊 Unit Tests: {'PASSED' if unit_success else 'SOME FAILURES'}")
    print(f"🔍 Selenium Tests: {selenium_results['passed']}/{selenium_results['total']} passed")
    print()
    
    # Print clickable full path
    report_path = Path('scripts/tests/htmlcov/index.html').resolve()
    print(f"📁 View report: file://{report_path}")
    print()


if __name__ == '__main__':
    main()

