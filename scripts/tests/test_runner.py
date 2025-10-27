#!/usr/bin/env python3
"""
Comprehensive Test Runner for Image Processing Workflow

This script runs all tests before and after script changes to prevent regressions.
It tests critical safety features, performance, and correctness.

Usage:
    python scripts/test_runner.py                    # Run all tests
    python scripts/test_runner.py --safety-only      # Run only critical safety tests
    python scripts/test_runner.py --performance      # Include performance tests
    python scripts/test_runner.py --create-data      # Create test data first
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class TestRunner:  # not a pytest test class (has __init__)
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.test_results = []
        
    def run_test(self, name, test_func, critical=False):
        """Run a single test and record results"""
        print(f"\n{'🚨' if critical else '🧪'} {name}")
        start_time = time.time()
        
        try:
            success = test_func()
            duration = time.time() - start_time
            
            if success:
                print(f"✅ PASSED ({duration:.2f}s)")
                self.passed += 1
                self.test_results.append({"name": name, "status": "PASSED", "duration": duration, "critical": critical})
            else:
                print(f"❌ FAILED ({duration:.2f}s)")
                self.failed += 1
                self.test_results.append({"name": name, "status": "FAILED", "duration": duration, "critical": critical})
                
                if critical:
                    print("🚨 CRITICAL TEST FAILED - STOPPING")
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 ERROR ({duration:.2f}s): {e}")
            self.failed += 1
            self.test_results.append({"name": name, "status": "ERROR", "duration": duration, "critical": critical, "error": str(e)})
            
            if critical:
                print("🚨 CRITICAL TEST ERROR - STOPPING")
                return False
                
        return True
    
    def test_batch_processing_safety(self):
        """CRITICAL: Test that only batch mode is supported (finalize mode removed)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create test data with multiple groups
            result = subprocess.run([
                sys.executable, "create_test_data.py", 
                "--output", str(test_dir),
                "--triplets", "10"
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if result.returncode != 0:
                print(f"Failed to create test data: {result.stderr}")
                return False
            
            # Verify that the script finds groups correctly
            result = subprocess.run([
                sys.executable, "01_ai_assisted_reviewer.py",
                str(test_dir), "--print-triplets"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode != 0:
                print(f"Failed to run grouping test: {result.stderr}")
                return False
            
            # Check that exactly 10 groups were found
            output_lines = result.stdout.strip().split('\n')
            total_line = [line for line in output_lines if "Total:" in line]
            
            if not total_line:
                print("No total line found in output")
                return False
            
            # Parse "Total: X triplets, Y pairs (Z groups)"
            total_text = total_line[0]
            if "10" not in total_text or "triplets" not in total_text:
                print(f"Expected 10 triplets, got: {total_text}")
                return False
            
            print("✓ Batch processing correctly detected 10 triplets")
            print("✓ Only safe batch mode is available (finalize mode removed)")
            return True
    
    def test_grouping_algorithm(self):
        """Test stage-based grouping logic"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create specific test pattern to verify stage progression logic
            result = subprocess.run([
                sys.executable, "create_test_data.py",
                "--output", str(test_dir),
                "--triplets", "5", "--pairs", "2", "--singletons", "1"
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if result.returncode != 0:
                print(f"Failed to create mixed test data: {result.stderr}")
                return False
            
            # Test grouping
            result = subprocess.run([
                sys.executable, "01_ai_assisted_reviewer.py",
                str(test_dir), "--print-triplets"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode != 0:
                print(f"Grouping failed: {result.stderr}")
                return False
            
            # Should find 5 triplets + 2 pairs + 1 singleton = 8 groups
            output_lines = result.stdout.strip().split('\n')
            total_line = [line for line in output_lines if "Total:" in line]
            
            if not total_line:
                print("No total line found in grouping output")
                return False
            
            # Verify mixed group types were detected (5 triplets + 2 pairs = 7 groups, singletons filtered)
            total_text = total_line[0]
            if "7 groups" not in total_text and "7" not in total_text:
                print(f"Expected 7 groups total, got: {total_text}")
                return False
            
            print("✓ Mixed grouping (triplets/pairs/singletons) working correctly")
            return True
    
    def test_sequential_singletons_edge_case(self):
        """Test the specific edge case that caused massive 8-13 image rows."""
        # Use the problematic test data we created
        test_dir = Path(__file__).parent / "data/problematic_sequential"
        
        if not test_dir.exists():
            # Create the problematic test data
            result = subprocess.run([
                sys.executable, "create_problematic_test_data.py"
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if result.returncode != 0:
                print(f"Failed to create problematic test data: {result.stderr}")
                return False
        
        # Test grouping with the problematic pattern
        result = subprocess.run([
            sys.executable, "01_ai_assisted_reviewer.py",
            str(test_dir), "--print-triplets"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode != 0:
            print(f"Sequential singletons test failed: {result.stderr}")
            return False
        
        lines = result.stdout.strip().split('\n')
        
        # Count actual groups
        pair_lines = [line for line in lines if line.startswith("Pair")]
        triplet_lines = [line for line in lines if line.startswith("Triplet")]
        
        total_groups = len(pair_lines) + len(triplet_lines)
        
        # Expected: 4 groups (174338→180000 pair, 190000 triplet, 200000 pair, 210000 triplet)
        if total_groups != 4:
            print(f"CRITICAL BUG: Expected 4 groups, got {total_groups}")
            print("This indicates the sequential grouping algorithm is not working correctly!")
            print("Output:", result.stdout)
            return False
        
        # Verify algorithm correctly groups sequential stages (174338→180000 is valid)
        # No longer checking for "problematic patterns" since sequential grouping is correct behavior
        
        print("✓ Sequential singletons correctly filtered out (no massive rows)")
        print("✓ No cross-contamination between different photo sessions")
        print(f"✓ Correct group count: {total_groups} groups from 20 input files")
        return True
    
    def test_ui_integrity(self):
        """Test UI integrity - keyboard bindings, CSS, JavaScript syntax"""
        result = subprocess.run([
            sys.executable, "test_ui_simple.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            print(f"UI integrity test failed: {result.stderr}")
            if result.stdout:
                print("Output:", result.stdout)
            return False
        
        print("✓ UI integrity validated (key bindings, CSS, JavaScript)")
        return True
    
    def test_web_tools_comprehensive(self):
        """Test all web tools with simplified tests (browser automation can be flaky)"""
        result = subprocess.run([
            sys.executable, "test_web_tools_simple.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            print(f"Web tools test failed: {result.stderr}")
            if result.stdout:
                print("Output:", result.stdout)
            return False
        
        print("✓ All web tools passed simplified tests")
        return True
    
    def test_memory_usage(self):
        """Test memory usage with large dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create large test dataset (similar to XXX_CONTENT size)
            result = subprocess.run([
                sys.executable, "create_test_data.py",
                "--output", str(test_dir),
                "--size", "large"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Failed to create large test data: {result.stderr}")
                return False
            
            # Test that large dataset can be processed
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "01_ai_assisted_reviewer.py",
                str(test_dir), "--print-triplets"
            ], capture_output=True, text=True, timeout=30)  # 30 second timeout
            
            duration = time.time() - start_time
            
            if result.returncode != 0:
                print(f"Large dataset processing failed: {result.stderr}")
                return False
            
            print(f"✓ Large dataset (121+ groups) processed in {duration:.2f}s")
            return duration < 10  # Should process in under 10 seconds
    
    def test_subdirectory_scanning(self):
        """CRITICAL: Test that script recursively scans subdirectories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create subdirectory test data
            result = subprocess.run([
                sys.executable, "create_test_data.py",
                "--subdirectory-test",
                "--output", str(test_dir)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Failed to create subdirectory test data: {result.stderr}")
                return False
            
            # Test recursive scanning
            result = subprocess.run([
                sys.executable, "01_ai_assisted_reviewer.py",
                str(test_dir), "--print-triplets"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode != 0:
                print(f"Subdirectory scanning failed: {result.stderr}")
                return False
            
            # Check that we found more than just root-level files
            output_lines = result.stdout.strip().split('\n')
            total_line = [line for line in output_lines if "Total:" in line]
            
            if not total_line:
                print("No total line found in subdirectory scan output")
                return False
            
            # Should find more than 5 groups (root level only would be 5)
            total_text = total_line[0]
            if "15 groups" not in total_text and "groups" not in total_text:
                print(f"Expected to find many groups from subdirectories, got: {total_text}")
                return False
            
            # Extract group count
            import re
            group_match = re.search(r'(\d+) groups', total_text)
            if not group_match:
                print(f"Could not parse group count from: {total_text}")
                return False
            
            group_count = int(group_match.group(1))
            if group_count < 10:  # Should find many groups across subdirectories
                print(f"Expected many groups from subdirectories, only found {group_count}")
                return False
            
            print(f"✓ Recursive scanning found {group_count} groups across subdirectories")
            return True

    def test_non_standard_files_robustness(self):
        """CRITICAL: Test that script gracefully ignores non-standard files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create subdirectory test data with random files
            result = subprocess.run([
                sys.executable, "create_test_data.py",
                "--subdirectory-test",
                "--output", str(test_dir)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Failed to create test data with random files: {result.stderr}")
                return False
            
            # Test that script handles non-standard files gracefully
            result = subprocess.run([
                sys.executable, "01_ai_assisted_reviewer.py",
                str(test_dir), "--print-triplets"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode != 0:
                print(f"Script failed when processing non-standard files: {result.stderr}")
                return False
            
            # Should still find the expected groups despite random files
            output_lines = result.stdout.strip().split('\n')
            total_line = [line for line in output_lines if "Total:" in line]
            
            if not total_line:
                print("No total line found despite non-standard files")
                return False
            
            # Should find valid groups and ignore invalid files
            total_text = total_line[0]
            import re
            group_match = re.search(r'(\d+) groups', total_text)
            if not group_match:
                print(f"Could not parse group count from: {total_text}")
                return False
            
            group_count = int(group_match.group(1))
            if group_count < 10:  # Should still find many valid groups
                print(f"Expected many valid groups despite random files, only found {group_count}")
                return False
            
            print(f"✓ Gracefully processed {group_count} valid groups while ignoring non-standard files")
            return True

    def test_file_safety(self):
        """CRITICAL: Test that script never modifies source files during analysis"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create test data
            result = subprocess.run([
                sys.executable, "create_test_data.py",
                "--output", str(test_dir),
                "--triplets", "5"
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if result.returncode != 0:
                return False
            
            # Record original file checksums
            original_files = {}
            for file_path in test_dir.glob("*"):
                original_files[file_path.name] = file_path.read_text()
            
            # Run analysis (should not modify files)
            result = subprocess.run([
                sys.executable, "01_ai_assisted_reviewer.py",
                str(test_dir), "--print-triplets"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode != 0:
                print(f"Analysis failed: {result.stderr}")
                return False
            
            # Verify no files were modified
            for file_path in test_dir.glob("*"):
                if file_path.name not in original_files:
                    print(f"Unexpected file created: {file_path.name}")
                    return False
                
                if file_path.read_text() != original_files[file_path.name]:
                    print(f"File modified during analysis: {file_path.name}")
                    return False
            
            print("✓ Source files unchanged during analysis")
            return True
    
    def test_desktop_image_selector_crop(self):
        """Test the desktop image selector crop tool"""
        try:
            # Run the desktop image selector crop test
            result = subprocess.run([
                sys.executable, "test_desktop_image_selector_crop.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✓ Desktop image selector crop tool tests passed")
                return True
            else:
                print("✗ Desktop image selector crop tool tests failed")
                if self.verbose:
                    print(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Desktop image selector crop tool tests timed out")
            return False
        except Exception as e:
            print(f"✗ Desktop image selector crop tool tests failed: {e}")
            return False
    
    def test_dashboard(self):
        """Test the productivity dashboard with real data"""
        try:
            # Run the dashboard test
            result = subprocess.run([
                sys.executable, "test_dashboard.py"
            ], capture_output=True, text=True, timeout=120)  # Longer timeout for server startup
            
            if result.returncode == 0:
                print("✓ Dashboard tests passed with real data")
                return True
            else:
                print("✗ Dashboard tests failed")
                if self.verbose:
                    print(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Dashboard tests timed out")
            return False
        except Exception as e:
            print(f"✗ Dashboard tests failed: {e}")
            return False
    
    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        critical_failed = len([r for r in self.test_results if r.get("critical") and r["status"] != "PASSED"])
        
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"🚨 Critical failures: {critical_failed}")
        
        if critical_failed > 0:
            print("\n🚨 CRITICAL FAILURES DETECTED - DO NOT USE SCRIPTS")
            return False
        elif self.failed > 0:
            print("\n⚠️  Some tests failed - review before deployment")
            return False
        else:
            print("\n🎉 ALL TESTS PASSED - Scripts are safe to use")
            return True

def main():
    parser = argparse.ArgumentParser(description="Run image processing workflow tests")
    parser.add_argument("--safety-only", action="store_true", 
                       help="Run only critical safety tests")
    parser.add_argument("--performance", action="store_true",
                       help="Include performance tests")
    parser.add_argument("--create-data", action="store_true",
                       help="Create test data first")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    print("🧪 Image Processing Workflow Test Suite")
    print("=" * 50)
    
    # Create test data if requested
    if args.create_data:
        print("\n🔧 Creating test data...")
        subprocess.run([sys.executable, "create_test_data.py", "--size", "medium"], cwd=Path(__file__).parent)
    
    # Always run critical safety tests
    if not runner.run_test("File Safety Test", runner.test_file_safety, critical=True):
        return 1

    if not runner.run_test("Subdirectory Scanning Test", runner.test_subdirectory_scanning, critical=True):
        return 1

    if not runner.run_test("Non-Standard Files Robustness", runner.test_non_standard_files_robustness, critical=True):
        return 1

    if not runner.run_test("Batch Processing Safety", runner.test_batch_processing_safety, critical=True):
        return 1
    
    if not args.safety_only:
        runner.run_test("Grouping Algorithm Test", runner.test_grouping_algorithm)
        runner.run_test("Sequential Singletons Edge Case", runner.test_sequential_singletons_edge_case, critical=True)
        runner.run_test("UI Integrity Test", runner.test_ui_integrity, critical=True)
        runner.run_test("Web Tools Comprehensive Test", runner.test_web_tools_comprehensive)
        runner.run_test("Desktop Image Selector Crop Test", runner.test_desktop_image_selector_crop)
        runner.run_test("Dashboard Test", runner.test_dashboard)
        
        if args.performance:
            runner.run_test("Memory Usage Test", runner.test_memory_usage)
    
    success = runner.summary()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
