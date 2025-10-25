#!/usr/bin/env python3
"""
Simplified Web Tools Tests
Tests basic functionality without complex browser automation
"""

import sys
import subprocess
import tempfile
import time
from pathlib import Path

class SimpleWebToolsTest:
    def __init__(self):
        self.results = {}
    
    def test_web_image_selector_startup(self):
        """Test that Web Image Selector starts without errors"""
        test_dir = Path(__file__).parent / "data/test_images_medium"
        
        if not test_dir.exists():
            print("⚠️  Test data not found, skipping")
            return True
        
        # Test that the script can start and analyze data
        result = subprocess.run([
            sys.executable, "01_ai_assisted_reviewer.py",
            str(test_dir), "--print-triplets"
        ], capture_output=True, text=True, timeout=10, cwd=Path(__file__).parent.parent)
        
        if result.returncode != 0:
            print(f"❌ Web Image Selector failed: {result.stderr}")
            return False
        
        # Check that it found some groups
        if "Total:" not in result.stdout:
            print("❌ Web Image Selector didn't find any groups")
            return False
        
        print("✅ Web Image Selector startup test passed")
        return True
    
    def test_web_character_sorter_startup(self):
        """Test that Web Character Sorter can start with test data"""
        test_dir = Path(__file__).parent / "data/test_subdirs"
        
        if not test_dir.exists():
            print("⚠️  Test data not found, skipping")
            return True
        
        # Start server in background and test it starts
        process = subprocess.Popen([
            sys.executable, "03_web_character_sorter.py",
            str(test_dir), "--port", "5010", "--no-browser"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=Path(__file__).parent.parent)
        
        try:
            # Wait a bit for startup
            time.sleep(2)
            
            # Check if process is still running (didn't crash)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ Character Sorter crashed: {stderr.decode()}")
                return False
            
            print("✅ Web Character Sorter startup test passed")
            return True
            
        finally:
            process.terminate()
            process.wait()
    
    def test_web_multi_directory_viewer_startup(self):
        """Test that Multi-Directory Viewer can start with test data"""
        test_dir = Path(__file__).parent / "data/test_subdirs"
        
        if not test_dir.exists():
            print("⚠️  Test data not found, skipping")
            return True
        
        # Start server in background and test it starts
        process = subprocess.Popen([
            sys.executable, "05_web_multi_directory_viewer.py",
            str(test_dir), "--port", "5011", "--no-browser"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=Path(__file__).parent.parent)
        
        try:
            # Wait a bit for startup
            time.sleep(2)
            
            # Check if process is still running (didn't crash)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ Multi-Directory Viewer crashed: {stderr.decode()}")
                return False
            
            print("✅ Web Multi-Directory Viewer startup test passed")
            return True
            
        finally:
            process.terminate()
            process.wait()
    
    def test_style_guide_references(self):
        """Test that all web files reference the style guide"""
        web_files = [
            "01_ai_assisted_reviewer.py",
            "03_web_character_sorter.py", 
            "05_web_multi_directory_viewer.py"
        ]
        
        for file_path in web_files:
            full_path = Path(__file__).parent.parent / file_path
            if not full_path.exists():
                print(f"⚠️  {file_path} not found, skipping")
                continue
            
            content = full_path.read_text()
            
            if "WEB_STYLE_GUIDE.md" not in content:
                print(f"❌ {file_path} missing style guide reference")
                return False
            
            if "🎨 STYLE GUIDE:" not in content:
                print(f"❌ {file_path} missing style guide section")
                return False
        
        print("✅ Style guide references test passed")
        return True
    
    def test_recent_enhancements_in_code(self):
        """Test that recent enhancements are present in the code"""
        # Test Web Image Selector enhancements
        selector_content = Path(__file__).parent.parent / "01_ai_assisted_reviewer.py"
        if not selector_content.exists():
            print("⚠️  Web image selector not found, skipping")
            return True
        selector_content = selector_content.read_text()
        
        # Check for unselect functionality
        if "UNSELECT FUNCTIONALITY" not in selector_content:
            print("❌ Unselect functionality not found in Web Image Selector")
            return False
        
        # Check for batch size 100
        if 'default=100' not in selector_content:
            print("❌ Batch size 100 not found in Web Image Selector")
            return False
        
        # Check for simplified workflow (all selections go to selected)
        if "selected: state ? (state.selectedImage !== undefined) : false" not in selector_content:
            print("❌ Simplified selected workflow not found in Web Image Selector")
            return False
        
        # Check for navigation keys
        if "ArrowUp" not in selector_content:
            print("❌ Up arrow navigation not found in Web Image Selector")
            return False
        
        # Test Multi-Directory Viewer enhancements
        viewer_path = Path(__file__).parent.parent / "05_web_multi_directory_viewer.py"
        if viewer_path.exists():
            viewer_content = viewer_path.read_text()
            
            # Check for interactive features
            if "toggleDelete" not in viewer_content:
                print("❌ Delete toggle not found in Multi-Directory Viewer")
                return False
            
            if "toggleCrop" not in viewer_content:
                print("❌ Crop toggle not found in Multi-Directory Viewer")
                return False
            
            # Check for sticky header
            if "toolbar" not in viewer_content:
                print("❌ Sticky header not found in Multi-Directory Viewer")
                return False
        
        print("✅ Recent enhancements test passed")
        return True
    
    def run_all_tests(self):
        """Run all simplified tests"""
        print("🧪 Starting Simplified Web Tools Tests...")
        
        tests = [
            ("Style Guide References", self.test_style_guide_references),
            ("Recent Enhancements in Code", self.test_recent_enhancements_in_code),
            ("Web Image Selector Startup", self.test_web_image_selector_startup),
            ("Web Character Sorter Startup", self.test_web_character_sorter_startup),
            ("Web Multi-Directory Viewer Startup", self.test_web_multi_directory_viewer_startup),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                    self.results[test_name] = "PASSED"
                else:
                    failed += 1
                    self.results[test_name] = "FAILED"
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                failed += 1
                self.results[test_name] = "ERROR"
        
        # Print summary
        print(f"\n📊 SIMPLIFIED TEST SUMMARY")
        print(f"{'='*50}")
        
        for test_name, result in self.results.items():
            status = "✅" if result == "PASSED" else "❌"
            print(f"{status} {test_name}: {result}")
        
        print(f"\nTotal: {passed + failed} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed == 0:
            print("\n🎉 All simplified web tool tests passed!")
        else:
            print(f"\n⚠️  {failed} test(s) failed.")
        
        return failed == 0

def main():
    """Run simplified tests"""
    test = SimpleWebToolsTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
