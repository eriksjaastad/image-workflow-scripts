#!/usr/bin/env python3
"""
Comprehensive Test Runner for All Web Tools
Runs tests for Web Image Selector, Character Sorter, and Multi-Directory Viewer
"""

import sys
import time
from pathlib import Path

# Add the tests directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from test_web_character_sorter import WebCharacterSorterTest
from test_web_image_selector import WebImageSelectorTest
from test_web_multi_directory_viewer import WebMultiDirectoryViewerTest


class WebToolsTestSuite:
    def __init__(self):
        self.results = {}
    
    def run_web_image_selector_tests(self):
        """Run Web Image Selector tests"""
        print("\n" + "="*60)
        print("🖼️  WEB IMAGE SELECTOR TESTS")
        print("="*60)
        
        test = WebImageSelectorTest()
        success = test.run_all_tests()
        self.results['web_image_selector'] = success
        return success
    
    def run_web_character_sorter_tests(self):
        """Run Web Character Sorter tests"""
        print("\n" + "="*60)
        print("👥 WEB CHARACTER SORTER TESTS")
        print("="*60)
        
        test = WebCharacterSorterTest()
        success = test.run_all_tests()
        self.results['web_character_sorter'] = success
        return success
    
    def run_web_multi_directory_viewer_tests(self):
        """Run Web Multi-Directory Viewer tests"""
        print("\n" + "="*60)
        print("📁 WEB MULTI-DIRECTORY VIEWER TESTS")
        print("="*60)
        
        test = WebMultiDirectoryViewerTest()
        success = test.run_all_tests()
        self.results['web_multi_directory_viewer'] = success
        return success
    
    def run_all_tests(self):
        """Run all web tool tests"""
        print("🚀 Starting Comprehensive Web Tools Test Suite")
        print("Testing all recent enhancements and style guide compliance")
        
        start_time = time.time()
        
        # Run all test suites
        tests = [
            ("Web Image Selector", self.run_web_image_selector_tests),
            ("Web Character Sorter", self.run_web_character_sorter_tests),
            ("Web Multi-Directory Viewer", self.run_web_multi_directory_viewer_tests)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                if success:
                    passed += 1
                else:
                    failed += 1
                    
                # Wait between tests to avoid port conflicts
                time.sleep(2)
                    
            except Exception as e:
                print(f"❌ {test_name} test suite failed with exception: {e}")
                failed += 1
                self.results[test_name.lower().replace(" ", "_")] = False
        
        # Print summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("📊 TEST SUITE SUMMARY")
        print("="*60)
        
        for test_name, success in self.results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nTotal: {passed + failed} test suites")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration:.2f} seconds")
        
        if failed == 0:
            print("\n🎉 All web tool tests passed! Your enhancements are working perfectly!")
        else:
            print(f"\n⚠️  {failed} test suite(s) failed. Check the output above for details.")
        
        return failed == 0
    
    def run_failing_tests_only(self):
        """Run only tests that are expected to fail (for demonstration)"""
        print("🧪 Running Expected Failing Tests (for demonstration)")
        
        # These would be tests that should fail with current implementation
        # For example, testing features not yet implemented
        
        print("📝 Note: All current tests are designed to pass with the implemented features")
        print("   To create failing tests, we would need to test unimplemented features")
        
        return True

def main():
    """Main test runner"""
    suite = WebToolsTestSuite()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--failing":
            success = suite.run_failing_tests_only()
        elif sys.argv[1] == "--image-selector":
            success = suite.run_web_image_selector_tests()
        elif sys.argv[1] == "--character-sorter":
            success = suite.run_web_character_sorter_tests()
        elif sys.argv[1] == "--multi-directory":
            success = suite.run_web_multi_directory_viewer_tests()
        else:
            print("Usage: python test_all_web_tools.py [--failing|--image-selector|--character-sorter|--multi-directory]")
            sys.exit(1)
    else:
        success = suite.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
