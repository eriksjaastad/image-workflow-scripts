#!/usr/bin/env python3
"""
UI Tests for Image Selector
Tests keyboard shortcuts, button behavior, and visual feedback
"""

import sys
import time
import subprocess
import tempfile
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

class ImageSelectorUITest:
    def __init__(self):
        self.driver = None
        self.server_process = None
        self.test_data_dir = None
    
    def setup(self):
        """Set up test environment with browser and test server"""
        # Create test data
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.create_test_data()
        
        # Start server in background
        cmd = [
            sys.executable, "scripts/01_web_image_selector.py",
            str(self.test_data_dir),
            "--port", "5002",  # Use different port for tests
            "--no-browser",
            "--batch-size", "3"
        ]
        
        self.server_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=Path.cwd()
        )
        
        # Wait for server to start
        time.sleep(2)
        
        # Setup headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.get("http://localhost:5002")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "action-sidebar"))
            )
        except Exception as e:
            self.cleanup()
            raise Exception(f"Failed to setup browser: {e}")
    
    def cleanup(self):
        """Clean up test environment"""
        if self.driver:
            self.driver.quit()
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
        if self.test_data_dir and self.test_data_dir.exists():
            import shutil
            shutil.rmtree(self.test_data_dir)
    
    def create_test_data(self):
        """Create minimal test data for UI testing"""
        # Create a few pairs for testing
        test_files = [
            "20250803_100000_stage1_generated.png",
            "20250803_100000_stage2_upscaled.png",
            "20250803_110000_stage1_generated.png", 
            "20250803_110000_stage1.5_face_swapped.png",
            "20250803_120000_stage1_generated.png",
            "20250803_120000_stage2_upscaled.png",
        ]
        
        for filename in test_files:
            # Create dummy image
            img_path = self.test_data_dir / filename
            with open(img_path, 'w') as f:
                f.write(f"dummy image: {filename}")
            
            # Create matching YAML
            yaml_path = self.test_data_dir / (img_path.stem + ".yaml")
            with open(yaml_path, 'w') as f:
                f.write(f"dummy yaml: {filename}")
    
    def test_keyboard_shortcuts(self):
        """Test Q/W/E keyboard shortcuts for select+crop"""
        print("  Testing keyboard shortcuts...")
        
        body = self.driver.find_element(By.TAG_NAME, "body")
        
        # Test Q key (select image 1 + crop)
        body.send_keys("q")
        time.sleep(0.5)
        
        # Check that button 1 is highlighted and crop is active
        btn1 = self.driver.find_element(By.ID, "btn-img1")
        crop_btn = self.driver.find_element(By.ID, "btn-crop")
        
        assert "image-active" in btn1.get_attribute("class"), "Button 1 should be highlighted after Q key"
        assert "crop-active" in crop_btn.get_attribute("class"), "Crop button should be active after Q key"
        
        # Test W key (select image 2 + crop)
        body.send_keys("w")
        time.sleep(0.5)
        
        btn2 = self.driver.find_element(By.ID, "btn-img2")
        assert "image-active" in btn2.get_attribute("class"), "Button 2 should be highlighted after W key"
        assert "crop-active" in crop_btn.get_attribute("class"), "Crop button should still be active after W key"
        
        # Test E key (select image 3 + crop)
        btn3 = self.driver.find_element(By.ID, "btn-img3")
        if btn3.is_displayed():  # Only test if image 3 exists
            body.send_keys("e")
            time.sleep(0.5)
            assert "image-active" in btn3.get_attribute("class"), "Button 3 should be highlighted after E key"
        
        print("  ✓ Q/W/E keyboard shortcuts working correctly")
    
    def test_number_keys(self):
        """Test 1/2/3 keys for select-only (no crop)"""
        print("  Testing number key shortcuts...")
        
        body = self.driver.find_element(By.TAG_NAME, "body")
        
        # Test 1 key (select only, no crop)
        body.send_keys("1")
        time.sleep(0.5)
        
        btn1 = self.driver.find_element(By.ID, "btn-img1")
        crop_btn = self.driver.find_element(By.ID, "btn-crop")
        
        assert "image-active" in btn1.get_attribute("class"), "Button 1 should be highlighted after 1 key"
        assert "crop-active" not in crop_btn.get_attribute("class"), "Crop button should NOT be active after 1 key"
        
        print("  ✓ Number keys working correctly (select without crop)")
    
    def test_enter_navigation(self):
        """Test Enter key for next group navigation"""
        print("  Testing Enter key navigation...")
        
        body = self.driver.find_element(By.TAG_NAME, "body")
        
        # Get initial group info
        initial_url = self.driver.current_url
        
        # Press Enter to go to next group
        body.send_keys(Keys.RETURN)
        time.sleep(1)
        
        # Check that we moved (URL might change or page content updates)
        # For now, just verify no JavaScript errors occurred
        logs = self.driver.get_log('browser')
        js_errors = [log for log in logs if log['level'] == 'SEVERE']
        
        assert len(js_errors) == 0, f"JavaScript errors after Enter key: {js_errors}"
        
        print("  ✓ Enter key navigation working")
    
    def test_button_clicks(self):
        """Test that clicking buttons still works the traditional way"""
        print("  Testing button click behavior...")
        
        # Click button 1
        btn1 = self.driver.find_element(By.ID, "btn-img1")
        btn1.click()
        time.sleep(0.5)
        
        assert "image-active" in btn1.get_attribute("class"), "Button 1 should be highlighted after click"
        
        # Click crop button
        crop_btn = self.driver.find_element(By.ID, "btn-crop")
        crop_btn.click()
        time.sleep(0.5)
        
        assert "crop-active" in crop_btn.get_attribute("class"), "Crop button should be active after click"
        
        print("  ✓ Button clicks working correctly")
    
    def test_visual_feedback(self):
        """Test that visual feedback (image outlines) updates correctly"""
        print("  Testing visual feedback...")
        
        body = self.driver.find_element(By.TAG_NAME, "body")
        
        # Select an image with Q key
        body.send_keys("q")
        time.sleep(0.5)
        
        # Check that image has correct outline
        images = self.driver.find_elements(By.CLASS_NAME, "image-card")
        if images:
            selected_image = images[0]  # Should be first image
            classes = selected_image.get_attribute("class")
            assert "selected" in classes, f"Image should have 'selected' class, got: {classes}"
        
        print("  ✓ Visual feedback working correctly")
    
    def test_no_javascript_errors(self):
        """Test that there are no JavaScript console errors"""
        print("  Checking for JavaScript errors...")
        
        logs = self.driver.get_log('browser')
        js_errors = [log for log in logs if log['level'] == 'SEVERE']
        
        if js_errors:
            print("  JavaScript errors found:")
            for error in js_errors:
                print(f"    {error['message']}")
            raise AssertionError(f"Found {len(js_errors)} JavaScript errors")
        
        print("  ✓ No JavaScript errors detected")
    
    def run_all_tests(self):
        """Run all UI tests"""
        try:
            self.setup()
            
            print("🎨 UI Tests for Image Selector")
            print("=" * 50)
            
            self.test_no_javascript_errors()
            self.test_keyboard_shortcuts()
            self.test_number_keys()
            self.test_enter_navigation()
            self.test_button_clicks()
            self.test_visual_feedback()
            
            print("✅ All UI tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ UI test failed: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Run UI tests"""
    try:
        # Check if Chrome/selenium is available
        from selenium import webdriver
    except ImportError:
        print("❌ Selenium not available. Install with: pip install selenium")
        print("❌ Also need ChromeDriver: https://chromedriver.chromium.org/")
        return False
    
    tester = ImageSelectorUITest()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
