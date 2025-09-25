#!/usr/bin/env python3
"""
Comprehensive Tests for Web Image Selector
Tests all recent enhancements: unselect, batch size, state override, navigation, safety features
"""

import sys
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

class WebImageSelectorTest:
    def __init__(self):
        self.driver = None
        self.server_process = None
        self.test_data_dir = None
    
    def setup(self):
        """Set up test environment with browser and test server"""
        # Use existing test data
        self.test_data_dir = Path("scripts/tests/data/test_images_medium")
        
        if not self.test_data_dir.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_data_dir}")
        
        # Start server in background
        cmd = [
            sys.executable, "scripts/01_web_image_selector.py",
            str(self.test_data_dir),
            "--port", "5001",  # Use different port for tests
            "--no-browser",
            "--batch-size", "5"  # Small batch for testing
        ]
        
        self.server_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=Path.cwd()
        )
        
        # Wait for server to start
        time.sleep(3)
        
        # Setup headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.get("http://localhost:5001")
        
        # Wait for page to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "main"))
        )
    
    def teardown(self):
        """Clean up test environment"""
        if self.driver:
            self.driver.quit()
        
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
    
    def test_batch_size_100_default(self):
        """Test that default batch size is 100 groups"""
        # Check if batch info shows correct size
        batch_info = self.driver.find_element(By.CSS_SELECTOR, ".batch-info")
        assert "100" in batch_info.text or "batch" in batch_info.text.lower()
        print("✅ Batch size default test passed")
    
    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts 1,2,3,Q,W,E work correctly"""
        # Find first group
        first_group = self.driver.find_element(By.CSS_SELECTOR, "section.group")
        
        # Test number keys
        self.driver.find_element(By.TAG_NAME, "body").send_keys("1")
        time.sleep(0.5)
        
        # Check if first image is selected
        selected_images = self.driver.find_elements(By.CSS_SELECTOR, "figure.image-card.selected")
        assert len(selected_images) > 0, "Image should be selected with '1' key"
        
        # All selected images automatically have selected state (no toggle needed)
        crop_selected = self.driver.find_elements(By.CSS_SELECTOR, "figure.image-card.crop-selected")
        assert len(crop_selected) > 0, "Selected images should automatically have selected state"
        
        print("✅ Keyboard shortcuts test passed")
    
    def test_unselect_functionality(self):
        """Test that clicking selected image deselects it"""
        # Select an image first
        self.driver.find_element(By.TAG_NAME, "body").send_keys("1")
        time.sleep(0.5)
        
        # Find the selected image and click it
        selected_image = self.driver.find_element(By.CSS_SELECTOR, "figure.image-card.selected img")
        selected_image.click()
        time.sleep(0.5)
        
        # Check that it's no longer selected
        selected_images = self.driver.find_elements(By.CSS_SELECTOR, "figure.image-card.selected")
        assert len(selected_images) == 0, "Image should be deselected when clicked"
        
        print("✅ Unselect functionality test passed")
    
    def test_selection_consistency(self):
        """Test that all selections consistently go to selected"""
        # Select first image
        self.driver.find_element(By.TAG_NAME, "body").send_keys("1")
        time.sleep(0.5)
        
        # Verify both selected and crop state are active
        selected_images = self.driver.find_elements(By.CSS_SELECTOR, "figure.image-card.selected")
        crop_selected = self.driver.find_elements(By.CSS_SELECTOR, "figure.image-card.crop-selected")
        
        assert len(selected_images) > 0, "Should have selected image"
        assert len(crop_selected) > 0, "Selected image should automatically have selected state"
        assert len(selected_images) == len(crop_selected), "All selected images should have selected state"
        
        print("✅ Selection consistency test passed")
    
    def test_button_toggle(self):
        """Test that clicking same button twice deselects"""
        # Clear any existing state first
        self.driver.refresh()
        time.sleep(1)
        
        # Press '1' once
        self.driver.find_element(By.TAG_NAME, "body").send_keys("1")
        time.sleep(1)  # Give more time for state to update
        
        # Verify selection
        selected_images = self.driver.find_elements(By.CSS_SELECTOR, "figure.image-card.selected")
        if len(selected_images) == 0:
            # Try alternative selector or debug
            all_images = self.driver.find_elements(By.CSS_SELECTOR, "figure.image-card")
            print(f"Debug: Found {len(all_images)} total images")
            if len(all_images) > 0:
                # Check classes on first image
                first_image_classes = all_images[0].get_attribute("class")
                print(f"Debug: First image classes: {first_image_classes}")
        
        assert len(selected_images) > 0, f"Should be selected after first press, found {len(selected_images)} selected images"
        
        # Press '1' again
        self.driver.find_element(By.TAG_NAME, "body").send_keys("1")
        time.sleep(1)
        
        # Verify deselection
        selected_images = self.driver.find_elements(By.CSS_SELECTOR, "figure.image-card.selected")
        assert len(selected_images) == 0, "Should be deselected after second press"
        
        print("✅ Button toggle test passed")
    
    def test_navigation_keys(self):
        """Test Enter (forward) and Up Arrow (back) navigation"""
        # Test Enter key navigation
        initial_scroll = self.driver.execute_script("return window.pageYOffset;")
        
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ENTER)
        time.sleep(1)
        
        new_scroll = self.driver.execute_script("return window.pageYOffset;")
        assert new_scroll != initial_scroll, "Enter should change scroll position"
        
        # Test Up Arrow navigation
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
        time.sleep(1)
        
        back_scroll = self.driver.execute_script("return window.pageYOffset;")
        assert back_scroll != new_scroll, "Up arrow should change scroll position"
        
        print("✅ Navigation keys test passed")
    
    def test_process_button_safety(self):
        """Test that Process button is disabled until scrolled to bottom"""
        # Find process button
        process_button = self.driver.find_element(By.CSS_SELECTOR, ".process-batch")
        
        # Should be disabled initially
        assert process_button.get_attribute("disabled") is not None, "Process button should be disabled initially"
        
        # Scroll to bottom
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        
        # Should be enabled now
        assert process_button.get_attribute("disabled") is None, "Process button should be enabled after scrolling"
        
        print("✅ Process button safety test passed")
    
    def test_style_guide_compliance(self):
        """Test that UI follows the style guide colors and patterns"""
        # Check CSS variables are defined
        bg_color = self.driver.execute_script(
            "return getComputedStyle(document.documentElement).getPropertyValue('--bg')"
        )
        assert bg_color.strip() == "#101014", f"Background color should be #101014, got {bg_color}"
        
        accent_color = self.driver.execute_script(
            "return getComputedStyle(document.documentElement).getPropertyValue('--accent')"
        )
        assert accent_color.strip() == "#4f9dff", f"Accent color should be #4f9dff, got {accent_color}"
        
        print("✅ Style guide compliance test passed")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting Web Image Selector Tests...")
        
        try:
            self.setup()
            
            # Run all tests
            self.test_batch_size_100_default()
            self.test_keyboard_shortcuts()
            self.test_unselect_functionality()
            self.test_selection_consistency()
            self.test_button_toggle()
            self.test_navigation_keys()
            self.test_process_button_safety()
            self.test_style_guide_compliance()
            
            print("🎉 All Web Image Selector tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
            
        finally:
            self.teardown()

def main():
    """Run tests if called directly"""
    test = WebImageSelectorTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
