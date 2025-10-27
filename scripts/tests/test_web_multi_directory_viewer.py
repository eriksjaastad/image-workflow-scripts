#!/usr/bin/env python3
"""
Comprehensive Tests for Web Multi-Directory Viewer
Tests crop/delete functionality, sticky header with live stats, and style guide compliance
"""

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class WebMultiDirectoryViewerTest:
    def __init__(self):
        self.driver = None
        self.server_process = None
        self.test_data_dir = None
    
    def setup(self):
        """Set up test environment with browser and test server"""
        # Create test data structure
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.create_test_data()
        
        # Start server in background
        cmd = [
            sys.executable, "scripts/05_web_multi_directory_viewer.py",
            str(self.test_data_dir),
            "--port", "5004",  # Use different port for tests
            "--no-browser"
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
        self.driver.get("http://localhost:5004")
        
        # Wait for page to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "header"))
        )
    
    def create_test_data(self):
        """Create test data structure for multi-directory viewing"""
        # Create subdirectories with images (nested structure)
        group1_dir = self.test_data_dir / "group_1"
        group2_dir = self.test_data_dir / "group_2"
        
        group1_dir.mkdir(parents=True)
        group2_dir.mkdir(parents=True)
        
        # Create dummy image files
        for i in range(3):
            (group1_dir / f"image_{i}.png").touch()
            (group1_dir / f"image_{i}.yaml").touch()
            (group2_dir / f"image_{i}.png").touch()
            (group2_dir / f"image_{i}.yaml").touch()
        
        # Also test flat structure
        for i in range(2):
            (self.test_data_dir / f"flat_image_{i}.png").touch()
            (self.test_data_dir / f"flat_image_{i}.yaml").touch()
    
    def teardown(self):
        """Clean up test environment"""
        if self.driver:
            self.driver.quit()
        
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
        
        # Clean up test data
        if self.test_data_dir and self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
    
    def test_page_loads(self):
        """Test that the multi-directory viewer page loads correctly"""
        # Check for main elements
        header = self.driver.find_element(By.TAG_NAME, "header")
        assert header is not None, "Header should be present"
        
        # Check for directory sections or images
        directory_sections = self.driver.find_elements(By.CSS_SELECTOR, ".directory-section")
        images = self.driver.find_elements(By.TAG_NAME, "img")
        
        assert len(directory_sections) > 0 or len(images) > 0, "Should display directories or images"
        
        print("✅ Page loads test passed")
    
    def test_sticky_header_with_stats(self):
        """Test sticky header with live selection statistics"""
        header = self.driver.find_element(By.TAG_NAME, "header")
        
        # Check if header has sticky positioning
        position = header.value_of_css_property("position")
        assert position == "sticky", f"Header should be sticky, got {position}"
        
        # Check for selection stats
        delete_count = self.driver.find_element(By.ID, "deleteCount")
        crop_count = self.driver.find_element(By.ID, "cropCount")
        
        assert delete_count is not None, "Should have delete count display"
        assert crop_count is not None, "Should have crop count display"
        
        # Initial counts should be 0
        assert "0 delete" in delete_count.text, f"Initial delete count should be 0, got {delete_count.text}"
        assert "0 crop" in crop_count.text, f"Initial crop count should be 0, got {crop_count.text}"
        
        print("✅ Sticky header with stats test passed")
    
    def test_three_column_header_layout(self):
        """Test that header has proper three-column layout"""
        header = self.driver.find_element(By.TAG_NAME, "header")
        
        # Check for three sections
        toolbar_left = header.find_element(By.CSS_SELECTOR, ".toolbar-left")
        toolbar_center = header.find_element(By.CSS_SELECTOR, ".toolbar-center")
        toolbar_right = header.find_element(By.CSS_SELECTOR, ".toolbar-right")
        
        assert toolbar_left is not None, "Should have left toolbar section"
        assert toolbar_center is not None, "Should have center toolbar section"
        assert toolbar_right is not None, "Should have right toolbar section"
        
        # Check that center contains stats
        stats = toolbar_center.find_element(By.ID, "selectionStats")
        assert stats is not None, "Center should contain selection stats"
        
        # Check that right contains process button
        process_button = toolbar_right.find_element(By.ID, "processButton")
        assert process_button is not None, "Right should contain process button"
        
        print("✅ Three-column header layout test passed")
    
    def test_image_click_delete_toggle(self):
        """Test that clicking images toggles delete state"""
        # Find first image
        images = self.driver.find_elements(By.CSS_SELECTOR, ".image-container")
        
        if len(images) > 0:
            first_image = images[0]
            
            # Click to select for delete
            first_image.click()
            time.sleep(0.5)
            
            # Check for delete state
            delete_selected = self.driver.find_elements(By.CSS_SELECTOR, ".image-container.delete-selected")
            assert len(delete_selected) > 0, "Image should have delete state after click"
            
            # Check that delete count updated
            delete_count = self.driver.find_element(By.ID, "deleteCount")
            assert "1 delete" in delete_count.text, f"Delete count should be 1, got {delete_count.text}"
            
            # Click again to deselect
            first_image.click()
            time.sleep(0.5)
            
            # Check that delete state is cleared
            delete_selected = self.driver.find_elements(By.CSS_SELECTOR, ".image-container.delete-selected")
            assert len(delete_selected) == 0, "Image should not have delete state after second click"
            
            # Check that delete count reset
            delete_count = self.driver.find_element(By.ID, "deleteCount")
            assert "0 delete" in delete_count.text, f"Delete count should be 0, got {delete_count.text}"
        
        print("✅ Image click delete toggle test passed")
    
    def test_crop_button_functionality(self):
        """Test crop button toggles crop state"""
        # Find first crop button
        crop_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".crop-button")
        
        if len(crop_buttons) > 0:
            first_crop_button = crop_buttons[0]
            
            # Click crop button
            first_crop_button.click()
            time.sleep(0.5)
            
            # Check for crop state
            crop_selected = self.driver.find_elements(By.CSS_SELECTOR, ".image-container.crop-selected")
            assert len(crop_selected) > 0, "Image should have crop state after crop button click"
            
            # Check that crop count updated
            crop_count = self.driver.find_element(By.ID, "cropCount")
            assert "1 crop" in crop_count.text, f"Crop count should be 1, got {crop_count.text}"
            
            # Click again to deselect
            first_crop_button.click()
            time.sleep(0.5)
            
            # Check that crop state is cleared
            crop_selected = self.driver.find_elements(By.CSS_SELECTOR, ".image-container.crop-selected")
            assert len(crop_selected) == 0, "Image should not have crop state after second click"
            
            # Check that crop count reset
            crop_count = self.driver.find_element(By.ID, "cropCount")
            assert "0 crop" in crop_count.text, f"Crop count should be 0, got {crop_count.text}"
        
        print("✅ Crop button functionality test passed")
    
    def test_state_override(self):
        """Test that delete and crop states override each other"""
        images = self.driver.find_elements(By.CSS_SELECTOR, ".image-container")
        crop_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".crop-button")
        
        if len(images) > 0 and len(crop_buttons) > 0:
            first_image = images[0]
            first_crop_button = crop_buttons[0]
            
            # First select for delete
            first_image.click()
            time.sleep(0.5)
            
            # Verify delete state
            delete_selected = self.driver.find_elements(By.CSS_SELECTOR, ".image-container.delete-selected")
            assert len(delete_selected) > 0, "Should have delete state"
            
            # Then select for crop
            first_crop_button.click()
            time.sleep(0.5)
            
            # Verify crop state and delete state cleared
            crop_selected = self.driver.find_elements(By.CSS_SELECTOR, ".image-container.crop-selected")
            delete_selected = self.driver.find_elements(By.CSS_SELECTOR, ".image-container.delete-selected")
            
            assert len(crop_selected) > 0, "Should have crop state"
            assert len(delete_selected) == 0, "Delete state should be cleared"
            
            # Test reverse: crop then delete
            first_image.click()
            time.sleep(0.5)
            
            # Verify delete state and crop state cleared
            crop_selected = self.driver.find_elements(By.CSS_SELECTOR, ".image-container.crop-selected")
            delete_selected = self.driver.find_elements(By.CSS_SELECTOR, ".image-container.delete-selected")
            
            assert len(delete_selected) > 0, "Should have delete state"
            assert len(crop_selected) == 0, "Crop state should be cleared"
        
        print("✅ State override test passed")
    
    def test_image_aspect_ratios(self):
        """Test that images maintain proper aspect ratios"""
        images = self.driver.find_elements(By.CSS_SELECTOR, ".image-container img")
        
        if len(images) > 0:
            first_image = images[0]
            
            # Check object-fit property
            object_fit = first_image.value_of_css_property("object-fit")
            assert object_fit == "contain", f"Images should use object-fit: contain, got {object_fit}"
            
            # Check max-width and max-height
            max_width = first_image.value_of_css_property("max-width")
            max_height = first_image.value_of_css_property("max-height")
            
            assert max_width == "100%", f"Images should have max-width: 100%, got {max_width}"
            assert "300px" in max_height, f"Images should have max-height: 300px, got {max_height}"
        
        print("✅ Image aspect ratios test passed")
    
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
        
        danger_color = self.driver.execute_script(
            "return getComputedStyle(document.documentElement).getPropertyValue('--danger')"
        )
        assert danger_color.strip() == "#ff6b6b", f"Danger color should be #ff6b6b, got {danger_color}"
        
        print("✅ Style guide compliance test passed")
    
    def test_process_button_functionality(self):
        """Test that process button works correctly"""
        process_button = self.driver.find_element(By.ID, "processButton")
        
        # Button should be present and clickable
        assert process_button is not None, "Process button should be present"
        assert process_button.is_enabled(), "Process button should be enabled"
        
        # Click without selections should show error
        process_button.click()
        time.sleep(1)
        
        # Should show status message (though we can't easily test the actual processing)
        status_bar = self.driver.find_element(By.ID, "statusBar")
        assert status_bar is not None, "Status bar should be present"
        
        print("✅ Process button functionality test passed")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting Web Multi-Directory Viewer Tests...")
        
        try:
            self.setup()
            
            # Run all tests
            self.test_page_loads()
            self.test_sticky_header_with_stats()
            self.test_three_column_header_layout()
            self.test_image_click_delete_toggle()
            self.test_crop_button_functionality()
            self.test_state_override()
            self.test_image_aspect_ratios()
            self.test_style_guide_compliance()
            self.test_process_button_functionality()
            
            print("🎉 All Web Multi-Directory Viewer tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.teardown()

def main():
    """Run tests if called directly"""
    test = WebMultiDirectoryViewerTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
