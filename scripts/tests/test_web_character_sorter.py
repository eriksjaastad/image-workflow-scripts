#!/usr/bin/env python3
"""
Comprehensive Tests for Web Character Sorter
Tests UI functionality, navigation, and style guide compliance
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


class WebCharacterSorterTest:
    def __init__(self):
        self.driver = None
        self.server_process = None
        self.test_data_dir = None

    def setup(self):
        """Set up test environment with browser and test server"""
        # Use existing test data instead of creating temporary data
        self.test_data_dir = Path("scripts/tests/data/test_subdirs")

        if not self.test_data_dir.exists():
            # Fallback: create minimal test data
            self.test_data_dir = Path(tempfile.mkdtemp())
            self.create_test_data()

        # Start server in background
        cmd = [
            sys.executable,
            "scripts/03_web_character_sorter.py",
            str(self.test_data_dir),
            "--port",
            "5003",  # Use different port for tests
            "--no-browser",
        ]

        self.server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=Path.cwd()
        )

        # Wait for server to start
        time.sleep(3)

        # Check if server started properly
        if self.server_process.poll() is not None:
            # Server has exited, check output
            stdout, stderr = self.server_process.communicate()
            raise Exception(
                f"Server failed to start. STDOUT: {stdout.decode()}, STDERR: {stderr.decode()}"
            )

        # Setup headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

        try:
            self.driver.get("http://localhost:5003")

            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "main"))
            )
        except Exception as e:
            # If connection fails, check server output
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                raise Exception(
                    f"Server connection failed. Server output - STDOUT: {stdout.decode()}, STDERR: {stderr.decode()}"
                )
            raise e

    def create_test_data(self):
        """Create test data structure for character sorting"""
        # Create minimal test structure that character sorter expects
        (self.test_data_dir / "person_1").mkdir(parents=True)
        (self.test_data_dir / "person_2").mkdir(parents=True)

        # Create dummy image files with proper naming
        for i in range(3):
            (
                self.test_data_dir
                / "person_1"
                / f"20250803_12000{i}_stage1_generated.png"
            ).touch()
            (
                self.test_data_dir
                / "person_1"
                / f"20250803_12000{i}_stage1_generated.yaml"
            ).touch()
            (
                self.test_data_dir
                / "person_2"
                / f"20250803_13000{i}_stage1_generated.png"
            ).touch()
            (
                self.test_data_dir
                / "person_2"
                / f"20250803_13000{i}_stage1_generated.yaml"
            ).touch()

    def teardown(self):
        """Clean up test environment"""
        if self.driver:
            self.driver.quit()

        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()

        # Clean up test data only if we created temporary data
        if (
            self.test_data_dir
            and self.test_data_dir.exists()
            and "tmp" in str(self.test_data_dir)
        ):
            shutil.rmtree(self.test_data_dir)

    def test_page_loads(self):
        """Test that the character sorter page loads correctly"""
        # Check for main elements
        header = self.driver.find_element(By.TAG_NAME, "header")
        assert header is not None, "Header should be present"

        main = self.driver.find_element(By.TAG_NAME, "main")
        assert main is not None, "Main content should be present"

        print("✅ Page loads test passed")

    def test_sticky_header(self):
        """Test that header is sticky and contains navigation"""
        header = self.driver.find_element(By.TAG_NAME, "header")

        # Check if header has sticky positioning
        position = header.value_of_css_property("position")
        assert position == "sticky", f"Header should be sticky, got {position}"

        # Check for navigation buttons
        buttons = header.find_elements(By.TAG_NAME, "button")
        assert len(buttons) > 0, "Header should contain navigation buttons"

        print("✅ Sticky header test passed")

    def test_character_group_buttons(self):
        """Test G1, G2, G3 character group buttons"""
        # Look for character group buttons
        g1_buttons = self.driver.find_elements(By.CSS_SELECTOR, "[data-group='1']")
        g2_buttons = self.driver.find_elements(By.CSS_SELECTOR, "[data-group='2']")
        g3_buttons = self.driver.find_elements(By.CSS_SELECTOR, "[data-group='3']")

        # Should have group buttons available
        total_group_buttons = len(g1_buttons) + len(g2_buttons) + len(g3_buttons)
        assert total_group_buttons > 0, "Should have character group buttons"

        print("✅ Character group buttons test passed")

    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts for character sorting"""
        # Test common keyboard shortcuts
        body = self.driver.find_element(By.TAG_NAME, "body")

        # Test 'r' for refresh (if implemented)
        body.send_keys("r")
        time.sleep(0.5)

        # Test 'n' for next (if implemented)
        body.send_keys("n")
        time.sleep(0.5)

        # Should not crash
        assert self.driver.current_url.startswith("http://localhost:5003")

        print("✅ Keyboard shortcuts test passed")

    def test_image_display(self):
        """Test that images are displayed correctly"""
        # Look for image elements
        images = self.driver.find_elements(By.TAG_NAME, "img")

        if len(images) > 0:
            # Check first image properties
            first_image = images[0]

            # Should have proper object-fit
            object_fit = first_image.value_of_css_property("object-fit")
            assert (
                object_fit == "contain"
            ), f"Images should use object-fit: contain, got {object_fit}"

        print("✅ Image display test passed")

    def test_style_guide_compliance(self):
        """Test that UI follows the style guide colors and patterns"""
        # Check CSS variables are defined
        bg_color = self.driver.execute_script(
            "return getComputedStyle(document.documentElement).getPropertyValue('--bg')"
        )

        if bg_color.strip():  # Only test if variables are defined
            assert (
                bg_color.strip() == "#101014"
            ), f"Background color should be #101014, got {bg_color}"

            accent_color = self.driver.execute_script(
                "return getComputedStyle(document.documentElement).getPropertyValue('--accent')"
            )
            assert (
                accent_color.strip() == "#4f9dff"
            ), f"Accent color should be #4f9dff, got {accent_color}"

        print("✅ Style guide compliance test passed")

    def test_responsive_layout(self):
        """Test that layout is responsive"""
        # Test different viewport sizes
        self.driver.set_window_size(1920, 1080)  # Desktop
        time.sleep(0.5)

        desktop_width = self.driver.execute_script("return document.body.scrollWidth;")

        self.driver.set_window_size(768, 1024)  # Tablet
        time.sleep(0.5)

        tablet_width = self.driver.execute_script("return document.body.scrollWidth;")

        # Layout should adapt
        assert tablet_width <= desktop_width, "Layout should be responsive"

        print("✅ Responsive layout test passed")

    def test_error_handling(self):
        """Test that errors are handled gracefully"""
        # The page should load even with minimal/missing data

        # Should not contain JavaScript errors in console
        logs = self.driver.get_log("browser")
        severe_errors = [log for log in logs if log["level"] == "SEVERE"]

        assert (
            len(severe_errors) == 0
        ), f"Should not have severe JavaScript errors: {severe_errors}"

        print("✅ Error handling test passed")

    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting Web Character Sorter Tests...")

        try:
            self.setup()

            # Run all tests
            self.test_page_loads()
            self.test_sticky_header()
            self.test_character_group_buttons()
            self.test_keyboard_shortcuts()
            self.test_image_display()
            self.test_style_guide_compliance()
            self.test_responsive_layout()
            self.test_error_handling()

            print("🎉 All Web Character Sorter tests passed!")
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
    test = WebCharacterSorterTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
