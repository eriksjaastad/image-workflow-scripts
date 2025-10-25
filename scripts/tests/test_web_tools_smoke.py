"""
Smoke tests for web tools - verifies each tool can start and basic UI loads.

These tests use subprocess to start the actual tools as they would be used in production,
then use Selenium to verify the UI loads correctly.

Focus: Can the tool start? Does the page load? Are key elements present?
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from PIL import Image

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class WebToolSmokeTest(unittest.TestCase):
    """Base class for web tool smoke tests."""
    
    driver = None
    process = None
    
    @classmethod
    def setUpClass(cls):
        """Set up Chrome driver once."""
        chrome_options = ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        service = ChromeService(ChromeDriverManager().install())
        cls.driver = webdriver.Chrome(service=service, options=chrome_options)
        cls.driver.implicitly_wait(5)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up Chrome driver."""
        if cls.driver:
            cls.driver.quit()
    
    def setUp(self):
        """Create temporary test directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        os.environ['EM_TEST_DATA_ROOT'] = str(self.temp_path)
    
    def tearDown(self):
        """Clean up process and temp directory."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            # Close pipes to prevent ResourceWarning
            if self.process.stdout:
                self.process.stdout.close()
            if self.process.stderr:
                self.process.stderr.close()
            self.process = None
        
        if 'EM_TEST_DATA_ROOT' in os.environ:
            del os.environ['EM_TEST_DATA_ROOT']
        
        if hasattr(self, 'temp_dir') and self.temp_dir:
            self.temp_dir.cleanup()
    
    def wait_for_server(self, port, timeout=10):
        """Wait for server to start accepting connections."""
        import socket
        start = time.time()
        while time.time() - start < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                if result == 0:
                    time.sleep(0.5)  # Extra time to initialize
                    return True
            except:
                pass
            time.sleep(0.2)
        return False


class TestWebImageSelectorSmoke(WebToolSmokeTest):
    """Smoke tests for web image selector."""
    
    def create_test_images(self):
        """Create test images for selector."""
        test_dir = self.temp_path / "test_images"
        test_dir.mkdir()
        
        # Create 2 groups of 3 images each with proper naming
        # Format: YYYYMMDD_HHMMSS_stageN_descriptor.png
        descriptors = {1: "generated", 2: "upscaled", 3: "enhanced"}
        for group_id in range(2):
            timestamp = f"20250101_{group_id:02d}0000"
            for stage in [1, 2, 3]:
                descriptor = descriptors[stage]
                img_path = test_dir / f"{timestamp}_stage{stage}_{descriptor}.png"
                img = Image.new('RGB', (200, 200), color=(stage * 80, group_id * 50, 100))
                img.save(img_path)
        
        return test_dir
    
    def test_web_image_selector_starts(self):
        """Test that web image selector can start and page loads."""
        test_dir = self.create_test_images()
        
        # Start the web image selector
        script_path = Path(__file__).parent.parent / "01_ai_assisted_reviewer.py"
        port = 8765
        
        self.process = subprocess.Popen(
            [sys.executable, str(script_path), str(test_dir), 
             "--port", str(port), "--no-browser", "--batch-size", "10"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Wait for server to start
        if not self.wait_for_server(port):
            # Get error output
            stdout, stderr = self.process.communicate(timeout=1)
            error_msg = f"Server did not start. STDOUT: {stdout.decode()[:500]}, STDERR: {stderr.decode()[:500]}"
            self.fail(error_msg)
        
        # Navigate to the page
        self.driver.get(f"http://127.0.0.1:{port}")
        
        # Verify page loads
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Check title
        self.assertIn("Image", self.driver.title)
        self.assertIn("Selector", self.driver.title)
        
        # Verify images are present
        images = self.driver.find_elements(By.TAG_NAME, "img")
        self.assertGreater(len(images), 0, "No images found on page")
        
        # Verify action buttons exist
        buttons = self.driver.find_elements(By.CLASS_NAME, "action-btn")
        self.assertGreater(len(buttons), 0, "No action buttons found")


class TestWebCharacterSorterSmoke(WebToolSmokeTest):
    """Smoke tests for web character sorter."""
    
    def create_test_images(self):
        """Create test images for character sorter."""
        test_dir = self.temp_path / "test_chars"
        test_dir.mkdir()
        
        # Create 5 test images
        for i in range(5):
            img_path = test_dir / f"character_{i:03d}.png"
            img = Image.new('RGB', (200, 200), color=(i * 50, 100, 150))
            img.save(img_path)
        
        return test_dir
    
    def test_web_character_sorter_starts(self):
        """Test that web character sorter can start and page loads."""
        test_dir = self.create_test_images()
        
        # Start the character sorter
        script_path = Path(__file__).parent.parent / "03_web_character_sorter.py"
        port = 8766
        
        self.process = subprocess.Popen(
            [sys.executable, str(script_path), str(test_dir),
             "--port", str(port), "--no-browser"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Wait for server to start
        if not self.wait_for_server(port):
            self.fail("Character sorter server did not start in time")
        
        # Navigate to the page
        self.driver.get(f"http://127.0.0.1:{port}")
        
        # Verify page loads
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Check title
        self.assertIn("Character Sorter", self.driver.title)
        
        # Verify images are present
        images = self.driver.find_elements(By.TAG_NAME, "img")
        self.assertGreater(len(images), 0, "No images found on character sorter page")


class TestMultiDirectoryViewerSmoke(WebToolSmokeTest):
    """Smoke tests for multi-directory viewer."""
    
    def create_test_structure(self):
        """Create test directories with images."""
        base = self.temp_path / "test_dirs"
        base.mkdir()
        
        # Create 2 directories with images
        for dir_id in range(2):
            dir_path = base / f"dir_{dir_id}"
            dir_path.mkdir()
            for img_id in range(3):
                img_path = dir_path / f"image_{img_id}.png"
                img = Image.new('RGB', (200, 200), color=(dir_id * 100, img_id * 50, 150))
                img.save(img_path)
        
        return base
    
    def test_multi_directory_viewer_starts(self):
        """Test that multi-directory viewer can start and page loads."""
        test_base = self.create_test_structure()
        
        # Start the multi-directory viewer
        script_path = Path(__file__).parent.parent / "05_web_multi_directory_viewer.py"
        port = 8767
        
        self.process = subprocess.Popen(
            [sys.executable, str(script_path), str(test_base),
             "--port", str(port), "--no-browser"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Wait for server to start
        if not self.wait_for_server(port):
            self.fail("Multi-directory viewer server did not start in time")
        
        # Navigate to the page
        self.driver.get(f"http://127.0.0.1:{port}")
        
        # Verify page loads
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Check title
        self.assertIn("Viewer", self.driver.title)


class TestDuplicateFinderSmoke(WebToolSmokeTest):
    """Smoke tests for duplicate finder."""
    
    def create_test_directories(self):
        """Create two test directories."""
        dir1 = self.temp_path / "dir_left"
        dir2 = self.temp_path / "dir_right"
        dir1.mkdir()
        dir2.mkdir()
        
        # Create some images in each
        for i in range(3):
            img = Image.new('RGB', (200, 200), color=(i * 80, 100, 150))
            img.save(dir1 / f"image_{i}.png")
            img.save(dir2 / f"other_{i}.png")
        
        return dir1, dir2
    
    def test_duplicate_finder_starts(self):
        """Test that duplicate finder can start and page loads."""
        dir1, dir2 = self.create_test_directories()
        
        # Start the duplicate finder
        script_path = Path(__file__).parent.parent / "06_web_duplicate_finder.py"
        port = 8768
        
        self.process = subprocess.Popen(
            [sys.executable, str(script_path), str(dir1), str(dir2),
             "--port", str(port), "--no-browser"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Wait for server to start
        if not self.wait_for_server(port):
            self.fail("Duplicate finder server did not start in time")
        
        # Navigate to the page
        self.driver.get(f"http://127.0.0.1:{port}")
        
        # Verify page loads
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Check title
        self.assertIn("Duplicate Finder", self.driver.title)


if __name__ == "__main__":
    unittest.main()

