"""
Base classes and utilities for Selenium-based integration tests of Flask web tools.

Provides:
- BaseSeleniumTest: Base class with browser setup, Flask server management, and cleanup
- Utilities for headless browser configuration
- Port management to avoid conflicts
- Automatic server lifecycle management
"""

import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


def find_free_port() -> int:
    """Find a free port for the Flask server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class BaseSeleniumTest(unittest.TestCase):
    """
    Base class for Selenium integration tests of Flask web applications.

    Features:
    - Automatic headless Chrome setup
    - Flask server lifecycle management (starts in background thread)
    - Temporary directory creation for test data
    - Automatic cleanup of browser and server
    - Port management to avoid conflicts

    Subclasses should override:
    - get_flask_app(): Return the Flask app instance to test
    - prepare_test_data(): Create test data in self.temp_dir
    """

    driver: webdriver.Chrome | None = None
    server_thread: threading.Thread | None = None
    server_port: int = 0
    temp_dir: tempfile.TemporaryDirectory | None = None

    @classmethod
    def setUpClass(cls):
        """Set up Chrome driver once for all tests in the class."""
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--log-level=3")

        service = ChromeService(ChromeDriverManager().install())
        cls.driver = webdriver.Chrome(service=service, options=chrome_options)
        cls.driver.implicitly_wait(10)

    @classmethod
    def tearDownClass(cls):
        """Clean up Chrome driver."""
        if cls.driver:
            cls.driver.quit()
            cls.driver = None

    def setUp(self):
        """Set up test environment: temp directory, test data, and Flask server."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Set environment variable for test data root
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_path)

        # Prepare test data (subclass responsibility)
        self.prepare_test_data()

        # Find free port and start Flask server
        self.server_port = find_free_port()
        self.server_url = f"http://127.0.0.1:{self.server_port}"

        # Get Flask app from subclass
        app = self.get_flask_app()

        # Start server in background thread
        self.server_thread = threading.Thread(
            target=lambda: app.run(
                host="127.0.0.1", port=self.server_port, debug=False, use_reloader=False
            ),
            daemon=True,
        )
        self.server_thread.start()

        # Wait for server to be ready
        self.wait_for_server()

    def tearDown(self):
        """Clean up test environment."""
        # Clean up environment variable
        if "EM_TEST_DATA_ROOT" in os.environ:
            del os.environ["EM_TEST_DATA_ROOT"]

        # Clean up temporary directory
        if self.temp_dir:
            self.temp_dir.cleanup()
            self.temp_dir = None

        # Server thread is daemon, so it will stop when main thread exits
        # No explicit server shutdown needed

    def wait_for_server(self, timeout: int = 10):
        """Wait for Flask server to be ready to accept connections."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(("127.0.0.1", self.server_port))
                sock.close()
                if result == 0:
                    # Give it a bit more time to fully initialize
                    time.sleep(0.5)
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"Server did not start within {timeout} seconds")

    def get_flask_app(self):
        """
        Return the Flask app instance to test.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_flask_app()")

    def prepare_test_data(self):
        """
        Prepare test data in self.temp_path.
        Can be overridden by subclasses. Default is no-op.
        """

    def get(self, path: str = "/"):
        """Navigate to a path on the test server."""
        url = f"{self.server_url}{path}"
        self.driver.get(url)

    def wait_for_element(self, by, value, timeout=10):
        """Wait for an element to be present."""
        return WebDriverWait(self.driver, timeout).until(
            lambda d: d.find_element(by, value)
        )
