"""
Simple Selenium integration tests to verify basic Flask app functionality.

These tests create minimal Flask apps to verify that:
1. Chrome WebDriver works in headless mode
2. Flask apps can be started and accessed
3. Basic HTML rendering works
4. JavaScript can execute

This provides a foundation for more complex integration tests.
"""

import sys
import time
import unittest
from pathlib import Path

from flask import Flask
from selenium.webdriver.common.by import By

from scripts.tests.test_base_selenium import BaseSeleniumTest


class TestSeleniumInfrastructure(BaseSeleniumTest):
    """Test that Selenium infrastructure is working correctly."""
    
    def get_flask_app(self):
        """Create a minimal test Flask app."""
        app = Flask(__name__)
        
        @app.route("/")
        def index():
            return """
            <!DOCTYPE html>
            <html>
            <head><title>Test App</title></head>
            <body>
                <h1 id="heading">Hello Selenium!</h1>
                <button id="test-btn" class="action-btn">Click Me</button>
                <div id="result"></div>
                <script>
                    document.getElementById('test-btn').addEventListener('click', function() {
                        document.getElementById('result').textContent = 'Button clicked!';
                    });
                </script>
            </body>
            </html>
            """
        
        return app
    
    def test_page_loads(self):
        """Test that a basic page loads."""
        self.get("/")
        heading = self.driver.find_element(By.ID, "heading")
        self.assertEqual(heading.text, "Hello Selenium!")
    
    def test_button_present(self):
        """Test that buttons are found."""
        self.get("/")
        button = self.driver.find_element(By.ID, "test-btn")
        self.assertIsNotNone(button)
        self.assertEqual(button.text, "Click Me")
    
    def test_javascript_execution(self):
        """Test that JavaScript executes."""
        self.get("/")
        button = self.driver.find_element(By.ID, "test-btn")
        button.click()
        time.sleep(0.1)  # Give JS time to execute
        result = self.driver.find_element(By.ID, "result")
        self.assertEqual(result.text, "Button clicked!")


if __name__ == "__main__":
    unittest.main()

