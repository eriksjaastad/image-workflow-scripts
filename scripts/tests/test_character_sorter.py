#!/usr/bin/env python3
"""
Quick test script to run character sorter in headless browser and take screenshot
"""
import subprocess
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_character_sorter():
    # Start the character sorter in background
    print("Starting character sorter...")
    process = subprocess.Popen([
        "python", "scripts/03_web_character_sorter.py", 
        "mojo1_clustered/group_0001"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(3)
    
    # Setup headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("http://localhost:5000")
        
        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "image-name")))
        
        # Take screenshot
        screenshot_path = "character_sorter_test.png"
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved to: {screenshot_path}")
        
        # Check if image names are visible
        image_names = driver.find_elements(By.CLASS_NAME, "image-name")
        print(f"Found {len(image_names)} image name elements")
        
        for i, name_elem in enumerate(image_names):
            text = name_elem.text
            color = name_elem.value_of_css_property("color")
            print(f"Image {i+1}: '{text}' (color: {color})")
        
        driver.quit()
        
    except Exception as e:
        print(f"Error: {e}")
        driver.quit()
    
    finally:
        # Kill the character sorter process
        process.terminate()
        process.wait()

if __name__ == "__main__":
    test_character_sorter()
