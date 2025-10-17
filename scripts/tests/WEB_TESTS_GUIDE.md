# Web Tools Test Suite

Comprehensive test suite for all web-based image processing tools, ensuring functionality, safety, and style guide compliance.

## 🧪 Test Coverage

### Web Image Selector Tests (`test_web_image_selector.py`)
- **Batch Size**: Verifies default batch size is 100 groups
- **Keyboard Shortcuts**: Tests 1,2,3,Q,W,E key bindings
- **Unselect Functionality**: Tests clicking selected images to deselect
- **State Override**: Tests that 1,2,3 keys clear crop state from Q,W,E
- **Button Toggle**: Tests that pressing same button twice deselects
- **Navigation**: Tests Enter (forward) and Up Arrow (back) navigation
- **Process Button Safety**: Tests button is disabled until scrolled to bottom
- **Style Guide Compliance**: Verifies CSS variables and color scheme

### Web Character Sorter Tests (`test_web_character_sorter.py`)
- **Page Loading**: Verifies basic page structure and elements
- **Sticky Header**: Tests header positioning and navigation buttons
- **Character Groups**: Tests G1, G2, G3 button functionality
- **Keyboard Shortcuts**: Tests common keyboard bindings
- **Image Display**: Verifies proper image aspect ratios and object-fit
- **Style Guide Compliance**: Tests color variables and theming
- **Responsive Layout**: Tests layout adaptation to different screen sizes
- **Error Handling**: Ensures graceful handling of missing data

### Web Multi-Directory Viewer Tests (`test_web_multi_directory_viewer.py`)
- **Page Loading**: Tests basic functionality with nested/flat directory structures
- **Sticky Header with Live Stats**: Tests three-column header layout with real-time counters
- **Delete Toggle**: Tests clicking images to toggle delete state (red outline)
- **Crop Button**: Tests crop button functionality (white outline)
- **State Override**: Tests that delete and crop states override each other
- **Image Aspect Ratios**: Verifies images maintain proper proportions (object-fit: contain)
- **Style Guide Compliance**: Tests full color palette integration
- **Process Button**: Tests selection processing functionality

## 🚀 Running Tests

### Individual Test Suites
```bash
# Test Web Image Selector only
python scripts/tests/test_web_image_selector.py

# Test Web Character Sorter only
python scripts/tests/test_web_character_sorter.py

# Test Web Multi-Directory Viewer only
python scripts/tests/test_web_multi_directory_viewer.py
```

### Comprehensive Test Suite
```bash
# Run all web tool tests
python scripts/tests/test_all_web_tools.py

# Run specific tool tests
python scripts/tests/test_all_web_tools.py --image-selector
python scripts/tests/test_all_web_tools.py --character-sorter
python scripts/tests/test_all_web_tools.py --multi-directory
```

### Integrated with Main Test Runner
```bash
# Run all tests including web tools
python scripts/tests/test_runner.py

# Run only safety tests (excludes web tools)
python scripts/tests/test_runner.py --safety-only
```

## 🛠️ Test Requirements

### Dependencies
- **Selenium WebDriver**: For browser automation
- **Chrome/Chromium**: Headless browser for testing
- **Flask**: Web server for tools
- **Python 3.8+**: Test framework compatibility

### Test Data
Tests use existing test data from:
- `scripts/tests/data/test_images_medium/`
- `scripts/tests/data/problematic_sequential/`
- Temporary test directories (created automatically)

### Port Usage
Tests use different ports to avoid conflicts:
- Web Image Selector: `5001`
- Web Character Sorter: `5003`
- Web Multi-Directory Viewer: `5004`

## ✅ Expected Results

### Passing Tests
All tests are designed to pass with the current implementation:
- **Recent Enhancements**: Unselect, batch size 100, state override, navigation
- **Style Guide**: Consistent dark theme colors and patterns
- **Safety Features**: Process button scroll protection
- **Interactive Features**: Crop/delete toggles with live statistics

### Test Output
```
🧪 Starting Web Image Selector Tests...
✅ Batch size default test passed
✅ Keyboard shortcuts test passed
✅ Unselect functionality test passed
✅ State override test passed
✅ Button toggle test passed
✅ Navigation keys test passed
✅ Process button safety test passed
✅ Style guide compliance test passed
🎉 All Web Image Selector tests passed!
```

## 🎯 Style Guide Integration

Tests verify compliance with `WEB_STYLE_GUIDE.md`:
- **CSS Variables**: `--bg`, `--surface`, `--accent`, `--danger`, `--success`
- **Color Values**: `#101014` (bg), `#4f9dff` (accent), `#ff6b6b` (danger)
- **Consistent Theming**: Dark theme across all tools
- **Interactive States**: Proper outline colors for selections

## 🔧 Troubleshooting

### Common Issues
1. **Port Conflicts**: Tests use different ports, but ensure no other services are running
2. **Chrome Driver**: Ensure ChromeDriver is installed and in PATH
3. **Test Data**: Some tests create temporary data, others use existing test files
4. **Timeouts**: Headless browser tests may need longer timeouts on slower systems

### Debug Mode
For debugging failed tests, modify the Chrome options in test files:
```python
# Remove headless mode for visual debugging
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Comment out this line
```

## 📊 Test Metrics

### Coverage Areas
- **Functionality**: All interactive features and workflows
- **Safety**: Data protection and error handling
- **Performance**: Page load times and responsiveness
- **Accessibility**: Keyboard navigation and visual feedback
- **Consistency**: Style guide compliance across all tools

### Success Criteria
- All tests pass without errors
- No JavaScript console errors
- Proper CSS styling and layout
- Correct interactive behavior
- Style guide compliance verified

This comprehensive test suite ensures that all web tools maintain high quality, consistent user experience, and robust functionality across all recent enhancements.
