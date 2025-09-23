# Image Processing Workflow Tests

This directory contains all test-related files for the image processing workflow scripts.

## Directory Structure

```
scripts/tests/
├── README.md                    # This documentation
├── test_runner.py              # Main test runner (comprehensive test suite)
├── create_test_data.py         # Test data generator 
├── generate_test_manifest.py   # Test manifest generator
└── data/                       # Test data storage
    ├── test_manifest.json      # Analysis of XXX_CONTENT structure
    ├── test_images_medium/     # Medium-sized test dataset
    ├── test_subdirs/           # Subdirectory test data
    └── test_subdirs_with_random/ # Non-standard files test data
```

## Running Tests

### From Project Root (Recommended)
```bash
# Run all critical safety tests
python scripts/tests/test.py --safety-only

# Run all tests including performance
python scripts/tests/test.py

# Create fresh test data and run all tests
python scripts/tests/test.py --create-data
```

### From Scripts Directory
```bash
# Run all critical safety tests
python tests/test_runner.py --safety-only

# Run all tests including performance  
python tests/test_runner.py --performance
```

## Test Categories

### 🚨 Critical Safety Tests (Always Run)
These tests prevent catastrophic data loss and ensure script reliability:

1. **File Safety Test** - Ensures scripts never modify source files during analysis
2. **Subdirectory Scanning Test** - Verifies recursive file discovery works correctly
3. **Non-Standard Files Robustness** - Confirms graceful handling of random/malformed files
4. **Batch Isolation Test** - Prevents processing files outside the visible batch

### 🧪 Regression Tests (Optional)
- **Grouping Algorithm Test** - Verifies stage-based triplet/pair detection
- **Memory Usage Test** - Performance testing with large datasets

## Creating Test Data

### Basic Test Data
```bash
# Create 50 triplets for testing
python scripts/tests/create_test_data.py --triplets 50 --output my_test_dir/

# Create mixed data (triplets, pairs, singletons)
python scripts/tests/create_test_data.py --triplets 20 --pairs 5 --singletons 2 --output my_test_dir/
```

### Advanced Test Scenarios
```bash
# Create subdirectory test structure
python scripts/tests/create_test_data.py --subdirectory-test --output my_test_dir/

# Create data with non-standard files (robustness testing)
python scripts/tests/create_test_data.py --subdirectory-test --output my_test_dir/

# Create performance test data
python scripts/tests/create_test_data.py --size large --output my_test_dir/
```

## Test Data Sizes
- **Small**: 10 triplets, 2 pairs, 1 singleton (~39 files)
- **Medium**: 50 triplets, 5 pairs, 2 singletons (~324 files) 
- **Large**: 121 triplets, 10 pairs, 5 singletons (~816 files) - matches XXX_CONTENT size
- **Huge**: 500 triplets, 50 pairs, 10 singletons (~3,360 files) - stress testing

## Adding New Tests

1. Add test method to `TestRunner` class in `test_runner.py`
2. Call the test in the `main()` function
3. Mark as `critical=True` if it prevents data loss
4. Document the test purpose and expected behavior

Example:
```python
def test_my_feature(self):
    \"\"\"Test description\"\"\"
    # Test implementation
    return True  # or False

# In main():
runner.run_test("My Feature Test", runner.test_my_feature, critical=False)
```

## Continuous Testing

**IMPORTANT**: Always run tests after modifying any workflow script:

```bash
# Quick safety check (30 seconds)
python test.py --safety-only

# Full validation (2-3 minutes)  
python test.py --performance
```

The test suite prevents regressions and ensures the workflow remains reliable as features are added or modified.
