# Code Quality Rules (Ruff + Pre-commit)

**Audience:** Developers & AI Assistants  
**Last Updated:** October 31, 2025

---

## 🎯 Philosophy

**Strict but pragmatic:** Catch real bugs and enforce clean code, but don't make trivial things annoying.

---

## 🚨 Rules That Will FAIL Pre-commit

### 1. **Silent Broad Exceptions** (Custom Hook)

❌ **FORBIDDEN:**

```python
try:
    risky_operation()
except:  # Catches everything, hides bugs
    pass

try:
    risky_operation()
except Exception:  # Too broad, silent
    pass
```

✅ **REQUIRED:**

```python
try:
    risky_operation()
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise

try:
    risky_operation()
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise  # Re-raise after logging
```

**Why:** Silent failures hide bugs. Always log and re-raise, or catch specific exceptions.

---

### 2. **Print Statements in Library Code** (T201)

❌ **FORBIDDEN in:** `scripts/utils/`, `scripts/ai/`, library modules

```python
def process_data(data):
    print("Processing...")  # Don't print in libraries
    return data
```

✅ **ALLOWED in:** CLI scripts, tests, dashboard

```python
# scripts/00_start_project.py (CLI script - OK!)
print("Starting project...")

# scripts/utils/helper.py (library - use logging!)
import logging
logger = logging.getLogger(__name__)
logger.info("Processing data...")
```

**Exempted files:**

- `scripts/[0-9][0-9]_*.py` (numbered workflow scripts)
- `scripts/**/run_*.py` (run scripts)
- `scripts/**/test_*.py` (tests)
- `scripts/dashboard/**` (dashboard outputs)
- `scripts/tools/**` (CLI tools)

---

### 3. **Undefined Names** (F821)

❌ **FORBIDDEN:**

```python
result = undefined_variable  # Typo or missing import
```

✅ **REQUIRED:**

```python
from module import needed_function
result = needed_function()
```

---

### 4. **Unused Imports** (F401)

❌ **FORBIDDEN:**

```python
import os
import sys  # Never used - remove it
```

✅ **REQUIRED:**

```python
import os  # Only import what you use
```

---

### 5. **Debugger Left In Code** (T10)

❌ **FORBIDDEN:**

```python
import pdb; pdb.set_trace()  # Leftover debugging
breakpoint()  # Forgot to remove
```

---

### 6. **Poor Error Messages** (EM)

❌ **BAD:**

```python
raise ValueError(f"Invalid value: {value}")  # f-string in exception
```

✅ **GOOD:**

```python
msg = f"Invalid value: {value}"
raise ValueError(msg)  # Separate message construction
```

**Why:** Makes error messages easier to test and maintain.

---

### 7. **Timezone-Naive Datetimes** (DTZ)

❌ **DANGEROUS:**

```python
from datetime import datetime
now = datetime.now()  # Which timezone?
```

✅ **SAFE:**

```python
from datetime import datetime, timezone
now = datetime.now(timezone.utc)  # Explicit UTC
```

---

## ⚠️ Warnings (Won't Fail, But Review)

### Complexity Checks (Disabled but monitor manually)

- **PLR0911:** Too many return statements (>6)
- **PLR0912:** Too many branches (>12)
- **PLR0913:** Too many arguments (>5)
- **PLR0915:** Too many statements (>50)

**Action:** If Ruff suggests these, consider refactoring, but they won't block commits.

---

## ✅ Allowed Patterns

### 1. **Prints in CLI Scripts**

```python
# scripts/00_start_project.py - OK!
print("Project started successfully")
```

### 2. **Flexible Test Assertions**

```python
# scripts/tests/test_something.py - OK!
assert result == expected  # Simple asserts are fine
print(f"Debug: {result}")  # Debugging prints OK in tests
```

### 3. **Magic Values in Tests**

```python
# Tests can use magic numbers/strings directly
assert count == 42  # No need for EXPECTED_COUNT = 42
```

---

## 📋 Quick Reference for AI Models

When writing code, follow these rules:

1. **Never use bare `except:` or silent `except Exception:`**

   - Always log the error
   - Always re-raise or handle specifically

2. **Use logging in library code, prints only in CLI scripts**

   - Check if file is in `scripts/utils/` or `scripts/ai/` → use logging
   - Check if file starts with number like `00_` → prints OK

3. **Always import what you use, remove unused imports**

4. **Use timezone-aware datetimes:**

   ```python
   from datetime import datetime, timezone
   now = datetime.now(timezone.utc)
   ```

5. **Separate error message construction from raising:**

   ```python
   msg = f"Error: {detail}"
   raise ValueError(msg)
   ```

6. **Remove all debugger statements before committing**

---

## 🔧 Running Checks Manually

```bash
# Run Ruff on specific files
ruff check scripts/your_file.py

# Run Ruff on everything
ruff check scripts/

# Auto-fix what can be fixed
ruff check --fix scripts/

# Run pre-commit on staged files
pre-commit run

# Run pre-commit on all files
pre-commit run --all-files
```

---

## 🚫 What We DON'T Check

- **Line length (E501):** Black handles formatting
- **Import position (E402):** Sometimes needed for conditional imports
- **Ternary operators (SIM108):** Sometimes if/else is clearer
- **Contextlib.suppress (SIM105):** Explicit try/except often clearer

---

## 📝 Summary for AI Code Generation

**Before writing code, ask:**

1. Is this a CLI script or library code? (Determines print vs logging)
2. Am I catching exceptions? (Must log and re-raise or be specific)
3. Am I using datetime? (Must use timezone.utc)
4. Did I import everything I'm using? (Remove unused)
5. Are my error messages constructed separately? (For testability)

**These rules will catch 95% of bugs before they reach production.**
