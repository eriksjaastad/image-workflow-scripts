# Code Conventions

This document defines coding standards for the image-workflow-scripts project. Following these conventions prevents linting errors and improves code quality.

## Path Operations

**Use `Path.open()` instead of `open()`**

```python
# ✅ Good
from pathlib import Path

path = Path("data.json")
with path.open("r") as f:
    data = f.read()

# ❌ Bad
with open(path, "r") as f:
    data = f.read()
```

**Why:** `Path.open()` is type-safe, handles path-like objects correctly, and is the modern Pythonic approach.

---

## Datetime Handling

**Always use timezone-aware datetimes**

```python
# ✅ Good
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc)
iso_string = timestamp.isoformat()

# ❌ Bad
timestamp = datetime.now()  # Naive datetime
timestamp = datetime.utcnow()  # Deprecated, returns naive datetime
```

**Why:** Timezone-naive datetimes cause bugs when working with timestamps across timezones or DST boundaries.

---

## Logging

**Use lazy formatting (% style) instead of f-strings**

```python
# ✅ Good
logger.error("Failed to process %s: %s", filename, error)
logger.info("Processing group %s with %d items", group_id, count)

# ❌ Bad
logger.error(f"Failed to process {filename}: {error}")
logger.info(f"Processing group {group_id} with {count} items")
```

**Why:** Lazy formatting defers string construction until the log is actually written, improving performance and allowing log aggregation tools to group similar messages.

---

## Import Organization

**Keep imports at the module top-level**

```python
# ✅ Good
from pathlib import Path
import json
import re

def process_file(path: Path):
    pattern = re.compile(r"\d+")
    # ...

# ❌ Bad
def process_file(path):
    import re  # Don't import inside functions
    pattern = re.compile(r"\d+")
    # ...
```

**Exceptions allowed:**
- Conditional imports for optional dependencies
- Imports inside type-checking blocks (`if TYPE_CHECKING:`)

**Why:** Top-level imports make dependencies explicit, improve startup performance, and make code easier to understand.

---

## Exception Handling

**Catch specific exceptions, not blanket `Exception`**

```python
# ✅ Good
try:
    data = json.loads(text)
except json.JSONDecodeError as e:
    logger.warning("Invalid JSON: %s", e)
    data = {}

# ⚠️ Use with caution
try:
    plugin.optional_feature()
except Exception as e:
    logger.debug("Optional feature unavailable: %s", e)  # OK for optional features
```

**Why:** Catching `Exception` can hide bugs. Only use broad exception handling for:
- Optional feature detection
- Plugin systems where failures are expected
- Top-level error boundaries with proper logging

**Never silently swallow exceptions:**

```python
# ❌ Bad
try:
    risky_operation()
except Exception:
    pass  # Silent failure - bug hiding!

# ✅ Good
try:
    risky_operation()
except Exception as e:
    logger.warning("Non-critical operation failed: %s", e)
```

---

## Print Statements

**Use logging in library code, print in CLI scripts**

```python
# ✅ Good for workflow scripts (scripts/00_*.py, scripts/01_*.py, etc.)
print(f"✓ Processed {count} images")
print(f"✗ Error: {error}")

# ✅ Good for library code (scripts/utils/*)
logger.info("Processed %d images", count)
logger.error("Operation failed: %s", error)

# ❌ Bad for library code
print("Debug message")  # Use logger.debug() instead
```

**Why:** Print statements in library code can't be controlled by users and clutter output. CLI scripts are user-facing so prints are appropriate.

---

## Type Hints

**Use modern type hint syntax (Python 3.11+)**

```python
# ✅ Good
def process_items(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# ❌ Bad (old style)
from typing import List, Dict

def process_items(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}
```

**Why:** Python 3.10+ supports built-in generic types without importing from `typing`.

---

## Summary Checklist

When writing new code, ensure:

- [ ] Use `Path.open()` instead of `open()`
- [ ] Use `datetime.now(timezone.utc)` instead of `datetime.now()` or `datetime.utcnow()`
- [ ] Use lazy logging: `logger.info("Message %s", var)` not `logger.info(f"Message {var}")`
- [ ] Keep imports at top-level (except conditional imports)
- [ ] Catch specific exceptions, log broad ones
- [ ] Use logging in library code (`scripts/utils/**`), prints in CLI scripts
- [ ] Use modern type hints (`list[str]` not `List[str]`)

---

## Enforcement

These conventions are enforced by:
- **Ruff** linting in CI/CD
- **Pre-commit hooks** that auto-fix some violations
- **Code review**

See `pyproject.toml` for the full Ruff configuration.
