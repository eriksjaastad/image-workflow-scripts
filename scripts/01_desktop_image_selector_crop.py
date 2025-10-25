#!/usr/bin/env python3
"""
Compatibility shim for legacy import path expected by tests.
Delegates to the archived implementation in scripts/archive/01_desktop_image_selector_crop.py.
"""

from pathlib import Path
import importlib.util
import sys

_here = Path(__file__).parent
_target = _here / "archive" / "01_desktop_image_selector_crop.py"

_spec = importlib.util.spec_from_file_location("desktop_image_selector_crop", _target)
assert _spec is not None and _spec.loader is not None, "Failed to locate archived desktop selector"
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module  # type: ignore[index]
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]

# Re-export all public attributes
for name in dir(_module):
    if not name.startswith("_"):
        globals()[name] = getattr(_module, name)


