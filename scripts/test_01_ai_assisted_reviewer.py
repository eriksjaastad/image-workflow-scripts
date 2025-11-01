#!/usr/bin/env python3
"""
Adapter module so tests can import symbols from `01_ai_assisted_reviewer.py`.

The original file name starts with digits, which is not a valid Python module
identifier. This shim loads the module by path and re-exports required symbols
for the tests:
  - ImageGroup
  - RankerModel
  - CropProposerModel
  - build_app
  - DEFAULT_BATCH_SIZE
"""

import importlib.util
import sys
from pathlib import Path

_scripts_dir = Path(__file__).parent
_target = _scripts_dir / "01_ai_assisted_reviewer.py"

_spec = importlib.util.spec_from_file_location("ai_assisted_reviewer", _target)
_module = importlib.util.module_from_spec(_spec)
assert (
    _spec is not None and _spec.loader is not None
), "Failed to load 01_ai_assisted_reviewer.py"
# Ensure dataclasses and other introspection can resolve the module by name
sys.modules[_spec.name] = _module  # type: ignore[index]
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]

# Re-export the required symbols for tests
ImageGroup = _module.ImageGroup
RankerModel = _module.RankerModel
CropProposerModel = _module.CropProposerModel
build_app = _module.build_app
DEFAULT_BATCH_SIZE = _module.DEFAULT_BATCH_SIZE
