from __future__ import annotations

import importlib.util
from pathlib import Path

_target = Path(__file__).with_name("01_ai_assisted_reviewer.py")
_spec = importlib.util.spec_from_file_location(
    "scripts._ai_assisted_reviewer_01", _target
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load target module: {_target}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update({k: v for k, v in _module.__dict__.items() if not k.startswith("_")})
