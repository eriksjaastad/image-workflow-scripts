#!/usr/bin/env python3
"""Character Directory Analyzer (single-underscore bins)

Purpose
  Quickly analyze top-level single-underscore directories (e.g., `_middle_eastern`)
  for demographic breakdowns like ethnicity, age, hair color, body type.

Usage
  # Scan single-underscore subdirs under current directory
  python scripts/04_character_check.py --fields ethnicity,age

  # Scan single-underscore subdirs under the given directory
  python scripts/04_character_check.py mojo3 --fields hair,body

Notes
  - Read-only. Does not move or modify any files.
  - Scans ONLY top-level directories whose names start with a single underscore
    (name.startswith("_") and not name.startswith("__")).
  - For each PNG, attempts to read the same-stem YAML to extract prompt text.
  - Uses the shared extraction helpers from 02_character_processor.py for consistency.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Ensure project imports resolve
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import shared helpers from character_processor


def _load_character_processor_module():
    """Load 02_character_processor.py as a module regardless of its filename."""
    candidates = [
        PROJECT_ROOT / "scripts" / "02_character_processor.py",
        Path(__file__).parent / "02_character_processor.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "character_processor", str(path)
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                return mod
    raise ImportError("Could not locate 02_character_processor.py to import helpers")


_cp = _load_character_processor_module()
extract_ethnicity_from_prompt = _cp.extract_ethnicity_from_prompt
extract_age_from_prompt = _cp.extract_age_from_prompt
extract_hair_color_from_prompt = _cp.extract_hair_color_from_prompt
extract_body_type_from_prompt = _cp.extract_body_type_from_prompt
parse_yaml_file = _cp.parse_yaml_file


def build_field_extractors(
    fields: List[str],
) -> Dict[str, Callable[[str], Optional[str]]]:
    """Map field names to extraction functions operating on prompt_lower."""
    mapping: Dict[str, Callable[[str], Optional[str]]] = {}
    for f in fields:
        key = f.strip().lower()
        if key in {"ethnicity", "eth", "race"}:
            mapping["ethnicity"] = extract_ethnicity_from_prompt
        elif key in {"age", "ages"}:
            mapping["age"] = extract_age_from_prompt
        elif key in {"hair", "hair_color"}:
            mapping["hair"] = extract_hair_color_from_prompt
        elif key in {"body", "body_type"}:
            mapping["body"] = extract_body_type_from_prompt
        else:
            raise ValueError(f"Unknown field: {f}")
    return mapping


def read_prompt_from_yaml(yaml_path: Path) -> str:
    """Read prompt text from YAML using the shared parser (best-effort)."""
    try:
        parsed = parse_yaml_file(yaml_path)
        if parsed and isinstance(parsed.get("prompt"), str):
            return parsed["prompt"]
    except Exception:
        pass
    return ""


def analyze_directory(
    dir_path: Path, fields: Dict[str, Callable[[str], Optional[str]]]
) -> Dict[str, Counter]:
    """Analyze one directory's images, returning a Counter per field."""
    results: Dict[str, Counter] = {name: Counter() for name in fields.keys()}

    for png in sorted(dir_path.glob("*.png")):
        yaml_path = png.with_suffix(".yaml")
        if not yaml_path.exists():
            # Skip if no metadata; we don't guess from filename
            continue
        prompt_lower = read_prompt_from_yaml(yaml_path).lower()
        if not prompt_lower:
            continue

        for field_name, extractor in fields.items():
            try:
                value = extractor(prompt_lower)
            except Exception:
                value = None
            results[field_name][value or "unknown"] += 1

    return results


def is_single_underscore_dir(p: Path) -> bool:
    name = p.name
    return p.is_dir() and name.startswith("_") and not name.startswith("__")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze single-underscore character directories"
    )
    # Optional positional directory; defaults to current working directory
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(Path.cwd()),
        help="Directory whose single-underscore subdirectories will be scanned",
    )
    # Backward compatibility: --base still supported (overridden by positional if given)
    parser.add_argument(
        "--base",
        default=None,
        help="Deprecated: use positional directory instead",
    )
    parser.add_argument(
        "--fields",
        required=True,
        help="Comma-separated fields to analyze (ethnicity,age,hair,body)",
    )
    args = parser.parse_args()

    base_path_str = args.directory if args.directory else (args.base or str(Path.cwd()))
    if args.base and args.directory and args.base != args.directory:
        # Prefer positional; ignore --base silently
        pass

    base_dir = Path(base_path_str).expanduser().resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"❌ Not a directory: {base_dir}")
        sys.exit(1)

    raw_fields = [s.strip() for s in args.fields.split(",") if s.strip()]
    try:
        field_extractors = build_field_extractors(raw_fields)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Discover single-underscore bins
    bins = [p for p in base_dir.iterdir() if is_single_underscore_dir(p)]
    bins.sort(key=lambda p: p.name.lower())

    if not bins:
        print(f"No single-underscore directories found under {base_dir}")
        return

    print("📊 CHARACTER BIN CHECK (single-underscore dirs)\n" + "=" * 60)
    print(f"Base: {base_dir}")
    print(f"Fields: {', '.join(field_extractors.keys())}\n")

    for bin_dir in bins:
        pngs = list(bin_dir.glob("*.png"))
        if not pngs:
            continue
        print(f"{bin_dir.name} — {len(pngs)} images")

        results = analyze_directory(bin_dir, field_extractors)
        for field_name in field_extractors.keys():
            counts = results.get(field_name, Counter())
            total = sum(counts.values()) or 0
            if total == 0:
                print(f"  • {field_name}: (no YAML or no matches)")
                continue
            # Show top 8 values as bullets
            top = counts.most_common(8)
            print(f"  • {field_name}:")
            for k, v in top:
                print(f"    - {k}: {v}")
        print()


if __name__ == "__main__":
    main()
