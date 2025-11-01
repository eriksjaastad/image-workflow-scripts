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

Notes:
  - Read-only. Does not move or modify any files.
  - Scans ONLY top-level directories whose names start with a single underscore
    (name.startswith("_") and not name.startswith("__")).
  - For each PNG, attempts to read the same-stem YAML to extract prompt text.
  - Uses the shared extraction helpers from 02_character_processor.py for consistency.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

# Ensure project imports resolve
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging (overridable with --verbose)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Import shared helpers from character_processor


def _load_character_processor_module():
    """Load 02_character_processor.py as a module regardless of its filename."""
    candidates = [
        PROJECT_ROOT / "scripts" / "02_character_processor.py",
        Path(__file__).parent / "02_character_processor.py",
    ]
    for path in candidates:
        if path.exists():
            try:
                spec = importlib.util.spec_from_file_location(
                    "character_processor", str(path)
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                    logger.debug(f"Loaded character_processor from {path}")
                    return mod
            except Exception as e:
                msg = f"Found {path} but failed to load: {e}"
                logger.error(msg)
                raise ImportError(msg) from e
    msg = (
        f"Could not locate 02_character_processor.py in: {[str(c) for c in candidates]}"
    )
    raise ImportError(msg)


_cp = _load_character_processor_module()
extract_ethnicity_from_prompt = _cp.extract_ethnicity_from_prompt
extract_age_from_prompt = _cp.extract_age_from_prompt
extract_hair_color_from_prompt = _cp.extract_hair_color_from_prompt
extract_body_type_from_prompt = _cp.extract_body_type_from_prompt
parse_yaml_file = _cp.parse_yaml_file


def build_field_extractors(
    fields: list[str],
) -> dict[str, Callable[[str], str | None]]:
    """Map field names to extraction functions operating on prompt_lower."""
    mapping: dict[str, Callable[[str], str | None]] = {}
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
            msg = f"Unknown field: {f}"
            raise ValueError(msg)
    return mapping


def read_prompt_from_yaml(yaml_path: Path) -> str:
    """Read prompt text from YAML using the shared parser (best-effort)."""
    try:
        parsed = parse_yaml_file(yaml_path)
        if parsed and isinstance(parsed.get("prompt"), str):
            return parsed["prompt"]
        logger.debug(f"No prompt field in {yaml_path}")
    except Exception as e:
        logger.warning(f"Failed to parse {yaml_path}: {e}")
    return ""


def analyze_directory(
    dir_path: Path, fields: dict[str, Callable[[str], str | None]]
) -> dict[str, Counter]:
    """Analyze one directory's images, returning a Counter per field."""
    results: dict[str, Counter] = {name: Counter() for name in fields}
    extraction_failures = Counter()

    for png in sorted(dir_path.glob("*.png")):
        yaml_path = png.with_suffix(".yaml")
        if not yaml_path.exists():
            continue
        prompt_lower = read_prompt_from_yaml(yaml_path).lower()
        if not prompt_lower:
            continue

        for field_name, extractor in fields.items():
            try:
                value = extractor(prompt_lower)
            except Exception as e:
                logger.warning(f"Extractor '{field_name}' failed on {png.name}: {e}")
                extraction_failures[field_name] += 1
                value = None
            results[field_name][value or "unknown"] += 1

    if extraction_failures:
        logger.error(
            f"Extraction failures in {dir_path.name}: {dict(extraction_failures)}"
        )

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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging for troubleshooting",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if sanity checks fail (e.g., excessive 'unknown')",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

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
        msg = f"❌ {e}"
        print(msg)
        sys.exit(1)

    try:
        _self_test_extractors(field_extractors)
    except RuntimeError as e:
        msg = f"❌ {e}"
        print(msg)
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

    total_bins = 0
    total_images = 0
    strict_failure = False
    for bin_dir in bins:
        total_bins += 1
        pngs = list(bin_dir.glob("*.png"))
        if not pngs:
            continue
        total_images += len(pngs)
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
            unknown = counts.get("unknown", 0)
            if total > 0 and unknown / total > 0.5:
                logger.error(
                    f"High unknown ratio for '{field_name}' in {bin_dir.name}: {unknown}/{total}"
                )
                if args.strict:
                    strict_failure = True
        print()

    print(f"✅ Analysis complete: processed {total_bins} bins, {total_images} images")
    if args.strict and strict_failure:
        print("❌ Strict mode failure due to excessive 'unknown' ratios")
        sys.exit(1)


def _self_test_extractors(field_extractors: dict[str, Callable]) -> None:
    """Verify extractors are callable and return expected types."""
    test_prompt = "a 25 year old caucasian woman with blonde hair, athletic body"
    failures = []
    for field_name, extractor in field_extractors.items():
        try:
            result = extractor(test_prompt)
            if not (result is None or isinstance(result, str)):
                failures.append(
                    f"{field_name} returned {type(result)}, expected str|None"
                )
        except Exception as e:
            failures.append(f"{field_name} raised {e}")
    if failures:
        msg = f"Extractor self-test failed: {'; '.join(failures)}"
        logger.error(msg)
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()
