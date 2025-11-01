#!/usr/bin/env python3
"""
Companion Policy Tests
======================

Verifies that companion handling moves ALL same-name files across any extension
and raises on single-file image moves without companions.

Run:
  python scripts/tests/test_companion_policy.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.companion_file_utils import move_file_with_all_companions  # type: ignore


def write_dummy(p: Path, content: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content.encode("utf-8"))


def test_moves_all_same_name_extensions_and_group_decision() -> bool:
    print("\n🧪 Test: Move with all companions (any extension + group .decision)")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        dst = Path(td) / "dst"
        src.mkdir()
        dst.mkdir()

        # Group stem and full image name
        group_id = "20250101_000001"
        img = src / f"{group_id}_stage2_upscaled.png"
        # Same-name companions
        companions = [
            src / f"{group_id}_stage2_upscaled.yaml",
            src / f"{group_id}_stage2_upscaled.caption",
            src / f"{group_id}_stage2_upscaled.txt",
            src / f"{group_id}_stage2_upscaled.foo",  # made-up extension
        ]
        # Group-level decision sidecar (no _stage)
        decision = src / f"{group_id}.decision"

        write_dummy(img)
        for c in companions:
            write_dummy(c)
        write_dummy(decision)

        moved = move_file_with_all_companions(img, dst, dry_run=False)
        moved_set = set(moved)

        expected = {p.name for p in [img, decision] + companions}
        assert moved_set == expected, f"Expected moved {expected}, got {moved_set}"

        # Ensure files moved from src to dst
        for name in expected:
            assert not (src / name).exists(), f"Should be moved from src: {name}"
            assert (dst / name).exists(), f"Should exist in dst: {name}"

        print("   ✅ Moved all same-name companions and group .decision")
        return True


def test_raises_on_single_image_without_companions() -> bool:
    print("\n🧪 Test: Raise error on single image move without companions")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        dst = Path(td) / "dst"
        src.mkdir()
        dst.mkdir()

        img = src / "20250101_000002_stage1_generated.png"
        write_dummy(img)

        # Ensure override is not set
        os.environ.pop("COMPANION_ALLOW_SINGLE_FILE", None)

        raised = False
        try:
            move_file_with_all_companions(img, dst, dry_run=False)
        except RuntimeError as e:
            raised = True
            assert "COMPANION POLICY VIOLATION" in str(e)

        assert (
            raised
        ), "Expected RuntimeError when moving single image without companions"
        print("   ✅ Raised on single-file image move as expected")
        return True


def run_all() -> bool:
    tests = [
        test_moves_all_same_name_extensions_and_group_decision,
        test_raises_on_single_image_without_companions,
    ]
    passed = 0
    for t in tests:
        try:
            if t():
                passed += 1
        except AssertionError as e:
            print(f"   ❌ FAILED: {e}")
            return False
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False
    print(f"\n✅ Companion policy tests passed: {passed}/{len(tests)}")
    return True


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
