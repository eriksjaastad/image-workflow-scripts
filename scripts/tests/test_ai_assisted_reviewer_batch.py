#!/usr/bin/env python3
"""
Unit test: AI-Assisted Reviewer batch counts (no double-counting)

This test builds the reviewer app with temporary directories, submits a mixed
batch (approve, crop, and unselected groups), and verifies the response counts:
 - kept: number of selected images (including crop)
 - crop: number of selected images sent to crop/
 - deleted: exact number of images moved to delete_staging/

Run:
  python scripts/tests/test_ai_assisted_reviewer_batch.py
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from file_tracker import FileTracker
import importlib.util
import types
try:
    import flask  # type: ignore
    FLASK_AVAILABLE = True
except Exception:
    FLASK_AVAILABLE = False


def create_dummy_png(path: Path, size: int = 10):
    """Create a minimal PNG file for testing."""
    # Minimal PNG header and IHDR chunk for a tiny image; content doesn't matter for moves
    # To avoid complexity, just write some bytes; downstream code treats as files, not images
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(b"testdata")


def build_test_groups(tmp: Path):
    """Create three groups with staged filenames and return ImageGroup list."""
    # Dynamically load reviewer module (filename starts with a digit)
    module_path = PROJECT_ROOT / "scripts" / "01_ai_assisted_reviewer.py"
    spec = importlib.util.spec_from_file_location("ai_reviewer_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to load module spec for reviewer"
    spec.loader.exec_module(mod)  # type: ignore
    ImageGroup = mod.ImageGroup

    groups = []

    # Group A: 3 images, will approve index 2 (keep 1, delete 2)
    ga_dir = tmp / "groupA"
    a_imgs = [
        ga_dir / "20250101_000001_stage1_generated.png",
        ga_dir / "20250101_000001_stage1.5_face_swapped.png",
        ga_dir / "20250101_000001_stage2_upscaled.png",
    ]
    for p in a_imgs:
        create_dummy_png(p)
    groups.append(ImageGroup(group_id="20250101_000001", images=a_imgs, directory=ga_dir))

    # Group B: 2 images, will crop index 0 (keep 1 crop, delete 1)
    gb_dir = tmp / "groupB"
    b_imgs = [
        gb_dir / "20250101_000002_stage1_generated.png",
        gb_dir / "20250101_000002_stage2_upscaled.png",
    ]
    for p in b_imgs:
        create_dummy_png(p)
    groups.append(ImageGroup(group_id="20250101_000002", images=b_imgs, directory=gb_dir))

    # Group C: 4 images, unselected (delete all 4)
    gc_dir = tmp / "groupC"
    c_imgs = [
        gc_dir / f"20250101_000003_stage{i}_generated.png" for i in [1, 2, 3, 4]
    ]
    for p in c_imgs:
        create_dummy_png(p)
    groups.append(ImageGroup(group_id="20250101_000003", images=c_imgs, directory=gc_dir))

    return groups


def run_test_case():
    if not FLASK_AVAILABLE:
        print("[!] Flask not available - skipping batch counting test")
        return True
    # Import here to avoid Flask dependency during module import in other tests
    module_path = PROJECT_ROOT / "scripts" / "01_ai_assisted_reviewer.py"
    spec = importlib.util.spec_from_file_location("ai_reviewer_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to load module spec for reviewer"
    spec.loader.exec_module(mod)  # type: ignore
    build_app = mod.build_app

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Prepare groups and app directories
        groups = build_test_groups(tmp)
        selected_dir = tmp / "__selected"
        crop_dir = tmp / "__crop"
        delete_staging_dir = tmp / "__delete_staging"
        selected_dir.mkdir(exist_ok=True)
        crop_dir.mkdir(exist_ok=True)
        delete_staging_dir.mkdir(exist_ok=True)

        # Use sandbox tracker to avoid writing to logs
        tracker = FileTracker("ai_assisted_reviewer_test", sandbox=True)

        app = build_app(
            groups=groups,
            base_dir=tmp,
            tracker=tracker,
            selected_dir=selected_dir,
            crop_dir=crop_dir,
            delete_staging_dir=delete_staging_dir,
            ranker_model=None,
            crop_model=None,
            clip_info=None,
            batch_size=20,
        )

        client = app.test_client()

        # Prepare selections: Group A approve index 2; Group B crop index 0; Group C unselected
        payload = {
            "selections": [
                {"groupId": "20250101_000001", "selectedImage": 2, "crop": False},
                {"groupId": "20250101_000002", "selectedImage": 0, "crop": True},
            ]
        }

        resp = client.post("/process-batch", data=json.dumps(payload), content_type="application/json")
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}, body={resp.data!r}"
        data = resp.get_json()
        assert data["status"] == "ok"

        # Expected counts:
        # Group A: kept 1, deleted 2
        # Group B: kept 1 (crop), crop 1, deleted 1
        # Group C: kept 0, deleted 4 (unselected)
        # Totals: kept=2, crop=1, deleted=7, remaining=0
        assert data.get("kept") == 2, f"kept mismatch: {data}"
        assert data.get("crop") == 1, f"crop mismatch: {data}"
        assert data.get("deleted") == 7, f"deleted mismatch: {data}"
        assert data.get("remaining") == 0, f"remaining mismatch: {data}"

        return True


if __name__ == "__main__":
    ok = run_test_case()
    print("AI-Assisted Reviewer batch counting test:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


