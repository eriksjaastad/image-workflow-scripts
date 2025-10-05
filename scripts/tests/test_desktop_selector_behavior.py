#!/usr/bin/env python3
"""
Desktop Selector Behavior Tests
Verifies Enter and Reset behaviors without user interaction.
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from importlib.machinery import SourceFileLoader


def _load_desktop_tool(project_root: Path):
    """Dynamically load DesktopImageSelectorCrop from its script file."""
    script_path = project_root / "scripts/01_desktop_image_selector_crop.py"
    mod = SourceFileLoader("desktop_selector_crop", str(script_path)).load_module()
    return mod.DesktopImageSelectorCrop


def _make_image(path: Path, size=(32, 32), color=(128, 128, 128)):
    from PIL import Image
    img = Image.new("RGB", size, color)
    img.save(path)


def test_enter_no_selection_deletes_all_and_advances():
    project_root = Path(__file__).parent.parent.parent
    DesktopImageSelectorCrop = _load_desktop_tool(project_root)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # Create a pair that will group together
        _make_image(d / "20250705_000000_stage2_upscaled.png")
        _make_image(d / "20250705_000010_stage3_enhanced.png")

        # Instantiate tool (headless backend is handled in base tool)
        tool = DesktopImageSelectorCrop(d, aspect_ratio=None, reset_progress=True)

        # Stub out deletion/cropping side-effects
        deletions = []
        def fake_delete(png_path, yaml_path):
            deletions.append(png_path.name)
        tool.safe_delete = fake_delete  # type: ignore

        crops = []
        def fake_crop(image_info, crop_coords):
            crops.append(image_info['path'].name)
        tool.crop_and_save = fake_crop  # type: ignore

        # Ensure all statuses are delete (default); then submit
        for st in tool.image_states:
            st['status'] = 'delete'

        current_index_before = tool.progress_tracker.current_triplet_index
        tool.submit_batch()

        # All images should be deleted; no crops; and we should advance
        assert len(deletions) == len(tool.current_images)
        assert len(crops) == 0
        assert tool.progress_tracker.current_triplet_index == current_index_before + 1


def test_reset_sets_all_delete_and_full_crops():
    project_root = Path(__file__).parent.parent.parent
    DesktopImageSelectorCrop = _load_desktop_tool(project_root)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _make_image(d / "20250705_000000_stage2_upscaled.png")
        _make_image(d / "20250705_000010_stage3_enhanced.png")

        tool = DesktopImageSelectorCrop(d, aspect_ratio=None, reset_progress=True)

        # Mark one as selected and set a partial crop to verify reset clears it
        tool.image_states[0]['status'] = 'selected'
        tool.image_states[0]['has_selection'] = True
        tool.image_states[0]['crop_coords'] = (1, 1, 10, 10)

        tool.reset_entire_row()

        for i, st in enumerate(tool.image_states):
            assert st['status'] == 'delete'
            assert st['action'] is None
            # extents should match full image (x1, y1, x2, y2) format
            w, h = tool.current_images[i]['original_size']
            x1, y1, x2, y2 = tool.selectors[i].extents if tool.selectors[i] else (0, 0, 0, 0)
            assert (x1, y1, x2, y2) == (0, 0, w, h), f"Expected full image extents (0, 0, {w}, {h}), got ({x1}, {y1}, {x2}, {y2})"


def test_selected_crops_others_deleted_then_advance():
    project_root = Path(__file__).parent.parent.parent
    DesktopImageSelectorCrop = _load_desktop_tool(project_root)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _make_image(d / "20250705_000000_stage2_upscaled.png")
        _make_image(d / "20250705_000010_stage3_enhanced.png")

        tool = DesktopImageSelectorCrop(d, aspect_ratio=None, reset_progress=True)

        # Select first; set a crop
        tool.image_states[0]['status'] = 'selected'
        tool.image_states[0]['has_selection'] = True
        # full crop to avoid Pillow issues
        w, h = tool.current_images[0]['original_size']
        tool.image_states[0]['crop_coords'] = (0, 0, w, h)

        deletions = []
        def fake_delete(png_path, yaml_path):
            deletions.append(png_path.name)
        tool.safe_delete = fake_delete  # type: ignore

        crops = []
        def fake_crop(image_info, crop_coords):
            crops.append(image_info['path'].name)
        tool.crop_and_save = fake_crop  # type: ignore

        before = tool.progress_tracker.current_triplet_index
        tool.submit_batch()

        assert crops == [tool.current_images[0]['path'].name]
        assert len(deletions) == len(tool.current_images) - 1
        assert tool.progress_tracker.current_triplet_index == before + 1


