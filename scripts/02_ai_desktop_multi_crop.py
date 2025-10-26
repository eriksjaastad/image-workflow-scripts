#!/usr/bin/env python3
"""
04-AI Desktop Multi-Crop
=========================

Purpose-built variant of the Desktop Multi-Crop tool that preloads AI crop
suggestions from sidecar `.decision` files (`ai_crop_coords`) for each image,
while preserving full manual editing and the original workflow.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Run on a directory containing PNGs (or subdirectories). Typical target is `crop_auto/`.
  python scripts/02_ai_desktop_multi_crop.py crop_auto/
  python scripts/02_ai_desktop_multi_crop.py crop/            # also works on regular crop dir
  python scripts/02_ai_desktop_multi_crop.py selected/        # single directory mode

Optional flags:
  --aspect-ratio 16:9     # lock crop to aspect ratio (e.g., 1:1, 4:3, 16:9)
  --no-ai-logging         # disable AI training data capture (SQLite v3)

FEATURES:
---------
• Preloads AI crop rectangles from `.decision` sidecars when present
• Falls back safely when sidecar or coords are missing/invalid
• Full manual editing retained (drag/resize, reset per image)
• Public API (apply_crop_rect) used to apply rectangles reliably
• Coordinate clamping ensures at least 1px width/height and in-bounds
• Clear console feedback (preloaded/failed counts, warnings)
• Optional training-data updates via SQLite v3 after cropping

PROGRESS TRACKING:
------------------
• Reuses MultiCropTool behavior:
  - Single-directory mode (process one folder), or
  - Multi-directory mode with session persistence in data/crop_progress/
• Batch-based UI (up to 3 images side-by-side)

WORKFLOW POSITION:
------------------
Step 1: Selection/AI Reviewer → scripts/01_ai_assisted_reviewer.py
Step 2: Cropping → THIS SCRIPT (scripts/02_ai_desktop_multi_crop.py)
Step 3: Review/QA → scripts/05_web_multi_directory_viewer.py

FILE HANDLING:
--------------
• This tool crops images in place, then moves the cropped image (and companions)
  to a sibling "_cropped" directory (e.g., crop/dalia_hannah → crop/dalia_hannah_cropped)
• Companions are moved together to preserve integrity

FILE SAFETY:
------------
See Documents/FILE_SAFETY_SYSTEM.md. This is the ONLY tool permitted to write
new image content (crop). All other tools must move/copy/delete only.

CONTROLS:
---------
Image 1: [1] Delete  [X] Reset crop
Image 2: [2] Delete  [C] Reset crop
Image 3: [3] Delete  [V] Reset crop

Global:  [Enter] Submit Batch  [Q] Quit  [←] Previous Batch
Multi-Directory: [N] Next Directory  [P] Previous Directory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
except Exception:
    raise RuntimeError("Pillow is required. Install with: pip install pillow")


# Import the existing MultiCropTool and base behavior
import sys
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util as _il_util  # noqa: E402
from importlib.machinery import SourceFileLoader as _SrcLoader  # noqa: E402
from utils.companion_file_utils import move_file_with_all_companions  # noqa: E402

_module_path = Path(__file__).parent / "02_desktop_multi_crop.py"
_spec = _il_util.spec_from_file_location("desktop_multi_crop", str(_module_path))
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load MultiCropTool from {_module_path}")
_desktop_multi_crop = _il_util.module_from_spec(_spec)
_spec.loader.exec_module(_desktop_multi_crop)  # type: ignore
MultiCropTool = _desktop_multi_crop.MultiCropTool  # type: ignore
from utils.ai_crop_utils import normalize_and_clamp_rect, decision_matches_image  # type: ignore  # noqa: E402


class AIMultiCropTool(MultiCropTool):
    """Desktop multi-crop that preloads AI crop rectangles when available."""
    
    def crop_and_save(self, image_info, crop_coords):
        """Crop image, save in place, then move to central __cropped directory."""
        # Perform the actual crop/save using parent implementation
        super().crop_and_save(image_info, crop_coords)

        # After saving, move the file (and companions) to the central __cropped directory
        try:
            from utils.standard_paths import get_cropped_dir
            cropped_dir = get_cropped_dir()
            cropped_dir.mkdir(exist_ok=True)
        except Exception:
            from pathlib import Path as _Path
            cropped_dir = _Path(__file__).parent.parent / "__cropped"
            cropped_dir.mkdir(exist_ok=True)

        try:
            png_path = image_info['path']
            moved_files = move_file_with_all_companions(png_path, cropped_dir, dry_run=False)
            try:
                count = len([f for f in moved_files if str(f).lower().endswith('.png')])
            except Exception:
                count = len(moved_files)
            print(f"[*] Moved {count} file(s) to {cropped_dir.name}/")
        except Exception as e:
            print(f"[!] Error moving files to {cropped_dir.name}: {e}")

    def load_batch(self):
        # Load batch normally (sets up selectors and default full-image crops)
        super().load_batch()

        # After images are loaded and selectors created, try to preload AI crops
        preloaded_count = 0
        failed_count = 0
        for i, image_info in enumerate(self.current_images):
            try:
                png_path: Path = image_info.get('path')  # type: ignore
                if not isinstance(png_path, Path):
                    continue
                decision_path = png_path.with_suffix('.decision')
                if not decision_path.exists():
                    continue

                with open(decision_path, 'r') as f:
                    data = json.load(f)

                # Optional validation: ensure decision references this image when provided
                if not decision_matches_image(data, png_path.name):
                    print(f"[AI preload] Warning: {decision_path.name} does not reference {png_path.name}")
                    continue

                ai_coords = data.get('ai_crop_coords')
                if not ai_coords or not isinstance(ai_coords, (list, tuple)) or len(ai_coords) != 4:
                    continue

                # Convert normalized coords to pixels
                with Image.open(png_path) as img:
                    w, h = img.size

                rect = normalize_and_clamp_rect(ai_coords, w, h)
                if rect is None:
                    print(f"[AI preload] Warning: invalid box after clamp for {png_path.name}")
                    failed_count += 1
                    continue
                x1, y1, x2, y2 = rect

                # Update selector and internal state to the AI suggestion
                if i < len(self.selectors) and self.selectors[i] is not None:
                    self.apply_crop_rect(i, x1, y1, x2, y2)

                self.image_states[i]['crop_coords'] = (x1, y1, x2, y2)
                self.image_states[i]['has_selection'] = True
                self.image_states[i]['action'] = None  # keep/crop by default
                preloaded_count += 1
            except Exception as e:
                failed_count += 1
                print(f"[AI preload] Warning: failed to preload crop for {image_info.get('path')}: {e}")

        # Refresh titles/labels after preloading
        try:
            self.update_image_titles(self.image_states)
            self.update_control_labels()
            import matplotlib.pyplot as plt  # local import to avoid early backend issues
            plt.draw()
        except Exception as e:
            print(f"[AI preload] Warning: UI refresh failed: {e}")

        # User feedback summary
        try:
            total = len(self.current_images)
            print(f"[AI Multi-Crop] Loaded {total} images")
            if preloaded_count > 0:
                print(f"[AI Multi-Crop] ✓ Preloaded {preloaded_count} AI crop suggestions")
            if failed_count > 0:
                print(f"[AI Multi-Crop] ⚠ Failed to load {failed_count} suggestions")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-assisted desktop multi-crop (preloads AI crop rectangles)")
    parser.add_argument("directory", help="Directory containing PNG images (or parent of subdirectories)")
    parser.add_argument("--aspect-ratio", help="Target aspect ratio (e.g., '16:9', '4:3', '1:1')")
    parser.add_argument("--no-ai-logging", action="store_true", help="Disable AI training data capture")
    args = parser.parse_args()

    tool = AIMultiCropTool(args.directory, aspect_ratio=args.aspect_ratio, enable_ai_logging=(not args.no_ai_logging))
    tool.run()


if __name__ == "__main__":
    main()


