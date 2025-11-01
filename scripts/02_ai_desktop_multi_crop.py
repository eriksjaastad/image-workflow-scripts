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

OPTIONAL FLAGS (read first):
----------------------------
  --defer-crop              # Do NOT crop now; queue for later processing. Saves
                            # selections/coords and moves files to __crop_queued/.
  --no-preload              # Skip AI preload of crop rectangles (faster batch loading)
  --no-ai-logging           # Disable AI training data capture (SQLite v3)

USAGE:
------
Run on a directory containing PNGs (or subdirectories). Typical target is `__crop_auto/`.
  python scripts/02_ai_desktop_multi_crop.py __crop_auto/
  python scripts/02_ai_desktop_multi_crop.py __crop/          # also works on regular crop dir
  python scripts/02_ai_desktop_multi_crop.py __selected/      # single directory mode

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
  to the central "__cropped/" directory at the repository root
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
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except Exception:
    raise RuntimeError("Pillow is required. Install with: pip install pillow")


# Import the existing MultiCropTool and base behavior
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util as _il_util

from utils.companion_file_utils import move_file_with_all_companions
from utils.error_monitoring import (
    get_error_monitor,
    monitor_errors,
    validate_data_quality,
)


def _ensure_interactive_matplotlib_backend() -> None:
    """Ensure an interactive Matplotlib backend is selected before pyplot import.

    Preference order on macOS: MacOSX → Qt5Agg → TkAgg. On other OS: Qt5Agg → TkAgg → MacOSX.
    If MPLBACKEND is already set, respect it. Best-effort only; non-fatal on failure.
    """
    if os.environ.get("MPLBACKEND"):
        return

    candidates = ["Qt5Agg", "TkAgg", "MacOSX"]
    if sys.platform == "darwin":
        candidates = ["MacOSX", "Qt5Agg", "TkAgg"]

    try:
        import matplotlib  # type: ignore

        for name in candidates:
            try:
                matplotlib.use(name, force=True)  # Must be before pyplot import
                os.environ["MPLBACKEND"] = name
                print(f"[Matplotlib] Using backend: {name}")
                return
            except Exception:
                continue
    except Exception:
        # matplotlib not installed or other import error; archive module may handle/raise
        pass

    print(
        "[Matplotlib] Warning: interactive backend not configured; default may be Agg. "
        "Set MPLBACKEND=MacOSX or install PyQt5 for Qt5Agg if UI does not open."
    )


_ensure_interactive_matplotlib_backend()

candidate_paths = [
    Path(__file__).parent / "02_desktop_multi_crop.py",
    Path(__file__).parent / "archive" / "04_desktop_multi_crop.py",
]
_module_path = next((p for p in candidate_paths if p.exists()), None)
if _module_path is None:  # pragma: no cover
    searched = ", ".join(str(p) for p in candidate_paths)
    raise ImportError(f"Could not find base MultiCropTool; looked in: {searched}")

_spec = _il_util.spec_from_file_location("desktop_multi_crop", str(_module_path))
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load MultiCropTool from {_module_path}")
_desktop_multi_crop = _il_util.module_from_spec(_spec)
_spec.loader.exec_module(_desktop_multi_crop)  # type: ignore
MultiCropTool = _desktop_multi_crop.MultiCropTool  # type: ignore
from datetime import UTC

from utils.ai_crop_utils import (  # type: ignore
    decision_matches_image,
    normalize_and_clamp_rect,
)


class AIMultiCropTool(MultiCropTool):
    """Desktop multi-crop that preloads AI crop rectangles when available."""

    def __init__(self, *args, queue_mode=False, preload_ai=True, **kwargs):
        """Initialize with optional queue mode and AI preload control."""
        # Set flags BEFORE calling parent, since parent __init__ calls load_batch()
        self.queue_mode = queue_mode
        self.preload_ai = preload_ai
        self.queue_manager = None
        # Hide verbose perf logs from base selector updates for this tool
        self.suppress_perf_logs = True
        super().__init__(*args, **kwargs)

        if self.queue_mode:
            from file_tracker import FileTracker
            from utils.crop_queue import CropQueueManager

            # Queue manager defaults to safe zone (data/ai_data/crop_queue/)
            self.queue_manager = CropQueueManager()
            self.tracker = FileTracker("ai_desktop_multi_crop_queue")
            print("[Queue Mode] Enabled - crops will be queued for later processing")
            print(f"[Queue Mode] Queue file: {self.queue_manager.queue_file}")
        else:
            # Normal mode still logs file operations for metrics
            import logging

            try:
                from file_tracker import FileTracker as _FT

                self.tracker = _FT("ai_desktop_multi_crop")
                print("[FileTracker] Initialized for ai_desktop_multi_crop")
            except Exception as e:
                error_monitor = get_error_monitor("ai_desktop_multi_crop")
                error_monitor.validation_error(
                    "FileTracker initialization failed - refusing to proceed without audit trail",
                    {"exception": str(e)},
                )
                logging.getLogger(__name__).exception(
                    "FileTracker initialization failed - refusing to proceed without audit trail"
                )
                raise RuntimeError(
                    "FileTracker initialization failed: refusing to proceed without audit trail"
                ) from e

    # ---- Lightweight UI alert helpers ----
    def _show_alert(self, message: str, color: str = "red") -> None:
        import logging

        logger = logging.getLogger(__name__)
        try:
            if hasattr(self, "_alert_artist") and self._alert_artist is not None:
                try:
                    self._alert_artist.remove()
                except Exception:
                    pass
                self._alert_artist = None
            self._alert_artist = self.fig.text(
                0.5,
                0.02,
                message,
                ha="center",
                va="bottom",
                color=color,
                fontsize=11,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#fff3cd",
                    edgecolor=color,
                    alpha=0.95,
                ),
            )
            self.fig.canvas.draw_idle()
        except Exception:
            logger.warning("Failed to show UI alert: %s", message, exc_info=True)

    def _clear_alert(self) -> None:
        import logging

        logger = logging.getLogger(__name__)
        try:
            if hasattr(self, "_alert_artist") and self._alert_artist is not None:
                try:
                    self._alert_artist.remove()
                except Exception:
                    pass
                self._alert_artist = None
                self.fig.canvas.draw_idle()
        except Exception:
            logger.warning("Failed to clear UI alert", exc_info=True)

    def crop_and_save(self, image_info, crop_coords):
        """Crop image, save in place, then move to central __cropped directory (or queue)."""
        if self.queue_mode and self.queue_manager:
            # Queue mode: record operation AND update decisions DB
            png_path = image_info["path"]

            try:
                from utils.standard_paths import get_cropped_dir

                cropped_dir = get_cropped_dir()
            except Exception:
                from pathlib import Path as _Path

                cropped_dir = _Path(__file__).parent.parent / "__cropped"

            # Compute normalized coordinates (DB source of truth)
            x1, y1, x2, y2 = crop_coords
            width = image_info["image"].width
            height = image_info["image"].height

            # VALIDATE: Ensure crop coordinates are valid (prevent silent corruption)
            validate_data_quality(
                "crop coordinates",
                (x1, y1, x2, y2, width, height),
                lambda coords: (
                    coords[0] >= 0
                    and coords[1] >= 0  # x1, y1 >= 0
                    and coords[2] > coords[0]
                    and coords[3] > coords[1]  # x2 > x1, y2 > y1
                    and coords[2] <= coords[4]
                    and coords[3] <= coords[5]  # within image bounds
                ),
                f"Invalid crop coordinates: ({x1}, {y1}, {x2}, {y2}) for image {width}x{height}",
            )

            # VALIDATE: Ensure image dimensions are valid (catch the 0x0 bug)
            validate_data_quality(
                "image dimensions",
                (width, height),
                lambda dims: dims[0] > 0 and dims[1] > 0,
                f"Invalid image dimensions: {width}x{height} - this will corrupt training data!",
            )

            normalized_coords = [x1 / width, y1 / height, x2 / width, y2 / height]

            # Try to update decisions DB with final crop
            try:
                from utils.ai_training_decisions_v3 import (
                    init_decision_db,
                    update_decision_with_crop,
                )

                # Look for .decision file to get group_id
                decision_path = png_path.with_suffix(".decision")
                if decision_path.exists():
                    with open(decision_path) as f:
                        decision_data = json.load(f)

                    group_id = decision_data.get("group_id")
                    project_id = getattr(
                        self, "project_id", decision_data.get("project_id", "unknown")
                    )

                    if group_id:
                        # Initialize/get DB path
                        db_path = init_decision_db(project_id)

                        # Update with final crop coordinates (normalized)
                        update_decision_with_crop(db_path, group_id, normalized_coords)
            except Exception as e:
                # Don't fail the crop if DB update fails
                print(f"[Queue] Warning: couldn't update decisions DB: {e}")

            # Store absolute paths for queue
            crop_entry = {
                "source_path": str(png_path.absolute()),
                "crop_rect": list(crop_coords),  # Pixel coords for convenience
                "crop_rect_normalized": normalized_coords,  # DB source of truth
                "dest_directory": str(cropped_dir.absolute()),
                "image_width": width,
                "image_height": height,
                "index_in_batch": 0,  # Will be set by submit_batch_to_queue
            }

            # Store in instance variable to be batched
            if not hasattr(self, "_queue_batch"):
                self._queue_batch = []
            self._queue_batch.append(crop_entry)

            return  # Don't do actual processing

        # Normal mode: Do the crop ourselves (don't call parent which moves to _cropped subdir)
        from PIL import Image

        png_path = image_info["path"]
        x1, y1, x2, y2 = crop_coords

        # Load and crop image
        original_size = png_path.stat().st_size if png_path.exists() else -1
        img = Image.open(png_path)
        original_dimensions = img.size
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_dimensions = cropped_img.size

        # Save over the original file (in place)
        cropped_img.save(png_path)
        # Verify output exists and is non-empty
        if not png_path.exists():
            raise RuntimeError(f"Crop failed: output file does not exist: {png_path}")
        cropped_size = png_path.stat().st_size
        if cropped_size == 0:
            raise RuntimeError(f"Crop failed: output file is 0 bytes: {png_path}")
        # Verify dimensions changed unless full-image crop
        if cropped_dimensions == original_dimensions and (x1, y1, x2, y2) != (
            0,
            0,
            *original_dimensions,
        ):
            raise RuntimeError(
                f"Crop failed: dimensions unchanged despite crop coords: {png_path}"
            )
        # Heuristic: size unchanged is suspicious unless full-image crop
        if (
            original_size >= 0
            and cropped_size == original_size
            and (x1, y1, x2, y2) != (0, 0, *original_dimensions)
        ):
            raise RuntimeError(
                f"Crop failed: file size unchanged after crop (coords={(x1, y1, x2, y2)}, path={png_path})"
            )
        print(f"Cropped and saved in place: {png_path.name}")
        # Log crop operation (create/update) - fail-fast if tracking is enabled
        if getattr(self, "tracker", None) is not None:
            self.tracker.log_operation(
                "crop",
                source_dir=str(png_path.parent),
                dest_dir=str(png_path.parent),
                file_count=1,
                files=[png_path.name],
            )

        # Update SQLite database with final crop coordinates
        try:
            from utils.ai_training_decisions_v3 import (
                init_decision_db,
                update_decision_with_crop,
            )

            decision_path = png_path.with_suffix(".decision")
            if decision_path.exists():
                with open(decision_path) as f:
                    decision_data = json.load(f)

                group_id = decision_data.get("group_id")
                project_id = getattr(
                    self, "project_id", decision_data.get("project_id", "unknown")
                )

                if group_id:
                    # Initialize/get DB path
                    db_path = init_decision_db(project_id)

                    # Normalize crop coordinates (original_size is (width, height) tuple)
                    width, height = image_info["original_size"]
                    normalized_coords = [
                        x1 / width,
                        y1 / height,
                        x2 / width,
                        y2 / height,
                    ]

                    # Update with final crop coordinates (normalized)
                    update_decision_with_crop(db_path, group_id, normalized_coords)
                    print(f"[SQLite] Updated decision: {group_id}")
        except Exception as e:
            # Don't fail the crop if DB update fails
            print(f"[!] Warning: couldn't update decisions DB: {e}")

        # Now move from original location to central __cropped directory
        try:
            from utils.standard_paths import get_cropped_dir

            cropped_dir = get_cropped_dir()
            cropped_dir.mkdir(exist_ok=True)
        except Exception:
            from pathlib import Path as _Path

            cropped_dir = _Path(__file__).parent.parent / "__cropped"
            cropped_dir.mkdir(exist_ok=True)

        try:
            moved_files = move_file_with_all_companions(
                png_path, cropped_dir, dry_run=False
            )
            try:
                count = len([f for f in moved_files if str(f).lower().endswith(".png")])
            except Exception:
                count = len(moved_files)
            print(f"[*] Centralized {count} file(s) to {cropped_dir.name}/")
            self._clear_alert()
            # Log move to __cropped
            try:
                if getattr(self, "tracker", None) is not None:
                    self.tracker.log_operation(
                        "move",
                        source_dir=str(png_path.parent),
                        dest_dir=str(cropped_dir),
                        file_count=count,
                    )
            except Exception:
                pass
        except Exception as e:
            msg = str(e)
            print(f"[!] Error centralizing to {cropped_dir.name}: {msg}")
            if "COMPANION POLICY VIOLATION" in msg:
                self._show_alert(
                    f"Companion files missing for {png_path.name}. Cropped file left in place.",
                    color="red",
                )
            else:
                self._show_alert(
                    f"Move error: {png_path.name} → {cropped_dir.name}. See console for details.",
                    color="red",
                )

    def load_batch(self):
        # Load batch normally (sets up selectors and default full-image crops)
        super().load_batch()
        # Snapshot total batches on FIRST load only (when files are populated)
        if not hasattr(self, "_total_batches_snapshot"):
            try:
                total_images = len(getattr(self, "png_files", []))
                images_per_batch = max(1, getattr(self, "images_per_batch", 3))
                self._total_batches_snapshot = max(
                    1, (total_images + images_per_batch - 1) // images_per_batch
                )
                self._batches_completed = 0
            except Exception:
                self._total_batches_snapshot = None
                self._batches_completed = 0
        # After images are loaded and selectors created, try to preload AI crops
        if not self.preload_ai:
            print("[AI Multi-Crop] Skipping AI preload (--no-preload)")
            return

        preloaded_count = 0
        failed_count = 0
        for i, image_info in enumerate(self.current_images):
            try:
                png_path: Path = image_info.get("path")  # type: ignore
                if not isinstance(png_path, Path):
                    continue
                decision_path = png_path.with_suffix(".decision")
                if not decision_path.exists():
                    continue

                with open(decision_path) as f:
                    data = json.load(f)

                # Optional validation: ensure decision references this image when provided
                if not decision_matches_image(data, png_path.name):
                    print(
                        f"[AI preload] Warning: {decision_path.name} does not reference {png_path.name}"
                    )
                    continue

                ai_coords = data.get("ai_crop_coords")
                if (
                    not ai_coords
                    or not isinstance(ai_coords, (list, tuple))
                    or len(ai_coords) != 4
                ):
                    continue

                # Convert normalized coords to pixels
                with Image.open(png_path) as img:
                    w, h = img.size

                rect = normalize_and_clamp_rect(ai_coords, w, h)
                if rect is None:
                    print(
                        f"[AI preload] Warning: invalid box after clamp for {png_path.name}"
                    )
                    failed_count += 1
                    continue
                x1, y1, x2, y2 = rect

                # Update selector and internal state to the AI suggestion
                if i < len(self.selectors) and self.selectors[i] is not None:
                    self.apply_crop_rect(i, x1, y1, x2, y2)

                self.image_states[i]["crop_coords"] = (x1, y1, x2, y2)
                self.image_states[i]["has_selection"] = True
                self.image_states[i]["action"] = None  # keep/crop by default
                preloaded_count += 1
            except Exception as e:
                failed_count += 1
                print(
                    f"[AI preload] Warning: failed to preload crop for {image_info.get('path')}: {e}"
                )

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
                print(
                    f"[AI Multi-Crop] ✓ Preloaded {preloaded_count} AI crop suggestions"
                )
            if failed_count > 0:
                print(f"[AI Multi-Crop] ⚠ Failed to load {failed_count} suggestions")
        except Exception:
            pass

    def submit_batch(self):
        """Override submit_batch to flush queue in queue mode."""
        if self.queue_mode and self.queue_manager and hasattr(self, "_queue_batch"):
            # Submit accumulated crops to queue
            if self._queue_batch:
                # Set index_in_batch for each entry
                for idx, crop_entry in enumerate(self._queue_batch):
                    crop_entry["index_in_batch"] = idx

                # Get session/project info
                from datetime import datetime

                session_id = getattr(
                    self,
                    "session_id",
                    # Use UTC to ensure consistent session IDs across timezones/DST
                    f"crop_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                )

                project_id = getattr(self, "project_id", "unknown")

                # Enqueue the batch
                batch_id = self.queue_manager.enqueue_batch(
                    crops=self._queue_batch,
                    session_id=session_id,
                    project_id=project_id,
                )

                print(
                    f"[Queue] Batch {batch_id} queued ({len(self._queue_batch)} crops)"
                )

                # Move source files to __crop_queued directory to get them out of the way
                try:
                    from pathlib import Path as _Path

                    queued_dir = _Path(__file__).parent.parent / "__crop_queued"
                    queued_dir.mkdir(exist_ok=True)

                    total_moved = 0
                    for crop_entry in self._queue_batch:
                        source_path = _Path(crop_entry["source_path"])
                        if source_path.exists():
                            try:
                                moved_files = move_file_with_all_companions(
                                    source_path, queued_dir, dry_run=False
                                )
                                total_moved += len(moved_files)
                                # Silently move files - no per-file output in queue mode
                            except Exception as e:
                                print(
                                    f"[Queue] Warning: couldn't move {source_path.name}: {e}"
                                )

                    # Log the move operation
                    if hasattr(self, "tracker") and total_moved > 0:
                        self.tracker.log_operation(
                            "move",
                            dest_dir=str(queued_dir),
                            file_count=total_moved,
                            notes=f"batch={batch_id}, queued_for_processing",
                        )
                except Exception as e:
                    print(f"[Queue] Warning: file movement error: {e}")

                # Clear the batch
                self._queue_batch = []

        # Call parent submit_batch (which handles progress tracking and loading next batch)
        try:
            # Increment batches completed before parent potentially reloads state
            if hasattr(self, "_batches_completed"):
                self._batches_completed += 1
        except Exception:
            pass
        super().submit_batch()

    # ---- UI: Title/Progress (Batch X/Y) ----
    def update_title(self):
        """Show batch-based progress instead of per-image counts."""
        try:
            # Expect base class/progress tracker to expose these attributes
            # Fallbacks ensure no crash if running older base module
            images_per_batch = max(1, getattr(self, "images_per_batch", 3))
            # Stable denominator captured at first load; fallback to live calc if missing
            if getattr(self, "_total_batches_snapshot", None):
                total_batches = int(self._total_batches_snapshot)
            else:
                total_images_live = len(getattr(self, "png_files", [])) or len(
                    getattr(self, "current_images", [])
                )
                total_batches = max(
                    1, (total_images_live + images_per_batch - 1) // images_per_batch
                )
            # Numerator based on batches completed in this session
            batches_done = int(getattr(self, "_batches_completed", 0))
            remaining_batches = max(0, total_batches - batches_done)

            aspect_info = (
                " • [🔒 LOCKED] Aspect Ratio"
                if getattr(self, "aspect_ratio", None)
                else ""
            )
            title = f"Batch {batches_done}/{remaining_batches} • [Enter] Submit • [Q] Quit{aspect_info}"
            self.fig.suptitle(title, fontsize=12, y=0.98)
            self.fig.canvas.draw_idle()
        except Exception:
            # Safe fallback to base behavior if anything changes upstream
            try:
                super().update_title()
            except Exception:
                pass


@monitor_errors("ai_desktop_multi_crop")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-assisted desktop multi-crop (preloads AI crop rectangles)"
    )
    parser.add_argument(
        "directory",
        help="Directory containing PNG images (or parent of subdirectories)",
    )
    parser.add_argument(
        "--no-ai-logging", action="store_true", help="Disable AI training data capture"
    )
    parser.add_argument(
        "--defer-crop",
        action="store_true",
        help="Do not crop now; write decisions/coords and queue for later processing",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip AI preload of crop rectangles for faster batch transitions",
    )
    args = parser.parse_args()

    tool = AIMultiCropTool(
        args.directory,
        aspect_ratio=None,
        enable_ai_logging=(not args.no_ai_logging),
        queue_mode=args.defer_crop,
        preload_ai=(not args.no_preload),
    )
    tool.run()

    # Heartbeat: successful completion with basic counters if available
    try:
        import logging

        files_processed = len(getattr(tool, "png_files", [])) or len(
            getattr(tool, "current_images", [])
        )
        logging.getLogger(__name__).info(
            "ai_desktop_multi_crop completed successfully",
            extra={"files_processed": files_processed},
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
