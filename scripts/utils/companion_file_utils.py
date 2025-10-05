#!/usr/bin/env python3
"""
Standardized Companion File Utilities
====================================
Shared utilities for handling companion files with wildcard logic.
Any file with the same base name as an image will be considered a companion.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import time
import webbrowser
from io import BytesIO
from functools import lru_cache
import sys
from datetime import datetime, timedelta
from typing import Iterable
import csv
import json


try:
    from PIL import Image
except ImportError:
    Image = None

_SEND2TRASH_AVAILABLE = False
try:
    from send2trash import send2trash  # type: ignore
    _SEND2TRASH_AVAILABLE = True
except Exception:
    _SEND2TRASH_AVAILABLE = False


class Logger:
    """
    Centralized logging utility for consistent error/info messages across all scripts.
    """
    
    # Color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'SUCCESS': '\033[32m',  # Green
        'RESET': '\033[0m'      # Reset
    }
    
    # Message prefixes
    PREFIXES = {
        'DEBUG': '[DEBUG]',
        'INFO': '[*]',
        'WARNING': '[!]',
        'ERROR': '[!]',
        'SUCCESS': '✅'
    }
    
    def __init__(self, script_name: str = "script", enable_colors: bool = True):
        self.script_name = script_name
        self.enable_colors = enable_colors and sys.stdout.isatty()
    
    def _format_message(self, level: str, message: str) -> str:
        """Format message with appropriate prefix and color."""
        prefix = self.PREFIXES.get(level, '[*]')
        
        if self.enable_colors and level in self.COLORS:
            color = self.COLORS[level]
            reset = self.COLORS['RESET']
            return f"{color}{prefix}{reset} {message}"
        else:
            return f"{prefix} {message}"
    
    def debug(self, message: str):
        """Log debug message."""
        print(self._format_message('DEBUG', message))
    
    def info(self, message: str):
        """Log info message."""
        print(self._format_message('INFO', message))
    
    def warning(self, message: str):
        """Log warning message."""
        print(self._format_message('WARNING', message))
    
    def error(self, message: str):
        """Log error message."""
        print(self._format_message('ERROR', message), file=sys.stderr)
    
    def success(self, message: str):
        """Log success message."""
        print(self._format_message('SUCCESS', message))
    
    def error_with_exception(self, message: str, exception: Exception):
        """Log error message with exception details."""
        error_msg = f"{message}: {exception}"
        self.error(error_msg)
    
    def import_error(self, package: str):
        """Log import error with installation instructions."""
        self.error(f"{package} is required. Install with: pip install {package}")
    
    def directory_not_found(self, directory: str):
        """Log directory not found error."""
        self.error(f"Directory '{directory}' not found")
    
    def file_not_found(self, file_path: str):
        """Log file not found error."""
        self.error(f"File '{file_path}' not found")
    
    def operation_failed(self, operation: str, error: Exception):
        """Log operation failure."""
        self.error(f"{operation} failed: {error}")


# Global logger instance
logger = Logger()


def find_all_companion_files(image_path: Path) -> List[Path]:
    """
    Find ALL companion files with the same base name as the image (wildcard approach).
    
    This includes: .yaml, .content, .caption, .txt, .json, etc.
    Any file with the same base name as an image will be moved together.
    
    Args:
        image_path: Path to the main image file
        
    Returns:
        List of companion file paths
    """
    companions = []
    base_name = image_path.stem
    parent_dir = image_path.parent
    
    # Find ALL files in the same directory with the same base name
    for file_path in parent_dir.iterdir():
        if (file_path.is_file() and 
            file_path.stem == base_name and 
            file_path != image_path):  # Don't include the image file itself
            companions.append(file_path)
    
    return companions


def move_file_with_all_companions(src_path: Path, dst_dir: Path, dry_run: bool = False) -> List[str]:
    """
    Move image file and ALL its companion files, preserving original names.
    
    Args:
        src_path: Path to the main image file
        dst_dir: Destination directory
        dry_run: If True, only simulate the move operation
        
    Returns:
        List of moved file names
    """
    moved_files = []
    
    # Create destination directory if it doesn't exist
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Move the main image file
    dst_path = dst_dir / src_path.name
    if not dry_run and not dst_path.exists():
        import shutil
        shutil.move(str(src_path), str(dst_path))
        moved_files.append(src_path.name)
        print(f"Moved: {src_path.name}")
    elif dry_run:
        moved_files.append(src_path.name)
        print(f"[DRY RUN] Would move: {src_path.name}")
    
    # Move companion files
    companions = find_all_companion_files(src_path)
    for companion in companions:
        companion_dst = dst_dir / companion.name
        if not dry_run and not companion_dst.exists():
            import shutil
            shutil.move(str(companion), str(companion_dst))
            moved_files.append(companion.name)
            print(f"Moved companion: {companion.name}")
        elif dry_run:
            moved_files.append(companion.name)
            print(f"[DRY RUN] Would move companion: {companion.name}")
    
    return moved_files


def scan_images(folder: Path, extensions: List[str] = None, recursive: bool = True) -> List[Path]:
    """
    Scan for image files in a directory.
    
    Args:
        folder: Directory to scan
        extensions: List of extensions to look for (default: ['.png', '.jpg', '.jpeg'])
        recursive: Whether to scan subdirectories recursively
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']
    
    extensions = [ext.lower() for ext in extensions]
    
    if recursive:
        pattern = "**/*"
        scan_func = folder.rglob
    else:
        pattern = "*"
        scan_func = folder.glob
    
    image_files = []
    for file_path in scan_func(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def find_all_image_files(source_dir: Path) -> List[Path]:
    """
    Find all image files recursively in a directory.
    
    Args:
        source_dir: Directory to scan recursively
        
    Returns:
        List of all image file paths found
    """
    return scan_images(source_dir, recursive=True)


def detect_stage(filename: str) -> str:
    """
    Detect the stage of an image file based on filename patterns.
    
    Args:
        filename: The filename to analyze
        
    Returns:
        Stage name (e.g., 'stage1_generated', 'stage1.5_face_swapped', 'stage2_upscaled', 'stage3_enhanced')
    """
    filename_lower = filename.lower()
    
    if 'stage1_generated' in filename_lower:
        return 'stage1_generated'
    elif 'stage1.5_face_swapped' in filename_lower:
        return 'stage1.5_face_swapped'
    elif 'stage2_upscaled' in filename_lower:
        return 'stage2_upscaled'
    elif 'stage3_enhanced' in filename_lower:
        return 'stage3_enhanced'
    else:
        return 'unknown'


def extract_stage_name(filename: str) -> str:
    """
    Extract stage name from filename after second underscore.
    
    Args:
        filename: The filename to parse
        
    Returns:
        Stage name (e.g., 'stage2_upscaled')
    """
    parts = filename.split('_')
    if len(parts) >= 3:
        # Join everything after the second underscore
        stage_name = '_'.join(parts[2:])
        # Remove file extension
        stage_name = stage_name.rsplit('.', 1)[0]
        return stage_name
    return filename.rsplit('.', 1)[0]  # Fallback to filename without extension


def sort_image_files_by_timestamp_and_stage(image_files: List[Path]) -> List[Path]:
    """
    Sort image files by timestamp first, then stage order, then filename.
    
    This is the STANDARD sorting logic for all image processing tools.
    Ensures consistent ordering: stage1 → stage1.5 → stage2 → stage3
    
    Args:
        image_files: List of image file paths to sort
        
    Returns:
        Sorted list of image file paths
    """
    def sort_key(path: Path) -> Tuple[str, float, str]:
        timestamp = extract_timestamp_from_filename(path.name) or "99999999_999999"
        stage = detect_stage(path.name)
        stage_num = get_stage_number(stage)
        return (timestamp, stage_num, path.name)
    
    return sorted(image_files, key=sort_key)


def find_consecutive_stage_groups(files: List[Path],
                                  stage_of=lambda p: float(get_stage_number(detect_stage(p.name))),
                                  dt_of=lambda p: extract_datetime_from_filename(p.name),
                                  min_group_size=2,
                                  time_gap_minutes=None,
                                  lookahead=50) -> List[List[Path]]:
    """
    NEAREST-UP grouping:
      - Files MUST be pre-sorted by (timestamp, then stage).
      - A run starts anywhere.
      - Next pick = the smallest stage > prev_stage among the upcoming window.
      - Window ends when we hit a stage <= prev_stage (series reset),
        or the optional time_gap is exceeded, or we exhaust `lookahead` items.
      - Duplicates (same stage again) end the run.

    This prevents 1→3 "stealing" when a 2 exists a bit later, but still allows 1→3 if no 1.5/2 show up.
    
    Args:
        files: List of image file paths (must be pre-sorted by timestamp)
        stage_of: Function to extract stage number from file path
        dt_of: Function to extract timestamp from file path
        min_group_size: Minimum number of files required to form a group
        time_gap_minutes: Optional time window limit for grouping
        lookahead: Maximum number of files to look ahead
        
    Returns:
        List of groups, where each group is a list of consecutive stage files
        
    Example:
        Files: ['stage1_generated.png', 'stage3_enhanced.png', 'stage2_upscaled.png']
        Result: [['stage1_generated.png', 'stage2_upscaled.png', 'stage3_enhanced.png']]
        (waits for stage2 instead of jumping to stage3)
    """
    time_gap = (timedelta(minutes=time_gap_minutes) if time_gap_minutes else None)

    def boundary(prev_stage, prev_dt, item):
        s = stage_of(item)
        if s <= prev_stage:
            return True
        if time_gap and prev_dt and dt_of(item):
            if (dt_of(item) - prev_dt) > time_gap:
                return True
        return False

    groups = []
    n, i = len(files), 0

    while i < n:
        g = [files[i]]
        prev_s = stage_of(files[i])
        prev_dt = dt_of(files[i])
        i += 1

        while i < n:
            # scan forward window to collect candidates > prev_s until boundary/limits
            candidates = []
            k = i
            steps = 0
            while k < n and steps < lookahead and not boundary(prev_s, prev_dt, files[k]):
                s = stage_of(files[k])
                if s > prev_s:
                    candidates.append((s, k))
                k += 1
                steps += 1

            if not candidates:
                break  # no valid next stage in window

            # choose the smallest stage > prev_s, earliest occurrence
            min_s = min(candidates, key=lambda t: t[0])[0]
            chosen_k = next(k for s, k in candidates if s == min_s)

            g.append(files[chosen_k])
            prev_s = min_s
            prev_dt = dt_of(files[chosen_k])
            i = chosen_k + 1  # advance past chosen; earlier larger stages are intentionally skipped

        if len(g) >= min_group_size:
            groups.append(g)

    return groups


def get_stage_number(stage: str) -> float:
    """
    Extract numeric stage number from stage string.
    
    Args:
        stage: Stage string (e.g., 'stage1_generated', 'stage2_upscaled')
        
    Returns:
        Numeric stage number (e.g., 1.0, 1.5, 2.0, 3.0)
    """
    stage_lower = stage.lower()
    
    if 'stage1_generated' in stage_lower:
        return 1.0
    elif 'stage1.5' in stage_lower:
        return 1.5
    elif 'stage2' in stage_lower:
        return 2.0
    elif 'stage3' in stage_lower:
        return 3.0
    else:
        return 0.0


def find_mismatched_files(directory: Path) -> Dict[str, List[Path]]:
    """
    Find files that don't have matching companions in a directory.
    
    Args:
        directory: Directory to scan
        
    Returns:
        Dictionary with 'orphaned_images' and 'orphaned_metadata' lists
    """
    image_extensions = {'.png', '.jpg', '.jpeg'}
    metadata_extensions = {'.yaml', '.caption', '.content'}
    
    # Get all files
    all_files = list(directory.glob("*"))
    
    # Separate images and metadata
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in image_extensions]
    metadata_files = [f for f in all_files if f.is_file() and f.suffix.lower() in metadata_extensions]
    
    # Find orphaned images (no metadata)
    orphaned_images = []
    for img_file in image_files:
        base_name = img_file.stem
        has_companion = any(md_file.stem == base_name for md_file in metadata_files)
        if not has_companion:
            orphaned_images.append(img_file)
    
    # Find orphaned metadata (no image)
    orphaned_metadata = []
    for md_file in metadata_files:
        base_name = md_file.stem
        has_image = any(img_file.stem == base_name for img_file in image_files)
        if not has_image:
            orphaned_metadata.append(md_file)
    
    return {
        'orphaned_images': orphaned_images,
        'orphaned_metadata': orphaned_metadata
    }


def move_multiple_files_with_companions(image_files: List[Path], dest_dir: Path, dry_run: bool = False, tracker=None) -> dict:
    """
    Move multiple image files and all their companions to destination directory.
    
    This is a shared utility for scripts that need to move multiple files.
    
    Args:
        image_files: List of image file paths to move
        dest_dir: Destination directory
        dry_run: If True, only simulate the move operation
        tracker: Optional FileTracker instance for logging
        
    Returns:
        Dictionary with counts: {'moved': int, 'skipped': int, 'errors': int}
    """
    moved_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        # Check if destination already exists - if so, skip this file
        dest_image = dest_dir / image_file.name
        if not dry_run and dest_image.exists():
            print(f"[{i:3d}/{len(image_files)}] SKIPPING: {image_file.name} (already exists in destination)")
            skipped_count += 1
            continue
        
        print(f"[{i:3d}/{len(image_files)}] Moving: {image_file.name}")
        
        if not dry_run:
            try:
                # Use wildcard logic to move image and ALL companion files
                moved_files = move_file_with_all_companions(image_file, dest_dir, dry_run=False)
                
                # Log the operations if tracker provided
                if tracker:
                    for moved_file in moved_files:
                        tracker.log_operation("move", str(image_file.parent / moved_file), str(dest_dir / moved_file))
                
                moved_count += 1
                
            except Exception as e:
                print(f"❌ ERROR moving {image_file.name}: {e}")
                if tracker:
                    tracker.log_operation("error", str(image_file), "", notes=str(e))
                error_count += 1
        else:
            # Dry run - just simulate
            moved_files = move_file_with_all_companions(image_file, dest_dir, dry_run=True)
            moved_count += 1
    
    return {
        'moved': moved_count,
        'skipped': skipped_count, 
        'errors': error_count
    }


def safe_delete_paths(paths: Iterable[Path], hard_delete: bool = False, tracker=None) -> List[str]:
    """
    Delete or send a list of file paths to Trash. Optionally logs via tracker.

    Args:
        paths: Iterable of file paths to remove
        hard_delete: If True, permanently delete files. Otherwise use system Trash.
        tracker: Optional FileTracker-like object with log_operation(operation, source_dir, dest_dir, file_count, notes, files)

    Returns:
        List of deleted file names
    """
    deleted: List[str] = []

    if hard_delete:
        for p in paths:
            try:
                if p.exists():
                    p.unlink()
                    deleted.append(p.name)
            except Exception as exc:
                logger.error_with_exception(f"Failed to delete {p}", exc)
    else:
        if not _SEND2TRASH_AVAILABLE:
            raise RuntimeError(
                "send2trash is not installed. Install with: pip install send2trash\n"
                "Or set hard_delete=True to permanently delete files (dangerous)."
            )
        for p in paths:
            try:
                if p.exists():
                    send2trash(str(p))
                    deleted.append(p.name)
            except Exception as exc:
                logger.error_with_exception(f"Failed to send to Trash {p}", exc)

    if tracker and deleted:
        source_dir = str(Path(paths.__iter__().__next__()).parent.name) if paths else "unknown"  # best-effort
        try:
            tracker.log_operation(
                operation="delete" if hard_delete else "send_to_trash",
                source_dir=source_dir,
                dest_dir="trash" if not hard_delete else "",
                file_count=len(deleted),
                files=deleted,
                notes="Deleted by shared safe_delete_paths",
            )
        except Exception:
            pass  # tracker is best-effort

    return deleted


def safe_delete_image_and_yaml(png_path: Path, hard_delete: bool = False, tracker=None) -> List[str]:
    """
    Delete an image and its .yaml companion (if present). Uses Trash by default.

    Args:
        png_path: Path to the .png image
        hard_delete: Permanently delete if True, else send to Trash
        tracker: Optional FileTracker-like object

    Returns:
        List of deleted file names
    """
    files: List[Path] = [png_path]
    yaml_path = png_path.with_suffix('.yaml')
    if yaml_path.exists():
        files.append(yaml_path)
    return safe_delete_paths(files, hard_delete=hard_delete, tracker=tracker)


# ============================
# Training data logging helpers
# ============================

def get_training_dir() -> Path:
    """Ensure and return the training data directory."""
    d = Path("data") / "training"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_csv_row(csv_path: Path, header: list, row: dict) -> None:
    """Append a row to CSV with header creation. Fail-open."""
    try:
        need_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if need_header:
                w.writeheader()
            w.writerow(row)
    except Exception as exc:
        logger.error_with_exception(f"Training log write failed: {csv_path}", exc)


def log_select_crop_entry(
    session_id: str,
    set_id: str,
    directory: str,
    image_paths: list,
    image_stages: list,
    image_sizes: list,
    chosen_index: int,
    crop_norm: Optional[Tuple[float, float, float, float]],
) -> None:
    """Log a single select+crop supervision row. crop_norm is None if delete-all.

    image_sizes: list of (width, height)
    """
    td = get_training_dir()
    csv_path = td / "select_crop_log.csv"
    from datetime import datetime as _dt
    header = [
        "session_id","set_id","directory","image_count","chosen_index","chosen_path",
        "crop_x1","crop_y1","crop_x2","crop_y2","timestamp",
    ]
    # dynamic fields for each image
    for i in range(len(image_paths)):
        header += [f"image_{i}_path", f"image_{i}_stage", f"width_{i}", f"height_{i}"]

    row = {
        "session_id": session_id,
        "set_id": set_id,
        "directory": directory,
        "image_count": len(image_paths),
        "chosen_index": chosen_index,
        "chosen_path": image_paths[chosen_index] if 0 <= chosen_index < len(image_paths) else "",
        "crop_x1": crop_norm[0] if crop_norm else "",
        "crop_y1": crop_norm[1] if crop_norm else "",
        "crop_x2": crop_norm[2] if crop_norm else "",
        "crop_y2": crop_norm[3] if crop_norm else "",
        "timestamp": _dt.utcnow().isoformat() + "Z",
    }
    for i, p in enumerate(image_paths):
        w, h = image_sizes[i] if i < len(image_sizes) else ("", "")
        row[f"image_{i}_path"] = p
        row[f"image_{i}_stage"] = image_stages[i] if i < len(image_stages) else ""
        row[f"width_{i}"] = w
        row[f"height_{i}"] = h

    _append_csv_row(csv_path, header, row)


def log_selection_only_entry(
    session_id: str,
    set_id: str,
    chosen_path: str,
    negative_paths: list,
) -> None:
    """Log a selection-only row from the web selector."""
    td = get_training_dir()
    csv_path = td / "selection_only_log.csv"
    from datetime import datetime as _dt
    header = ["session_id","set_id","chosen_path","neg_paths","timestamp"]
    row = {
        "session_id": session_id,
        "set_id": set_id,
        "chosen_path": chosen_path,
        "neg_paths": json.dumps(negative_paths),
        "timestamp": _dt.utcnow().isoformat() + "Z",
    }
    _append_csv_row(csv_path, header, row)


@lru_cache(maxsize=2048)
def generate_thumbnail(image_path: str, mtime_ns: int, file_size: int, max_dim: int = 200, quality: int = 85) -> bytes:
    """
    Generate thumbnail with caching based on file modification time and size.
    
    Args:
        image_path: Path to the image file
        mtime_ns: File modification time in nanoseconds (for cache invalidation)
        file_size: File size in bytes (for cache invalidation)
        max_dim: Maximum dimension for thumbnail (default: 200)
        quality: JPEG quality (default: 85)
        
    Returns:
        Thumbnail as JPEG bytes
        
    Raises:
        ImportError: If PIL is not available
        Exception: If thumbnail generation fails
    """
    if Image is None:
        raise ImportError("PIL (Pillow) is required for thumbnail generation")
    
    try:
        with Image.open(image_path) as img:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            return buffer.getvalue()
    except Exception as e:
        logger.error_with_exception(f"Error generating thumbnail for {image_path}", e)
        # Re-raise the exception so calling code knows it failed
        raise


def get_error_display_html() -> str:
    """
    Get HTML for a persistent error display system.
    Returns HTML that can be inserted into web templates.
    """
    return """
    <!-- Error Display System -->
    <div id="error-container" style="display: none; position: fixed; top: 0; left: 0; right: 0; z-index: 9999;">
        <div id="error-bar" style="background: #dc3545; color: white; padding: 12px 20px; margin: 0; border-bottom: 2px solid #c82333; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto;">
                <div style="flex: 1; font-family: monospace; font-size: 14px; line-height: 1.4;">
                    <strong>⚠️ Error:</strong> <span id="error-message">Error message will appear here</span>
                </div>
                <button id="error-dismiss" onclick="dismissError()" style="background: #c82333; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; margin-left: 15px;">
                    ✕ Dismiss
                </button>
            </div>
        </div>
    </div>
    
    <script>
        function showError(message) {
            const container = document.getElementById('error-container');
            const messageSpan = document.getElementById('error-message');
            
            messageSpan.textContent = message;
            container.style.display = 'block';
            
            // Scroll to top to show error
            window.scrollTo(0, 0);
        }
        
        function dismissError() {
            const container = document.getElementById('error-container');
            container.style.display = 'none';
        }
        
        // Auto-dismiss after 30 seconds (optional - can be removed if you prefer manual only)
        let errorTimeout;
        function showErrorWithAutoDismiss(message) {
            showError(message);
            clearTimeout(errorTimeout);
            errorTimeout = setTimeout(dismissError, 30000); // 30 seconds
        }
    </script>
    """


def format_image_display_name(filename: str, max_length: int = 30, context: str = "web") -> str:
    """
    Format image filename for display with consistent truncation and timestamp removal.
    
    Args:
        filename: The full filename (e.g., "20250725_035504_stage2_upscaled.png")
        max_length: Maximum length for display (default: 30)
        context: Display context - "web", "desktop", or "compact" (default: "web")
        
    Returns:
        Formatted display name (e.g., "stage2_upscaled")
    """
    # Remove file extension
    name_without_ext = Path(filename).stem
    
    # Extract timestamp (first 15 characters: YYYYMMDD_HHMMSS)
    timestamp_match = re.match(r'^(\d{8}_\d{6})_', name_without_ext)
    if timestamp_match:
        # Remove timestamp prefix (including the trailing underscore)
        display_name = name_without_ext[16:]  # Skip "YYYYMMDD_HHMMSS_"
    else:
        display_name = name_without_ext
    
    # Handle different contexts
    if context == "desktop":
        # For desktop tools, keep it shorter
        max_length = min(max_length, 25)
    elif context == "compact":
        # For compact displays
        max_length = min(max_length, 20)
    
    # Truncate if too long
    if len(display_name) > max_length:
        display_name = display_name[:max_length-3] + "..."
    
    # Ensure we have something to display
    if not display_name.strip():
        display_name = filename[:max_length-3] + "..."
    
    return display_name


def extract_timestamp_from_filename(filename: str) -> Optional[str]:
    """
    Extract timestamp from filename (YYYYMMDD_HHMMSS format).
    
    Args:
        filename: The filename to extract timestamp from
        
    Returns:
        Timestamp string (e.g., "20250725_035504") or None if not found
    """
    match = re.match(r'^(\d{8}_\d{6})', filename)
    return match.group(1) if match else None


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract datetime object from filename (YYYYMMDD_HHMMSS format).
    
    Args:
        filename: The filename to extract datetime from
        
    Returns:
        Datetime object or None if not found
    """
    timestamp_str = extract_timestamp_from_filename(filename)
    if not timestamp_str:
        return None
    
    try:
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def get_date_from_timestamp(timestamp_str: Optional[str]) -> Optional[str]:
    """
    Extract date part from timestamp string.
    
    Args:
        timestamp_str: Timestamp string (e.g., "20250725_035504")
        
    Returns:
        Date string (e.g., "20250725") or None if invalid
    """
    if not timestamp_str or len(timestamp_str) != 15:
        return None
    return timestamp_str.split("_")[0]


def extract_base_timestamp(filename: str) -> Optional[str]:
    """
    Extract the base timestamp from a filename (without stage suffix).
    This is a compatibility function for scripts that need just the timestamp.
    
    Args:
        filename: The filename to extract timestamp from
        
    Returns:
        Base timestamp string (e.g., "20250726_010033") or None if not found
    """
    return extract_timestamp_from_filename(filename)


def calculate_work_time_from_file_operations(file_operations: List[Dict], break_threshold_minutes: int = 5) -> float:
    """
    Calculate work time based on file operation timestamps with intelligent break detection.
    
    This function analyzes FileTracker logs to determine actual work time by:
    1. Measuring time between file operations
    2. Detecting breaks (gaps longer than threshold)
    3. Only counting time when actively working
    
    Args:
        file_operations: List of file operation dictionaries from FileTracker
        break_threshold_minutes: Minutes of inactivity considered a break (default: 5)
        
    Returns:
        Total work time in seconds
        
    Example:
        operations = [
            {"timestamp": "2025-01-15T14:30:15.123Z", "operation": "move"},
            {"timestamp": "2025-01-15T14:32:45.456Z", "operation": "delete"},
            {"timestamp": "2025-01-15T14:45:00.789Z", "operation": "move"}  # 12+ min gap = break
        ]
        work_time = calculate_work_time_from_file_operations(operations)
        # Returns: ~2.5 minutes (time between first two operations)
    """
    if not file_operations or len(file_operations) < 2:
        return 0.0
    
    # Sort operations by timestamp to ensure chronological order
    sorted_ops = sorted(file_operations, key=lambda x: x.get('timestamp', ''))
    
    total_work_time = 0.0
    break_threshold_seconds = break_threshold_minutes * 60
    
    for i in range(len(sorted_ops) - 1):
        try:
            current_time = datetime.fromisoformat(sorted_ops[i]['timestamp'].replace('Z', '+00:00'))
            next_time = datetime.fromisoformat(sorted_ops[i + 1]['timestamp'].replace('Z', '+00:00'))
            
            gap_seconds = (next_time - current_time).total_seconds()
            
            # If gap is less than threshold, count as work time
            # If gap is more than threshold, likely a break - don't count
            if gap_seconds <= break_threshold_seconds:
                total_work_time += gap_seconds
            else:
                logger.debug(f"Break detected: {gap_seconds/60:.1f} minutes between operations")
                
        except (ValueError, KeyError) as e:
            logger.warning(f"Error parsing timestamp in file operation: {e}")
            continue
    
    return total_work_time


def get_file_operation_metrics(file_operations: List[Dict]) -> Dict[str, any]:
    """
    Get comprehensive metrics from file operations for dashboard display.
    
    Args:
        file_operations: List of file operation dictionaries from FileTracker
        
    Returns:
        Dictionary with metrics including work time, operation counts, etc.
    """
    if not file_operations:
        return {
            'total_work_time_seconds': 0.0,
            'total_work_time_minutes': 0.0,
            'total_operations': 0,
            'operation_types': {},
            'files_processed': 0,
            'session_duration_seconds': 0.0,
            'efficiency_score': 0.0
        }
    
    # Calculate work time
    work_time_seconds = calculate_work_time_from_file_operations(file_operations)
    
    # Count operations by type
    operation_types = {}
    total_files = 0
    
    for op in file_operations:
        op_type = op.get('operation', 'unknown')
        operation_types[op_type] = operation_types.get(op_type, 0) + 1
        
        # Count files processed
        file_count = op.get('file_count', 0)
        if isinstance(file_count, (int, float)):
            total_files += file_count
    
    # Calculate session duration (first to last operation)
    sorted_ops = sorted(file_operations, key=lambda x: x.get('timestamp', ''))
    if len(sorted_ops) >= 2:
        try:
            start_time = datetime.fromisoformat(sorted_ops[0]['timestamp'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(sorted_ops[-1]['timestamp'].replace('Z', '+00:00'))
            session_duration = (end_time - start_time).total_seconds()
        except (ValueError, KeyError):
            session_duration = 0.0
    else:
        session_duration = 0.0
    
    # Calculate efficiency score (files per minute of work time)
    work_time_minutes = work_time_seconds / 60.0
    efficiency_score = total_files / work_time_minutes if work_time_minutes > 0 else 0.0
    
    return {
        'total_work_time_seconds': work_time_seconds,
        'total_work_time_minutes': work_time_minutes,
        'total_operations': len(file_operations),
        'operation_types': operation_types,
        'files_processed': total_files,
        'session_duration_seconds': session_duration,
        'efficiency_score': efficiency_score
    }


def timestamp_to_minutes(timestamp_str: Optional[str]) -> Optional[float]:
    """
    Convert timestamp string to minutes since midnight.
    
    Args:
        timestamp_str: Timestamp string (e.g., "20250725_035504")
        
    Returns:
        Minutes since midnight (e.g., 215.07) or None if invalid
    """
    if not timestamp_str or len(timestamp_str) != 15:
        return None
    try:
        time_part = timestamp_str.split("_")[1]
        hours = int(time_part[:2])
        minutes = int(time_part[2:4])
        seconds = int(time_part[4:6])
        return hours * 60 + minutes + seconds / 60.0
    except (ValueError, IndexError):
        return None


def launch_browser(host: str, port: int, delay: float = 1.2) -> None:
    """
    Launch browser to the specified host and port after a delay.
    
    Args:
        host: Host address (only launches for localhost/127.0.0.1/0.0.0.0)
        port: Port number
        delay: Delay in seconds before launching (default: 1.2)
    """
    # Only launch for local addresses
    if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        return
    
    url = f"http://{host}:{port}/"
    time.sleep(delay)
    
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.error_with_exception("Failed to launch browser", e)


# Legacy function names for backward compatibility
def find_companion_files(image_path: Path) -> List[Path]:
    """Legacy function name - use find_all_companion_files instead."""
    return find_all_companion_files(image_path)


def move_file_with_companions(src_path: Path, dst_dir: Path, dry_run: bool = False) -> List[str]:
    """Legacy function name - use move_file_with_all_companions instead."""
    return move_file_with_all_companions(src_path, dst_dir, dry_run)
