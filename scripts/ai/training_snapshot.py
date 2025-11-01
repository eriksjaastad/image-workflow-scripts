#!/usr/bin/env python3
"""
AI Training Snapshot Utility
=============================

Lightweight utility for capturing training data (images + decisions).
Designed for ZERO performance impact on production tools.

SAFETY GUARANTEES:
- Read-only on source images (copies only)
- Async writes (non-blocking)
- Fail-safe (errors don't break tools)
- NEW files only in data/ai_data/training_snapshots/

Usage:
    from scripts.ai.training_snapshot import capture_crop_decision
    
    capture_crop_decision(
        image_path="/path/to/image.png",
        crop_coords=(x1, y1, x2, y2),
        action="cropped"  # or "deleted", "skipped"
    )
"""

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

# Single thread pool for all async operations (reused across calls)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai_snapshot")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "ai_data" / "training_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _get_image_id(image_path: Path) -> str:
    """Generate stable ID for image (read-only)"""
    stat = image_path.stat()
    hash_input = f"{image_path.name}_{stat.st_size}_{stat.st_mtime_ns}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def _save_snapshot_async(
    image_path: Path,
    crop_coords: tuple[float, float, float, float] | None,
    action: str,
    metadata: dict[str, Any]
) -> None:
    """Background task to save snapshot (non-blocking)"""
    try:
        # Generate snapshot ID
        snapshot_id = _get_image_id(image_path)
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
        
        # Create snapshot directory
        snapshot_path = SNAPSHOT_DIR / snapshot_id
        snapshot_path.mkdir(exist_ok=True)
        
        # Copy image (read-only operation on source)
        if image_path.exists():
            dest_image = snapshot_path / f"{image_path.name}"
            if not dest_image.exists():  # Skip if already copied
                shutil.copy2(image_path, dest_image)
        
        # Save decision metadata
        decision_file = snapshot_path / "decision.json"
        decision_data = {
            "snapshot_id": snapshot_id,
            "timestamp": timestamp,
            "source_path": str(image_path),
            "action": action,
            "crop_coords": crop_coords,
            "image_size": metadata.get("image_size"),
            "tool": metadata.get("tool", "multi_crop_tool"),
            "session_id": metadata.get("session_id"),
        }
        
        with open(decision_file, 'w') as f:
            json.dump(decision_data, f, indent=2)
    
    except Exception:
        # Silently fail - don't break the tool
        # Could log to separate error file if needed
        pass


def capture_crop_decision(
    image_path: Path,
    crop_coords: tuple[float, float, float, float] | None,
    action: str = "cropped",
    image_size: tuple[int, int] | None = None,
    session_id: str | None = None,
    tool: str = "multi_crop_tool"
) -> None:
    """
    Capture a crop decision for AI training (async, non-blocking).
    
    Args:
        image_path: Path to source image (read-only)
        crop_coords: (x1, y1, x2, y2) normalized coords [0-1], or None if deleted
        action: "cropped", "deleted", or "skipped"
        image_size: (width, height) of original image
        session_id: Optional session identifier
        tool: Tool name for logging
    
    Returns immediately (work happens in background thread).
    """
    if not image_path.exists():
        return
    
    # Prepare metadata
    metadata = {
        "image_size": image_size,
        "session_id": session_id,
        "tool": tool
    }
    
    # Submit to background thread (returns immediately)
    _executor.submit(
        _save_snapshot_async,
        image_path,
        crop_coords,
        action,
        metadata
    )


def get_snapshot_count() -> int:
    """Get count of captured snapshots (for progress tracking)"""
    try:
        return len([d for d in SNAPSHOT_DIR.iterdir() if d.is_dir()])
    except Exception:
        return 0


def cleanup_snapshots_older_than_days(days: int = 30) -> int:
    """
    Clean up old snapshots (optional maintenance).
    Returns count of deleted snapshots.
    """
    import time
    cutoff_time = time.time() - (days * 86400)
    deleted = 0
    
    try:
        for snapshot_dir in SNAPSHOT_DIR.iterdir():
            if snapshot_dir.is_dir():
                decision_file = snapshot_dir / "decision.json"
                if decision_file.exists():
                    mtime = decision_file.stat().st_mtime
                    if mtime < cutoff_time:
                        shutil.rmtree(snapshot_dir)
                        deleted += 1
    except Exception:
        pass
    
    return deleted


if __name__ == '__main__':
    # Test/demo
    pass

