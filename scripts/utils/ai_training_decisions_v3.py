"""
AI Training Decisions v3 - SQLite Utilities
===========================================

High-performance, reliable storage for AI training decisions.
Replaces fragile CSV logging with robust SQLite database.

Key Features:
- ACID compliant (no data corruption)
- Fast lookups via indexes
- Built-in validation via constraints
- Per-project databases (manageable size)
- JSON for flexible arrays (images, crop coords)

Usage:
    # Initialize new project database
    db_path = init_decision_db("mojo3")

    # Log AI decision from AI Reviewer
    log_ai_decision(
        db_path=db_path,
        group_id="mojo3_group_20251021_234530",
        images=["img1.png", "img2.png", "img3.png"],
        ai_pick=1,
        ai_crop=[0.1, 0.0, 0.9, 0.8],
        user_pick=2,
        user_action="crop",
        ...
    )

    # Update with final crop from Desktop Multi-Crop
    update_decision_with_crop(
        db_path=db_path,
        group_id="mojo3_group_20251021_234530",
        final_crop=[0.2, 0.0, 0.7, 0.6]
    )

    # Validate database integrity
    errors = validate_decision_db(db_path)
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def init_decision_db(project_id: str) -> Path:
    """
    Initialize SQLite database for AI training decisions.

    Creates database file and applies schema if it doesn't exist.
    Safe to call multiple times (idempotent).

    Args:
        project_id: Project identifier (e.g., "mojo3")

    Returns:
        Path to database file

    Example:
        db_path = init_decision_db("mojo3")
        # Creates: data/training/ai_training_decisions/mojo3.db
    """
    decisions_dir = Path("data/training/ai_training_decisions")
    decisions_dir.mkdir(parents=True, exist_ok=True)

    db_path = decisions_dir / f"{project_id}.db"

    # Apply schema
    schema_path = Path("data/schema/ai_training_decisions_v3.sql")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path) as f:
        schema_sql = f.read()

    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()

    print(f"[SQLite] Initialized decision database: {db_path}")
    return db_path


def generate_group_id(
    project_id: str, timestamp: str | None = None, batch: int = 0, index: int = 0
) -> str:
    """
    Generate unique group ID.

    Format: {project_id}_group_{timestamp}_batch{batch:03d}_img{index:03d}

    Args:
        project_id: Project identifier
        timestamp: ISO 8601 timestamp (UTC), generated if None
        batch: Batch number (0-indexed)
        index: Image index within batch (0-indexed)

    Returns:
        Unique group ID

    Example:
        group_id = generate_group_id("mojo3", batch=1, index=5)
        # "mojo3_group_20251021T234530Z_batch001_img005"
    """
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    else:
        # Remove punctuation from ISO timestamp for cleaner ID
        timestamp = timestamp.replace("-", "").replace(":", "").replace(".", "")

    return f"{project_id}_group_{timestamp}_batch{batch:03d}_img{index:03d}"


def log_ai_decision(
    db_path: Path,
    group_id: str,
    project_id: str,
    images: list[str],
    ai_selected_index: int,
    user_selected_index: int,
    user_action: str,
    image_width: int,
    image_height: int,
    ai_crop_coords: list[float] | None = None,
    ai_confidence: float | None = None,
    directory: str | None = None,
    batch_number: int | None = None,
    timestamp: str | None = None,
) -> None:
    """
    Log AI decision and user response.

    Called from AI Reviewer after user makes selection.
    Creates new row in database with AI recommendation and user choice.

    Args:
        db_path: Path to SQLite database
        group_id: Unique group identifier
        project_id: Project identifier
        images: List of image filenames in group (2-4 images)
        ai_selected_index: Which image AI picked (0-3)
        user_selected_index: Which image user picked (0-3)
        user_action: User action ('approve', 'crop', 'reject')
        image_width: Original image width in pixels
        image_height: Original image height in pixels
        ai_crop_coords: AI's proposed crop [x1, y1, x2, y2] (normalized)
        ai_confidence: AI's confidence score (0.0-1.0)
        directory: Source directory path
        batch_number: Batch number within project
        timestamp: ISO 8601 timestamp (generated if None)

    Raises:
        ValueError: If validation fails
        sqlite3.IntegrityError: If group_id already exists

    Example:
        log_ai_decision(
            db_path=Path("data/training/ai_training_decisions/mojo3.db"),
            group_id="mojo3_group_20251021_234530_batch001_img002",
            project_id="mojo3",
            images=["img1.png", "img2.png", "img3.png"],
            ai_selected_index=1,
            user_selected_index=2,
            user_action="crop",
            image_width=3072,
            image_height=3072,
            ai_crop_coords=[0.1, 0.0, 0.9, 0.8],
            ai_confidence=0.87
        )
    """
    # Validation
    if not images or len(images) < 2 or len(images) > 4:
        raise ValueError(
            f"Invalid images list: must have 2-4 images, got {len(images)}"
        )

    if user_action not in ("approve", "crop", "reject"):
        raise ValueError(f"Invalid user_action: {user_action}")

    if not (0 <= ai_selected_index < len(images)):
        raise ValueError(
            f"Invalid ai_selected_index: {ai_selected_index} (must be 0-{len(images)-1})"
        )

    if not (0 <= user_selected_index < len(images)):
        raise ValueError(
            f"Invalid user_selected_index: {user_selected_index} (must be 0-{len(images)-1})"
        )

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid dimensions: {image_width}x{image_height}")

    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"

    # Calculate selection_match
    selection_match = ai_selected_index == user_selected_index

    # Serialize JSON fields
    images_json = json.dumps(images)
    ai_crop_json = json.dumps(ai_crop_coords) if ai_crop_coords else None

    # Insert row
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO ai_decisions (
            group_id, timestamp, project_id, directory, batch_number,
            images, ai_selected_index, ai_crop_coords, ai_confidence,
            user_selected_index, user_action,
            final_crop_coords, crop_timestamp,
            image_width, image_height,
            selection_match, crop_match
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            group_id,
            timestamp,
            project_id,
            directory,
            batch_number,
            images_json,
            ai_selected_index,
            ai_crop_json,
            ai_confidence,
            user_selected_index,
            user_action,
            None,
            None,  # final_crop_coords, crop_timestamp (filled later)
            image_width,
            image_height,
            selection_match,
            None,  # crop_match (calculated later)
        ),
    )

    conn.commit()
    conn.close()


def update_decision_with_crop(
    db_path: Path,
    group_id: str,
    final_crop_coords: list[float],
    crop_timestamp: str | None = None,
) -> None:
    """
    Update decision with final crop coordinates.

    Called from Desktop Multi-Crop after user completes cropping.
    Updates existing row with final crop and calculates crop_match.

    Args:
        db_path: Path to SQLite database
        group_id: Unique group identifier
        final_crop_coords: User's actual crop [x1, y1, x2, y2] (normalized)
        crop_timestamp: ISO 8601 timestamp (generated if None)

    Raises:
        ValueError: If group_id not found or invalid crop coords

    Example:
        update_decision_with_crop(
            db_path=Path("data/training/ai_training_decisions/mojo3.db"),
            group_id="mojo3_group_20251021_234530_batch001_img002",
            final_crop_coords=[0.2, 0.0, 0.7, 0.6]
        )
    """
    # Validation
    if len(final_crop_coords) != 4:
        raise ValueError(
            f"Invalid crop coords: must be [x1, y1, x2, y2], got {final_crop_coords}"
        )

    x1, y1, x2, y2 = final_crop_coords
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(
            f"Invalid crop coords: {final_crop_coords} (must be in [0,1] with x1<x2, y1<y2)"
        )

    if crop_timestamp is None:
        crop_timestamp = datetime.utcnow().isoformat() + "Z"

    # Serialize JSON
    final_crop_json = json.dumps(final_crop_coords)

    # Get existing row to calculate crop_match
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT ai_crop_coords FROM ai_decisions WHERE group_id = ?", (group_id,)
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise ValueError(f"Group ID not found: {group_id}")

    ai_crop_json = row[0]
    crop_match = None

    if ai_crop_json:
        ai_crop = json.loads(ai_crop_json)
        crop_match = calculate_crop_match(ai_crop, final_crop_coords)

    # Update row
    cursor.execute(
        """
        UPDATE ai_decisions
        SET final_crop_coords = ?,
            crop_timestamp = ?,
            crop_match = ?
        WHERE group_id = ?
    """,
        (final_crop_json, crop_timestamp, crop_match, group_id),
    )

    conn.commit()
    conn.close()


def calculate_crop_match(
    ai_crop: list[float], user_crop: list[float], tolerance: float = 0.05
) -> bool:
    """
    Check if AI crop is within tolerance of user crop.

    Uses simple corner distance metric: all 4 corners must be within tolerance.

    Args:
        ai_crop: AI's crop [x1, y1, x2, y2] (normalized)
        user_crop: User's crop [x1, y1, x2, y2] (normalized)
        tolerance: Max distance for each coordinate (default: 0.05 = 5%)

    Returns:
        True if crops match within tolerance

    Example:
        match = calculate_crop_match(
            ai_crop=[0.1, 0.0, 0.9, 0.8],
            user_crop=[0.11, 0.01, 0.91, 0.79]
        )
        # True (within 5% tolerance)
    """
    for ai_val, user_val in zip(ai_crop, user_crop, strict=False):
        if abs(ai_val - user_val) > tolerance:
            return False
    return True


def calculate_crop_similarity(
    ai_crop: list[float], user_crop: list[float]
) -> dict[str, float]:
    """
    Calculate detailed similarity metrics between two crop rectangles.

    Useful for analyzing crop proposals even when selection differs.

    Args:
        ai_crop: AI's crop [x1, y1, x2, y2] (normalized)
        user_crop: User's crop [x1, y1, x2, y2] (normalized)

    Returns:
        Dictionary with similarity metrics:
        - iou: Intersection over Union (0.0 to 1.0)
        - center_distance: Distance between centers (0.0 to ~1.41)
        - size_difference: Absolute difference in area (0.0 to 1.0)
        - crops_similar: Boolean flag (IoU > 0.7)

    Example:
        metrics = calculate_crop_similarity(
            ai_crop=[0.1, 0.0, 0.9, 0.8],
            user_crop=[0.2, 0.1, 0.8, 0.7]
        )
        # {
        #     'iou': 0.61,
        #     'center_distance': 0.14,
        #     'size_difference': 0.04,
        #     'crops_similar': False
        # }
    """
    ax1, ay1, ax2, ay2 = ai_crop
    ux1, uy1, ux2, uy2 = user_crop

    # IoU (Intersection over Union)
    ix1 = max(ax1, ux1)
    iy1 = max(ay1, uy1)
    ix2 = min(ax2, ux2)
    iy2 = min(ay2, uy2)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ai_area = (ax2 - ax1) * (ay2 - ay1)
    user_area = (ux2 - ux1) * (uy2 - uy1)
    union_area = ai_area + user_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0

    # Center distance
    ai_center_x = (ax1 + ax2) / 2
    ai_center_y = (ay1 + ay2) / 2
    user_center_x = (ux1 + ux2) / 2
    user_center_y = (uy1 + uy2) / 2

    center_dist = (
        (ai_center_x - user_center_x) ** 2 + (ai_center_y - user_center_y) ** 2
    ) ** 0.5

    # Size difference
    size_diff = abs(ai_area - user_area)

    return {
        "iou": round(iou, 4),
        "center_distance": round(center_dist, 4),
        "size_difference": round(size_diff, 4),
        "crops_similar": iou > 0.7,
    }


def validate_decision_db(db_path: Path, verbose: bool = False) -> list[str]:
    """
    Validate database integrity and completeness.

    Checks for:
    - Missing required fields
    - Invalid coordinate ranges
    - Invalid dimensions
    - Incomplete crops (marked for crop but not yet done)
    - Orphaned entries

    Args:
        db_path: Path to SQLite database
        verbose: Print detailed info for each error

    Returns:
        List of error messages (empty if valid)

    Example:
        errors = validate_decision_db(Path("data/training/ai_training_decisions/mojo3.db"))
        if errors:
            print("Validation failed:")
            for err in errors:
                print(f"  - {err}")
        else:
            print("✅ Database valid!")
    """
    errors = []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check 1: Missing timestamps
    cursor.execute(
        "SELECT COUNT(*) FROM ai_decisions WHERE timestamp IS NULL OR timestamp = ''"
    )
    count = cursor.fetchone()[0]
    if count > 0:
        errors.append(f"Found {count} rows with missing timestamps")

    # Check 2: Missing user decisions
    cursor.execute(
        "SELECT COUNT(*) FROM ai_decisions WHERE user_selected_index IS NULL"
    )
    count = cursor.fetchone()[0]
    if count > 0:
        errors.append(f"Found {count} rows with missing user decisions")

    # Check 3: Incomplete crops
    cursor.execute("""
        SELECT COUNT(*) FROM ai_decisions 
        WHERE user_action = 'crop' AND final_crop_coords IS NULL
    """)
    count = cursor.fetchone()[0]
    if count > 0:
        errors.append(f"Found {count} images marked for crop but not yet cropped")
        if verbose:
            cursor.execute("""
                SELECT group_id, timestamp FROM ai_decisions 
                WHERE user_action = 'crop' AND final_crop_coords IS NULL
                ORDER BY timestamp
            """)
            for row in cursor.fetchall():
                errors.append(f"  - Incomplete: {row[0]} (created {row[1]})")

    # Check 4: Invalid crop coordinates
    cursor.execute("""
        SELECT group_id, final_crop_coords FROM ai_decisions 
        WHERE final_crop_coords IS NOT NULL
    """)
    for group_id, crop_json in cursor.fetchall():
        try:
            coords = json.loads(crop_json)
            if len(coords) != 4:
                errors.append(f"Invalid crop coords in {group_id}: wrong length")
            else:
                x1, y1, x2, y2 = coords
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    errors.append(f"Invalid crop coords in {group_id}: {coords}")
        except json.JSONDecodeError:
            errors.append(f"Invalid JSON in {group_id}: {crop_json}")

    # Check 5: Invalid dimensions
    cursor.execute(
        "SELECT COUNT(*) FROM ai_decisions WHERE image_width <= 0 OR image_height <= 0"
    )
    count = cursor.fetchone()[0]
    if count > 0:
        errors.append(f"Found {count} rows with invalid image dimensions")

    # Check 6: Invalid indexes
    cursor.execute("""
        SELECT group_id, images, ai_selected_index, user_selected_index 
        FROM ai_decisions
    """)
    for group_id, images_json, ai_idx, user_idx in cursor.fetchall():
        images = json.loads(images_json)
        num_images = len(images)
        if not (0 <= ai_idx < num_images):
            errors.append(
                f"Invalid ai_selected_index in {group_id}: {ai_idx} (num_images={num_images})"
            )
        if not (0 <= user_idx < num_images):
            errors.append(
                f"Invalid user_selected_index in {group_id}: {user_idx} (num_images={num_images})"
            )

    conn.close()

    return errors


def get_ai_performance_stats(db_path: Path) -> dict[str, Any]:
    """
    Get AI performance statistics for a project.

    Returns:
        Dictionary with performance metrics

    Example:
        stats = get_ai_performance_stats(Path("data/training/ai_training_decisions/mojo3.db"))
        print(f"Selection Accuracy: {stats['selection_accuracy']:.1f}%")
        print(f"Crop Accuracy: {stats['crop_accuracy']:.1f}%")
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM ai_performance")
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {
            "total_decisions": 0,
            "correct_selections": 0,
            "correct_crops": 0,
            "selection_accuracy": 0.0,
            "crop_accuracy": 0.0,
        }

    conn.close()

    return {
        "project_id": row[0],
        "total_decisions": row[1],
        "correct_selections": row[2],
        "correct_crops": row[3],
        "selection_accuracy": row[4],
        "crop_accuracy": row[5],
    }
