#!/usr/bin/env python3
"""
AI Crop Utilities - Helper functions for AI crop coordinate handling.
"""

from typing import Dict, List, Optional, Tuple


def normalize_and_clamp_rect(
    rect: List[float],
    width: int,
    height: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert normalized [0,1] crop coordinates to pixel coordinates with validation.

    Args:
        rect: Normalized crop rectangle [x1, y1, x2, y2] in range [0, 1]
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Tuple of (x1, y1, x2, y2) in pixels, or None if invalid
    """
    if not rect or len(rect) != 4:
        return None

    try:
        # Convert to floats
        x1_norm, y1_norm, x2_norm, y2_norm = [float(v) for v in rect]

        # Validate normalized values are in [0, 1] range
        if not all(0 <= v <= 1 for v in [x1_norm, y1_norm, x2_norm, y2_norm]):
            return None

        # Ensure x1 < x2 and y1 < y2
        if x1_norm >= x2_norm or y1_norm >= y2_norm:
            return None

        # Convert to pixel coordinates
        x1 = int(x1_norm * width)
        y1 = int(y1_norm * height)
        x2 = int(x2_norm * width)
        y2 = int(y2_norm * height)

        # Clamp to image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(1, min(x2, width))
        y2 = max(1, min(y2, height))

        # Ensure at least 1px width and height
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1

        # Final clamp after adjustment
        x2 = min(x2, width)
        y2 = min(y2, height)

        return (x1, y1, x2, y2)

    except (ValueError, TypeError):
        return None


def decision_matches_image(decision: Dict, filename: str) -> bool:
    """
    Verify that a decision JSON references the target image.

    Args:
        decision: Decision dictionary (loaded from .decision file)
        filename: Image filename to match

    Returns:
        True if decision references this image, False otherwise
    """
    if not decision or not isinstance(decision, dict):
        return False

    # Check common fields that might contain the filename
    decision_filename = decision.get('filename')
    if decision_filename:
        return decision_filename == filename

    # Check if there's an image_path field
    image_path = decision.get('image_path')
    if image_path:
        import os
        return os.path.basename(image_path) == filename

    # Check images array (for multi-image decisions)
    images = decision.get('images')
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                img_filename = img.get('filename')
                if img_filename == filename:
                    return True
            elif isinstance(img, str):
                import os
                if os.path.basename(img) == filename:
                    return True

    # If no filename field found, assume it matches (permissive)
    return True
