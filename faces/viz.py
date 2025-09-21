"""Visualization helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import FaceDetection

try:  # pragma: no cover - optional font support
    DEFAULT_FONT = ImageFont.load_default()
except Exception:  # pragma: no cover - minimal fallback
    DEFAULT_FONT = None


def draw_detections(image: np.ndarray, detections: Iterable[FaceDetection]) -> np.ndarray:
    canvas = Image.fromarray(image.astype("uint8")).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        for landmark in detection.landmarks.values():
            pt = (int(landmark[0]), int(landmark[1]))
            draw.ellipse([(pt[0] - 2, pt[1] - 2), (pt[0] + 2, pt[1] + 2)], fill=(255, 0, 0))
        label = f"{detection.score:.2f}"
        if DEFAULT_FONT:
            draw.text((x1, max(y1 - 12, 0)), label, fill=(255, 255, 0), font=DEFAULT_FONT)
        else:
            draw.text((x1, max(y1 - 12, 0)), label, fill=(255, 255, 0))
    return np.array(canvas)


def montage(images: List[np.ndarray], grid: Tuple[int, int] | None = None, tile_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    if not images:
        return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    if grid is None:
        rows = cols = int(np.ceil(np.sqrt(len(images))))
    else:
        rows, cols = grid
    tiles: List[Image.Image] = []
    for img in images:
        pil = Image.fromarray(img.astype("uint8"))
        tiles.append(pil.resize(tile_size, Image.BILINEAR))
    while len(tiles) < rows * cols:
        tiles.append(Image.new("RGB", tile_size, color=(0, 0, 0)))
    rows_arr = []
    for r in range(rows):
        row_tiles = tiles[r * cols : (r + 1) * cols]
        row = np.hstack([np.array(tile) for tile in row_tiles])
        rows_arr.append(row)
    return np.vstack(rows_arr)


def save_montage(images: List[np.ndarray], path: Path, grid: Tuple[int, int] | None = None, tile_size: Tuple[int, int] = (160, 160)) -> None:
    img = montage(images, grid, tile_size)
    Image.fromarray(img).save(path)

