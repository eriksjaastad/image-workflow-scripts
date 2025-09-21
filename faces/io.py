"""I/O utilities for the face pipeline."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class PipelinePaths:
    """Organised collection of output paths."""

    root: Path
    people_dir: Path
    unknown_dir: Path
    embeddings_path: Path
    metadata_path: Path
    detections_json: Optional[Path] = None


@dataclass
class PipelineState:
    """Persistent progress for resume functionality."""

    processed_images: List[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"processed": self.processed_images}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            LOGGER.warning("Invalid resume file: %s", path)
            return cls()
        return cls(processed_images=list(data.get("processed", [])))


def discover_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            images.append(path)
    return images


def load_image(path: Path, max_size: int) -> np.ndarray:
    """Load an image with EXIF orientation handling and optional resizing."""

    with Image.open(path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        if max(pil_img.size) > max_size:
            pil_img.thumbnail((max_size, max_size), Image.LANCZOS)
        array = np.array(pil_img, dtype="uint8")
    return array


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(base: str) -> str:
    keep = [c if c.isalnum() or c in {"-", "_"} else "_" for c in base]
    return "".join(keep)


def save_face_crop(image: np.ndarray, directory: Path, filename: str) -> Path:
    ensure_dir(directory)
    path = directory / filename
    Image.fromarray(image.astype("uint8")).save(path)
    return path


def save_embeddings(embeddings: np.ndarray, metadata: List[Dict], paths: PipelinePaths) -> None:
    ensure_dir(paths.root)
    np.save(paths.embeddings_path, embeddings)
    paths.metadata_path.write_text(json.dumps(metadata, indent=2))


def save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))


def copy_or_link(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def build_paths(out_dir: Path, detections_json: Optional[Path] = None) -> PipelinePaths:
    people = out_dir / "people"
    unknown = out_dir / "unknown"
    embeddings_path = out_dir / "embeddings.npy"
    metadata_path = out_dir / "metadata.json"
    return PipelinePaths(
        root=out_dir,
        people_dir=people,
        unknown_dir=unknown,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        detections_json=detections_json,
    )


def crop_from_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], crop_size: Optional[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop_size is None or crop.size == 0:
        return crop.copy()
    pil_image = Image.fromarray(crop.astype("uint8"))
    pil_image = pil_image.resize((crop_size, crop_size), Image.BILINEAR)
    return np.array(pil_image, dtype="uint8")

