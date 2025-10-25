from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

from PIL import Image


def build_originals_index(originals_root: Path, cache_path: Path) -> Dict[str, str]:
    """Scan originals_root for *.png and build a filename->absolute path index.

    Writes cache_path as JSON for reuse. Returns the mapping.
    """
    originals_root = originals_root.resolve()
    index: Dict[str, str] = {}
    for p in originals_root.rglob("*.png"):
        try:
            index[p.name] = str(p)
        except Exception:
            continue
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(index))
    return index


def load_originals_index(cache_path: Path) -> Optional[Dict[str, str]]:
    try:
        data = json.loads(cache_path.read_text())
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items()}
    except Exception:
        return None
    return None


def iter_ranker_samples(db_paths: Iterable[Path]) -> Iterable[Tuple[str, int]]:
    """Yield (images_json, user_selected_index) for all decisions suitable for ranker."""
    for db in db_paths:
        try:
            conn = sqlite3.connect(str(db))
            cur = conn.cursor()
            cur.execute(
                "SELECT images, user_selected_index FROM ai_decisions WHERE user_selected_index IS NOT NULL"
            )
            while True:
                rows = cur.fetchmany(2000)
                if not rows:
                    break
                for imj, usi in rows:
                    yield imj, int(usi)
            conn.close()
        except Exception:
            continue


def iter_crop_samples(db_paths: Iterable[Path]) -> Iterable[Tuple[str, int, str, int, int]]:
    """Yield (images_json, user_selected_index, final_crop_coords_json, image_width, image_height)."""
    for db in db_paths:
        try:
            conn = sqlite3.connect(str(db))
            cur = conn.cursor()
            cur.execute(
                "SELECT images, user_selected_index, final_crop_coords, image_width, image_height "
                "FROM ai_decisions WHERE final_crop_coords IS NOT NULL"
            )
            while True:
                rows = cur.fetchmany(2000)
                if not rows:
                    break
                for imj, usi, fin, iw, ih in rows:
                    yield imj, int(usi), fin, int(iw), int(ih)
            conn.close()
        except Exception:
            continue


def resolve_image_path(filename: str, index: Dict[str, str]) -> Optional[Path]:
    p = index.get(filename)
    return Path(p) if p else None


def open_image_as_rgb(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


