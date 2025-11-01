#!/usr/bin/env python3
"""
Centralized standard paths for project directories (double-underscore variants).

Provides helpers to reference commonly used directories in a single place.
"""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root (parent of scripts/)."""
    return Path(__file__).parent.parent.parent


def get_selected_dir() -> Path:
    return get_project_root() / "__selected"


def get_crop_dir() -> Path:
    return get_project_root() / "__crop"


def get_crop_auto_dir() -> Path:
    return get_project_root() / "__crop_auto"


def get_delete_staging_dir() -> Path:
    return get_project_root() / "__delete_staging"


def get_cropped_dir() -> Path:
    return get_project_root() / "__cropped"


def get_character_group_dir(index: int) -> Path:
    if index not in (1, 2, 3):
        raise ValueError("Character group index must be 1, 2, or 3")
    return get_project_root() / f"__character_group_{index}"


def get_character_group_dirs() -> list[Path]:
    return [get_character_group_dir(i) for i in (1, 2, 3)]


def ensure_standard_dirs_exist() -> dict[str, Path]:
    """Create standard directories if they do not exist; return mapping for convenience."""
    dirs = {
        "selected": get_selected_dir(),
        "crop": get_crop_dir(),
        "crop_auto": get_crop_auto_dir(),
        "delete_staging": get_delete_staging_dir(),
        "cropped": get_cropped_dir(),
        "character_group_1": get_character_group_dir(1),
        "character_group_2": get_character_group_dir(2),
        "character_group_3": get_character_group_dir(3),
    }
    for p in dirs.values():
        try:
            p.mkdir(exist_ok=True)
        except Exception:
            # Directory creation is best-effort
            pass
    return dirs
