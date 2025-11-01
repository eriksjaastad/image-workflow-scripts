#!/usr/bin/env python3
"""Check which images in the final directory are missing from the database."""

import json
import sqlite3
from pathlib import Path

# Paths
WORKSPACE = Path(__file__).resolve().parents[2]


def get_all_images_from_directory(dir_path: Path) -> set[str]:
    """Get all image filenames from directory recursively."""
    patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
    images = set()
    for pattern in patterns:
        for img_path in dir_path.glob(pattern):
            images.add(img_path.name)
    return images


def get_all_images_from_database(db_path: Path) -> set[str]:
    """Get all image filenames referenced in the database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT images FROM ai_decisions")
    results = cursor.fetchall()
    conn.close()

    images = set()
    for (images_json,) in results:
        try:
            img_list = json.loads(images_json)
            for img in img_list:
                images.add(img)
        except Exception:
            continue

    return images


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Find images in directory not in database"
    )
    parser.add_argument("--image-dir", required=True, help="Directory with images")
    parser.add_argument("--database", required=True, help="Database path")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    db_path = Path(args.database)

    dir_images = get_all_images_from_directory(image_dir)

    db_images = get_all_images_from_database(db_path)

    # Find images in directory but not in database
    missing_from_db = dir_images - db_images

    # Find images in database but not in directory
    missing_from_dir = db_images - dir_images


    if missing_from_db:
        for _i, _img in enumerate(sorted(missing_from_db)[:20], 1):
            pass

        if len(missing_from_db) > 20:
            pass

    if missing_from_dir:
        pass


if __name__ == "__main__":
    main()
