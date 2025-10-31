#!/usr/bin/env python3
"""
Check which images in the final directory are missing from the database.
"""

import json
import sqlite3
from pathlib import Path
from typing import Set

# Paths
WORKSPACE = Path(__file__).resolve().parents[2]


def get_all_images_from_directory(dir_path: Path) -> Set[str]:
    """Get all image filenames from directory recursively."""
    patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
    images = set()
    for pattern in patterns:
        for img_path in dir_path.glob(pattern):
            images.add(img_path.name)
    return images


def get_all_images_from_database(db_path: Path) -> Set[str]:
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

    print(f"Scanning images in: {image_dir}")
    dir_images = get_all_images_from_directory(image_dir)
    print(f"Found {len(dir_images)} images in directory")

    print(f"\nQuerying database: {db_path}")
    db_images = get_all_images_from_database(db_path)
    print(f"Found {len(db_images)} unique images in database")

    # Find images in directory but not in database
    missing_from_db = dir_images - db_images

    # Find images in database but not in directory
    missing_from_dir = db_images - dir_images

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Images in directory:        {len(dir_images):,}")
    print(f"Images in database:         {len(db_images):,}")
    print(f"In directory but NOT in DB: {len(missing_from_db):,} ⚠️")
    print(f"In database but NOT in dir: {len(missing_from_dir):,}")

    if missing_from_db:
        print(f"\n⚠️  {len(missing_from_db):,} IMAGES IN DIRECTORY ARE NOT IN DATABASE!")
        print("\nSample missing images (first 20):")
        for i, img in enumerate(sorted(missing_from_db)[:20], 1):
            print(f"  {i}. {img}")

        if len(missing_from_db) > 20:
            print(f"  ... and {len(missing_from_db) - 20:,} more")

    if missing_from_dir:
        print(f"\n⚠️  {len(missing_from_dir):,} images in database are NOT in directory")
        print("(These may have been deleted or moved)")


if __name__ == "__main__":
    main()
