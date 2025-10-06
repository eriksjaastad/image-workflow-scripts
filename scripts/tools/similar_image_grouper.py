"""
Groups similar images by visual similarity using CLIP embeddings.
Moves original files (with ALL companion files having same base name) to group directories.
NEVER alters images or filenames - only moves files.

Companion files include: .yaml, .content, .caption, .txt, .json, etc.
Any file with the same base name as an image will be moved together.

Creates organized structure:
  target_parent_directory/
    ├── group_0001/
    ├── group_0002/
    ├── ...
    └── singles/

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
  python scripts/tools/similar_image_grouper.py mojo1 mojo1_clustered
  python scripts/tools/similar_image_grouper.py /path/to/photos /path/to/output
"""

import os, shutil, sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import hdbscan

# Import standardized companion file utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.companion_file_utils import find_all_companion_files, move_file_with_all_companions

# --- config ---
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./photos_in")
TARGET_PARENT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./groups_out")

MODEL_NAME = "clip-ViT-B-32"         # local after first download
MIN_CLUSTER_SIZE = 6                  # tweak smaller if you want more/smaller groups
MIN_SAMPLES = 1                       # 1 = more groups, higher = stricter
EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
# Companion files: any file with same base name as image (wildcard approach)
# This includes .yaml, .content, .caption, .txt, .json, etc.

def load_images(folder: Path):
    files = [p for p in folder.rglob("*") if p.suffix.lower() in EXTS]
    return files

def safe_open(path: Path):
    """Load image for analysis without altering the original file."""
    try:
        im = Image.open(path).convert("RGB")
        # Create a copy for analysis - NEVER alter the original
        # Resize only the copy for embedding generation
        analysis_img = im.copy()
        analysis_img.thumbnail((384, 384))
        return analysis_img
    except (UnidentifiedImageError, OSError):
        return None

def move_file_with_companions(src_path: Path, dst_dir: Path):
    """Move image file and all its companion files, preserving original names."""
    return move_file_with_all_companions(src_path, dst_dir, dry_run=False)

def main():
    files = load_images(SRC)
    if not files:
        print(f"No images found in {SRC}")
        return

    print(f"Found {len(files)} images. Embedding with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # prepare pixel batches
    imgs, good_idx = [], []
    for i, p in enumerate(tqdm(files, desc="Loading")):
        im = safe_open(p)
        if im is not None:
            imgs.append(im)
            good_idx.append(i)

    # model.encode can take raw PIL Images
    embs = model.encode(imgs, convert_to_numpy=True, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, metric='euclidean')
    labels = clusterer.fit_predict(embs)

    TARGET_PARENT.mkdir(parents=True, exist_ok=True)
    groups = {}
    for idx, lbl in zip(good_idx, labels):
        groups.setdefault(int(lbl), []).append(files[idx])

    # Move files to group directories - NEVER alter filenames or images
    print(f"\nMoving files to group directories...")
    
    for lbl, members in groups.items():
        name = "singles" if lbl == -1 else f"group_{lbl:04d}"
        gdir = TARGET_PARENT / name
        print(f"\n📁 {name} ({len(members)} files):")
        
        for p in members:
            try:
                move_file_with_companions(p, gdir)
            except OSError as e:
                print(f"Error moving {p.name}: {e}")

    # Report results
    sizes = {k: len(v) for k, v in groups.items()}
    clustered = sum(c for k,c in sizes.items() if k != -1)
    
    print(f"\n✅ Done! {len(sizes)-('singles' in sizes)} clusters, {clustered} images clustered, {sizes.get(-1,0)} singles.")
    print(f"📁 Output: {TARGET_PARENT.resolve()}")
    print("🔒 All original filenames and images preserved - no alterations made.")

if __name__ == "__main__":
    main()
