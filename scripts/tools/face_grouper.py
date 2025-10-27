#!/usr/bin/env python3
"""
Step 2: Hybrid Face Grouper with Similarity Mapping
====================================================
AI-powered person grouper using face recognition and person re-identification.
Creates person directories and similarity maps for intelligent character sorting.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Run face grouping with similarity mapping:
  python scripts/tools/face_grouper.py --images mixed-0919 --out ./sorted \
    --kmeans 7 --emit-map --map-topk 10 --map-threshold 0.22 --map-scope cluster

FEATURES:
---------
• Hybrid face + person-reID embeddings for robust grouping
• InsightFace ArcFace for high-quality face recognition
• TorchReID OSNet fallback for unclear/partial faces
• Agglomerative clustering with automatic threshold adjustment
• Similarity mapping for spatial layout in character sorter
• Configurable clustering parameters (K-means or Agglomerative)
• Comprehensive preview and dry-run capabilities

WORKFLOW POSITION:
------------------
Step 1: Image Version Selection → scripts/01_ai_assisted_reviewer.py
Step 2: Face Grouping → THIS SCRIPT (scripts/02_face_grouper.py)
Step 3: Character Sorting → scripts/03_web_character_sorter.py (uses similarity maps from this step)
Step 4: Final Cropping → scripts/04_batch_crop_tool.py
Step 5: Basic Review → scripts/05_multi_directory_viewer.py

🔍 OPTIONAL ANALYSIS TOOL:
   scripts/utils/similarity_viewer.py - Use between steps 2-3 to analyze face grouper results

How to use it (no renames, no UI changes)

For your current K=7 result, just re-run with map output (it won’t change file placement):

python scripts/02_face_grouper.py --images 00_white --out ./face_groups \
  --kmeans 7 --emit-map --map-topk 10 --map-threshold 0.22 --map-scope cluster


You'll get in face_groups/:

nodes.csv — index, label, filename

edges.csv — weighted edges (cosine similarity + distance)

neighbors.jsonl — for each image: its top neighbors (closest first)

From there you can:

Build a quick “related images” view per file from neighbors.jsonl.

Feed edges.csv into Gephi/Graphistry/NetworkX to compute degrees of separation or communities.

Write a small script to move “nearest 5 to X” anywhere you like using edges.csv.
"""

import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# Deps (install in your 3.11 venv):
# pip install onnxruntime insightface torch torchvision torchreid scikit-learn hdbscan opencv-python-headless

import cv2
import torch
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from insightface.app import FaceAnalysis
from torchreid.utils import FeatureExtractor
import onnxruntime as ort  # provider check for Apple Silicon

# ---- optional FileTracker ----
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.activity_timer import FileTracker
except Exception:
    print("⚠️  FileTracker not found. Continuing without logging.")
    FileTracker = None

# ---- optional companion file utilities ----
try:
    from utils.companion_file_utils import move_file_with_all_companions, safe_move_path
except Exception:
    print("⚠️  companion_file_utils not found. Using basic file moves.")
    move_file_with_all_companions = None
    safe_move_path = None


# ------------------------------ utils ------------------------------
def exif_fix(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img.convert("RGB"))

def device_auto():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def top_center_crop(img: Image.Image, frac_w=0.9, frac_h=0.65) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * frac_w), int(h * frac_h)
    x1 = max(0, (w - cw) // 2)
    y1 = 0
    x2 = min(w, x1 + cw)
    y2 = min(h, y1 + ch)
    return img.crop((x1, y1, x2, y2))


# ------------------------------ embeddings ------------------------------
class HybridEmbedder:
    """
    Face-first embeddings:
      - Try InsightFace (ArcFace). Require det_score >= face_min_score.
      - If not accepted, fall back to OSNet reID embedding.
    All vectors are L2-normalized.
    """
    def __init__(self, max_side: int = 1536, face_min_score: float = 0.65, use_reid_fallback: bool = True):
        self.max_side = max_side
        self.face_min_score = face_min_score
        self.use_reid_fallback = use_reid_fallback
        self.device = device_auto()

        # InsightFace (buffalo_l) providers
        available = ort.get_available_providers()
        prov = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in available]
        if not prov:
            prov = ["CPUExecutionProvider"]
        self.face_app = FaceAnalysis(name="buffalo_l", providers=prov)
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Torchreid OSNet extractor (fallback)
        self.reid_extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            device="mps" if self.device.type == "mps" else "cpu"
        ) if use_reid_fallback else None

    def _prep_image(self, path: Path) -> Image.Image:
        img = exif_fix(Image.open(path))
        w, h = img.size
        scale = max(w, h) / float(self.max_side)
        if scale > 1.0:
            img = img.resize((int(w/scale), int(h/scale)), Image.Resampling.LANCZOS)
        return img

    def _face_embedding(self, img: Image.Image) -> Optional[np.ndarray]:
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        faces = self.face_app.get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda x: getattr(x, "det_score", 0.0))
        if getattr(f, "det_score", 0.0) < self.face_min_score:
            return None
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            return None
        v = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        v = normalize(v, norm="l2")[0]
        return v  # 512-D

    def _reid_embedding(self, img: Image.Image) -> Optional[np.ndarray]:
        if not self.reid_extractor:
            return None
        crop = top_center_crop(img, 0.9, 0.65)
        try:
            vecs = self.reid_extractor([np.array(crop)])
        except Exception:
            vecs = self.reid_extractor([np.array(img)])

        # Accept torch.Tensor, np.ndarray, or list/tuple
        try:
            import torch as _torch
            if isinstance(vecs, _torch.Tensor):
                v = vecs[0].detach().cpu().numpy()
            elif isinstance(vecs, np.ndarray):
                v = vecs[0] if vecs.ndim > 1 else vecs
            elif isinstance(vecs, (list, tuple)) and len(vecs) > 0:
                v = vecs[0]
                if hasattr(v, "detach"):
                    v = v.detach().cpu().numpy()
                else:
                    v = np.asarray(v)
            else:
                return None
        except Exception:
            return None

        v = v.astype(np.float32).reshape(1, -1)
        v = normalize(v, norm="l2")[0]
        return v

    def embed(self, path: Path) -> Optional[np.ndarray]:
        try:
            img = self._prep_image(path)
        except Exception:
            return None

        face = self._face_embedding(img)
        if face is not None:
            return face
        reid = self._reid_embedding(img)
        if reid is not None:
            return reid
        return None


# ------------------------------ clustering helpers ------------------------------
def post_filter_small_clusters(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Convert clusters smaller than min_cluster_size to -1; compact surviving labels to 1..K."""
    if labels.size == 0:
        return labels
    unique, counts = np.unique(labels, return_counts=True)
    small = {int(u) for u, c in zip(unique, counts) if c < min_cluster_size}
    new_labels = labels.copy()
    for i, lab in enumerate(labels):
        new_labels[i] = -1 if int(lab) in small else lab
    # reindex to 1..K skipping -1
    keep = sorted([int(u) for u, c in zip(unique, counts) if c >= min_cluster_size])
    mapping = {lab: i for i, lab in enumerate(keep, start=1)}
    for i, lab in enumerate(new_labels):
        if lab != -1:
            new_labels[i] = mapping[int(lab)]
    return new_labels

def _calculate_distance_band(X: np.ndarray) -> tuple[float, float]:
    """Calculate a reasonable cosine distance band from k-NN distances."""
    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    nn3 = dists[:, 3]
    nn4 = dists[:, 4]
    q10, q25, q50 = np.quantile(np.concatenate([nn3, nn4]), [0.10, 0.25, 0.50])
    return max(0.04, q10), min(0.60, q50)

def cluster_with_backoff(X: np.ndarray, min_cluster_size: int,
                         start_thresh: Optional[float] = None,
                         min_thresh: Optional[float] = None,
                         step: float = 0.01) -> tuple[np.ndarray, float]:
    """
    Agglomerative (cosine, COMPLETE) with automatic threshold backoff.
    Success when >1 named clusters and largest <70% of data.
    """
    low, high = _calculate_distance_band(X)
    thr = high if start_thresh is None else start_thresh
    floor = low if min_thresh is None else min_thresh
    N = X.shape[0]
    best_labels, best_thr = None, thr

    while thr >= floor:
        agg = AgglomerativeClustering(
            n_clusters=None, distance_threshold=thr,
            metric="cosine", linkage="complete"
        )
        raw = agg.fit_predict(X)
        labels = post_filter_small_clusters(raw, min_cluster_size)
        kept = [u for u in np.unique(labels) if u != -1]
        biggest = max((labels == k).sum() for k in kept) if kept else 0
        if len(kept) > 1 and biggest < 0.70 * N:
            return labels, thr
        if kept and biggest < 0.95 * N:
            best_labels, best_thr = labels, thr
        thr = round(thr - step, 4)

    if best_labels is not None:
        return best_labels, best_thr
    return np.full(N, -1, dtype=int), floor

def merge_close_clusters(X: np.ndarray, labels: np.ndarray, dist_thresh: float = 0.18) -> np.ndarray:
    """Merge clusters whose centroids are within cosine distance <= dist_thresh."""
    keep = [int(u) for u in np.unique(labels) if u != -1]
    if not keep:
        return labels
    cents = {}
    for lab in keep:
        idx = np.where(labels == lab)[0]
        c = X[idx].mean(axis=0)
        n = float(np.linalg.norm(c)) or 1.0
        cents[lab] = c / n
    labs = sorted(cents.keys())
    parent = {lab: lab for lab in labs}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = labs[i], labs[j]
            dist = 1.0 - float(np.dot(cents[a], cents[b]))
            if dist <= dist_thresh:
                union(a, b)
    # remap
    root2new, next_id = {}, 1
    new_labels = labels.copy()
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        r = find(int(lab))
        if r not in root2new:
            root2new[r] = next_id
            next_id += 1
        new_labels[i] = root2new[r]
    return new_labels

def assign_unknowns(X: np.ndarray, labels: np.ndarray, assign_threshold: float = 0.18) -> tuple[np.ndarray, int]:
    """Assign -1 items to nearest centroid if cosine distance <= assign_threshold."""
    keep = [int(u) for u in np.unique(labels) if u != -1]
    if not keep:
        return labels, 0
    cents = {}
    for lab in keep:
        idx = np.where(labels == lab)[0]
        c = X[idx].mean(axis=0)
        n = float(np.linalg.norm(c)) or 1.0
        cents[lab] = c / n
    labs = sorted(cents.keys())
    C = np.stack([cents[lab] for lab in labs], axis=0)  # (K, D)

    new_labels = labels.copy()
    assigned = 0
    unk_idx = np.where(labels == -1)[0]
    if len(unk_idx) == 0:
        return new_labels, 0
    # cos distance = 1 - dot(x, c)
    sims = X[unk_idx] @ C.T   # (U, K)
    dists = 1.0 - sims
    mins = dists.min(axis=1)
    argmins = dists.argmin(axis=1)
    for i, di in enumerate(unk_idx):
        if mins[i] <= assign_threshold:
            new_labels[di] = labs[int(argmins[i])]
            assigned += 1
    return new_labels, assigned

def second_pass_unknowns(X: np.ndarray, labels: np.ndarray,
                         min_cluster_size: int,
                         start_thresh: float = 0.30) -> np.ndarray:
    """Re-cluster the unknowns with a looser threshold; append new clusters."""
    unk_idx = np.where(labels == -1)[0]
    if len(unk_idx) < min_cluster_size:
        return labels
    Xu = X[unk_idx]
    labs_u, used = cluster_with_backoff(Xu, min_cluster_size, start_thresh=start_thresh, min_thresh=0.06, step=0.01)
    keep_u = [u for u in np.unique(labs_u) if u != -1]
    if not keep_u:
        return labels
    # remap unknown clusters to new ids after current max
    cur_keep = [u for u in np.unique(labels) if u != -1]
    next_id = (max(cur_keep) if cur_keep else 0) + 1
    remap = {old: (i + next_id) for i, old in enumerate(sorted(keep_u))}
    new_labels = labels.copy()
    for idx_local, lab in enumerate(labs_u):
        gi = unk_idx[idx_local]
        if lab == -1:
            continue
        new_labels[gi] = remap[int(lab)]
    return new_labels


# ------------------------------ IO & grouping ------------------------------
def discover_image_metadata_pairs(root: Path) -> Tuple[List[Path], List[str]]:
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    errors: List[str] = []

    all_files = list(root.rglob("*"))
    images = {p.stem: p for p in all_files if p.suffix.lower() in image_exts}
    metadata  = {p.stem: p for p in all_files if p.suffix.lower() in [".yaml", ".caption"]}

    image_stems = set(images.keys())
    metadata_stems = set(metadata.keys())
    orphaned_images = image_stems - metadata_stems
    orphaned_metadata  = metadata_stems - image_stems
    valid_pairs     = image_stems & metadata_stems

    if orphaned_images:
        errors.append(f"🚨 ALARM: {len(orphaned_images)} images without metadata files:")
        for stem in sorted(orphaned_images):
            errors.append(f"   • {images[stem]}")
    if orphaned_metadata:
        errors.append(f"🚨 ALARM: {len(orphaned_metadata)} metadata files without images:")
        for stem in sorted(orphaned_metadata):
            errors.append(f"   • {metadata[stem]}")

    if not valid_pairs:
        errors.append("🚨 ALARM: No valid image/metadata pairs found!")
        return [], errors

    valid_image_paths = [images[stem] for stem in sorted(valid_pairs)]
    print(f"✅ Found {len(valid_pairs)} valid image/metadata pairs")
    orphan_count = len(orphaned_images) + len(orphaned_metadata)
    if orphan_count:
        print(f"⚠️  Found {orphan_count} orphaned files")
    return valid_image_paths, errors


def move_image_metadata_pairs(out_dir: Path, paths: List[Path], labels: np.ndarray,
                          min_cluster_size: int = 20, tracker=None) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"groups": [], "moved_pairs": 0, "errors": []}
    by_label: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(int(lab), []).append(i)

    total_pairs = 0
    for lab, idxs in sorted(by_label.items(), key=lambda kv: kv[0]):
        group_dir = out_dir / ("unknown" if lab == -1 else f"person_{int(lab):04d}")
        group_dir.mkdir(parents=True, exist_ok=True)
        moved_files, group_errors = [], []

        for i in idxs:
            image_path = paths[i]
            # Try .yaml first, then .caption
            yaml_path = image_path.with_suffix(".yaml")
            if not yaml_path.exists():
                yaml_path = image_path.with_suffix(".caption")
            if not yaml_path.exists():
                msg = f"🚨 MISSING METADATA: {yaml_path} (for image {image_path})"
                print(msg)
                manifest["errors"].append(msg)
                if tracker:
                    tracker.log_operation("error", notes=msg)
                continue
            try:
                # Move image and ALL companion files to destination directory
                if safe_move_path:
                    moved = safe_move_path(image_path, group_dir, dry_run=False)
                    moved_files.extend(moved)
                else:
                    shutil.move(str(image_path), str(group_dir / image_path.name))
                    shutil.move(str(yaml_path),  str(group_dir / yaml_path.name))
                    moved_files.extend([image_path.name, yaml_path.name])
                total_pairs += 1
                if tracker:
                    tracker.log_operation("move", source_dir=str(image_path.parent),
                                          dest_dir=str(group_dir), file_count=2,
                                          notes=f"Moved pair: {image_path.name}, {yaml_path.name}")
            except Exception as e:
                msg = f"🚨 MOVE FAILED: {image_path.name} - {e}"
                print(msg)
                group_errors.append(msg)
                manifest["errors"].append(msg)
                if tracker:
                    tracker.log_operation("error", notes=msg)

        manifest["groups"].append({
            "label": int(lab), "dir": group_dir.name,
            "count": len(idxs), "moved_files": moved_files, "errors": group_errors
        })

    manifest["moved_pairs"] = total_pairs
    manifest["total_groups"] = len([g for g in manifest["groups"] if g["label"] != -1])
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    if tracker:
        tracker.log_operation("group_move",
            source_dir=str(paths[0].parent) if paths else "unknown",
            dest_dir=str(out_dir), file_count=total_pairs * 2,
            notes=f"Moved {total_pairs} image/metadata pairs into {manifest['total_groups']} groups + unknown")
    return manifest

def write_similarity_map(out_dir: Path, kept_paths: List[Path], labels: np.ndarray, X: np.ndarray,
                         topk: int = 8, threshold: float = 0.20, scope: str = "cluster"):
    """
    Writes:
      - nodes.csv: index,label,filename
      - edges.csv: src_idx,dst_idx,src_label,dst_label,sim,dist,src_file,dst_file  (undirected, deduped)
      - neighbors.jsonl: one line per image with its top neighbors (within cluster by default)
    Distances are cosine distance (1 - cosine_similarity) on L2-normalized X.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(kept_paths)
    names = [p.name for p in kept_paths]
    labs  = labels.astype(int)

    # nodes.csv
    with (out_dir / "nodes.csv").open("w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["index","label","filename"])
        for i,(lab,name) in enumerate(zip(labs, names)):
            w.writerow([i, lab, name])

    # choose candidate sets (within cluster vs global)
    clusters: Dict[int, List[int]] = {}
    if scope == "cluster":
        for i, lab in enumerate(labs):
            clusters.setdefault(lab, []).append(i)
    else:
        clusters = {0: list(range(N))}

    # compute neighbors + edges
    edge_set = set()
    edges = []
    with (out_dir / "neighbors.jsonl").open("w") as jf:
        import json
        for key, idxs in clusters.items():
            if len(idxs) <= 1:
                # still write self with empty neighbors
                for i in idxs:
                    jf.write(json.dumps({"index": i, "label": int(labs[i]), "filename": names[i], "neighbors": []}) + "\n")
                continue

            Xi = X[idxs]                            # (M, D)
            sims = Xi @ Xi.T                        # cosine sim on L2-normalized
            np.fill_diagonal(sims, -np.inf)         # exclude self
            dists = 1.0 - sims

            # for each local row, pick neighbors
            for loc, i in enumerate(idxs):
                row_d = dists[loc]                  # distances to candidates
                order = np.argsort(row_d)           # ascending (closest first)
                neigh = []
                for jloc in order:
                    if len(neigh) >= topk:
                        break
                    j = idxs[jloc]
                    dist = float(row_d[jloc])
                    sim  = 1.0 - dist
                    if dist <= threshold or len(neigh) < topk:
                        neigh.append({"index": j, "label": int(labs[j]), "filename": names[j], "sim": sim, "dist": dist})

                jf.write(json.dumps({
                    "index": i, "label": int(labs[i]), "filename": names[i],
                    "neighbors": neigh
                }) + "\n")

                # collect edges (undirected, dedupe with sorted (i,j))
                for n in neigh:
                    a, b = (i, n["index"]) if i < n["index"] else (n["index"], i)
                    key = (a, b)
                    if a != b and key not in edge_set:
                        edge_set.add(key)
                        edges.append((a, b, int(labs[a]), int(labs[b]),
                                      float(1.0 - float(dists[idxs.index(a)][idxs.index(b)]) if scope=="cluster" else 1.0 - float(X[a] @ X[b])),
                                      float(1.0 - (X[a] @ X[b])),  # dist (recompute safely)
                                      names[a], names[b]))

    # edges.csv
    with (out_dir / "edges.csv").open("w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["src_idx","dst_idx","src_label","dst_label","sim","dist","src_file","dst_file"])
        # recompute sim/dist cleanly (above may mix scope paths)
        for a,b,la,lb,_,_,sa,sb in edges:
            sim  = float(X[a] @ X[b])
            dist = float(1.0 - sim)
            w.writerow([a,b,la,lb,sim,dist,sa,sb])



# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Hybrid face-first grouper with YAML-pair moves")
    ap.add_argument("--images", required=True, help="Input dir containing image/metadata pairs (.yaml or .caption)")
    ap.add_argument("--out", default="./face_groups", help="Output directory")
    ap.add_argument("--min-cluster-size", type=int, default=20,
                    help="Minimum items to form a named group; smaller clusters go to 'unknown'")
    ap.add_argument("--max-side", type=int, default=1536, help="Resize longest side before embedding")
    ap.add_argument("--start-threshold", type=float, default=None,
                    help="Initial cosine distance threshold for agglomerative (auto if omitted)")
    ap.add_argument("--face-min-score", type=float, default=0.65,
                    help="Minimum InsightFace det_score to accept as a real face")
    ap.add_argument("--merge-dist", type=float, default=0.18,
                    help="Merge clusters with centroid cosine distance <= this")
    ap.add_argument("--assign-threshold", type=float, default=0.18,
                    help="Assign unknowns to nearest centroid if distance <= this")
    ap.add_argument("--second-pass-start", type=float, default=0.30,
                    help="Start threshold for second-pass clustering on unknowns")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview clusters (write preview.csv) but do not move files")
    ap.add_argument("--kmeans", type=int, default=None,
                help="If set, force exactly K clusters with KMeans (skips agglomerative)")
    ap.add_argument("--emit-map", action="store_true",
                help="Write similarity map files (nodes.csv, edges.csv, neighbors.jsonl). No renames.")
    ap.add_argument("--map-topk", type=int, default=8,
                    help="Neighbors per image to include in the map.")
    ap.add_argument("--map-threshold", type=float, default=0.20,
                    help="Max cosine distance to include an edge (lower=closer).")
    ap.add_argument("--map-scope", choices=["cluster","all"], default="cluster",
                    help="Limit neighbors to same cluster or across all images.")
    args = ap.parse_args()

    set_seed(42)
    img_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = None
    if FileTracker:
        tracker = FileTracker("hybrid_grouper")
        tracker.log_batch_start(f"Hybrid grouping: {img_dir} -> {out_dir}")

    print("🔍 Discovering image/metadata pairs…")
    paths, errors = discover_image_metadata_pairs(img_dir)
    if errors:
        print("\n".join(errors))
        if not paths:
            print("❌ No valid image/metadata pairs found. Exiting.")
            sys.exit(1)
        print(f"\n⚠️  Continuing with {len(paths)} valid pairs, ignoring orphaned files.")

    print("\n🤖 Initializing embedders…")
    embedder = HybridEmbedder(max_side=args.max_side, face_min_score=args.face_min_score, use_reid_fallback=True)
    print(f"Device (Torch): {'mps' if torch.backends.mps.is_available() else 'cpu'}")

    embs: List[np.ndarray] = []
    kept_paths: List[Path] = []
    failed = used_face = used_reid = used_none = 0

    print(f"\n🎯 Embedding {len(paths)} images…")
    for p in tqdm(paths):
        vec = embedder.embed(p)
        if vec is None:
            failed += 1
            used_none += 1
            if FileTracker:
                tracker and tracker.log_operation("error", notes=f"Failed to embed: {p}")
            continue
        # quick attribution
        img = embedder._prep_image(p)
        if embedder._face_embedding(img) is not None:
            used_face += 1
        else:
            used_reid += 1
        embs.append(vec)
        kept_paths.append(p)

    print(f"\nℹ️  Embed summary: face={used_face}, reid={used_reid}, none={used_none}")
    if not embs:
        print("❌ No embeddings produced.")
        sys.exit(1)

    X = np.vstack(embs).astype(np.float32)  # L2-normalized

    if args.kmeans:
        print(f"\n🔗 Clustering {X.shape[0]} embeddings with KMeans (k={args.kmeans})…")
        km = KMeans(n_clusters=args.kmeans, n_init=10, random_state=42)
        raw = km.fit_predict(X)  # 0..K-1
        labels = (raw + 1).astype(int)  # make them 1..K (no -1s)
        kept = np.unique(labels)
        print(f"   • KMeans: produced {len(kept)} groups")
        if args.emit_map:
            write_similarity_map(out_dir, kept_paths, labels, X,
                                topk=args.map_topk, threshold=args.map_threshold, scope=args.map_scope)

        # write preview.csv and (if not --dry-run) move files just like before
        from csv import writer
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "preview.csv", "w", newline="") as f:
            w = writer(f)
            w.writerow(["label","filename"])
            for p, lab in zip(kept_paths, labels):
                w.writerow([int(lab), p.name])
        print(f"📝 wrote preview: {out_dir/'preview.csv'}")

        if not args.dry_run:
            print("\n📁 Moving image/metadata pairs to groups…")
            manifest = move_image_metadata_pairs(out_dir, kept_paths, labels,
                                            min_cluster_size=1, tracker=tracker)  # no filtering
            print("\n✅ COMPLETE!")
            print(f"   • Embedded: {len(embs)} / {len(paths)}")
            print(f"   • Named groups: {manifest['total_groups']}")
            print(f"   • Moved pairs: {manifest['moved_pairs']}")
            print(f"   • Output: {out_dir}")
            if FileTracker and tracker:
                tracker.log_batch_end()
        else:
            print("\n🔎 dry-run: not moving files")
        return  # skip the agglomerative path below

    print(f"\n🔗 Clustering {X.shape[0]} embeddings (Agglomerative, cosine)…")
    labels, used_thr = cluster_with_backoff(
        X, min_cluster_size=args.min_cluster_size,
        start_thresh=args.start_threshold, min_thresh=0.04, step=0.01
    )
    print(f"   • used distance_threshold: {used_thr:.2f}")
    # Merge near-duplicates
    labels = merge_close_clusters(X, labels, dist_thresh=args.merge_dist)
    # Second pass: pull more identities out of unknowns
    labels = second_pass_unknowns(X, labels, min_cluster_size=args.min_cluster_size,
                                  start_thresh=args.second_pass_start)
    # Assign remaining unknowns near existing centroids
    labels, assigned = assign_unknowns(X, labels, assign_threshold=args.assign_threshold)

    kept = [u for u in np.unique(labels) if u != -1]
    print(f"   • clusters (named): {len(kept)}  • assigned from unknown: {assigned}  • unknown left: {(labels==-1).sum()}")

    if args.emit_map:
        write_similarity_map(out_dir, kept_paths, labels, X,
                            topk=args.map_topk, threshold=args.map_threshold, scope=args.map_scope)


    # Preview CSV
    from csv import writer
    with open(out_dir / "preview.csv", "w", newline="") as f:
        w = writer(f)
        w.writerow(["label","filename"])
        for p, lab in zip(kept_paths, labels):
            w.writerow([int(lab), p.name])
    print(f"📝 wrote preview: {out_dir/'preview.csv'}")

    if not args.dry_run:
        print("\n📁 Moving image/metadata pairs to groups…")
        manifest = move_image_metadata_pairs(out_dir, kept_paths, labels,
                                         min_cluster_size=args.min_cluster_size, tracker=tracker)
        
        # Count images in each face group directory
        print("\n✅ COMPLETE!")
        print(f"   • Embedded: {len(embs)} / {len(paths)} (failed: {failed})")
        print(f"   • Named groups: {manifest['total_groups']}  (+ unknown)")
        print(f"   • Moved pairs: {manifest['moved_pairs']}")
        print(f"   • Output: {out_dir}")
        
        # Display image counts per face group
        print("\n📊 Face Group Image Counts:")
        total_images = 0
        group_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir()])
        for group_dir in group_dirs:
            image_count = len([f for f in group_dir.iterdir() 
                             if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}])
            total_images += image_count
            print(f"   • {group_dir.name}: {image_count} images")
        print(f"   • Total: {total_images} images across {len(group_dirs)} groups")
    else:
        print("\n🔎 dry-run: not moving files")

    if FileTracker and tracker:
        tracker.log_batch_end()


if __name__ == "__main__":
    main()
