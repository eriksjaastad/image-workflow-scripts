"""Command-line interface for the face grouping pipeline."""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback when PyYAML missing
    yaml = None
from rich.console import Console
from rich.progress import Progress
from PIL import Image

from faces.cluster import (
    ClusterConfig,
    ClusterResult,
    FaceSample,
    cluster_faces,
    load_existing_clusters,
    save_cluster_manifest,
    save_unknown_manifest,
)
from faces.detector import DetectorConfig, build_detector
from faces.embeddings import BaseEmbedder, EmbeddingConfig, build_embedder
from faces.io import (
    PipelineState,
    build_paths,
    copy_or_link,
    crop_from_bbox,
    discover_images,
    ensure_dir,
    load_image,
    safe_filename,
    save_embeddings,
    save_face_crop,
    save_json,
)
from faces.viz import draw_detections

LOGGER = logging.getLogger("face_pipeline")
console = Console()


@dataclass
class PipelineConfig:
    images: Path
    out_dir: Path
    backend: str = "insightface"
    embedder_backend: str = "insightface"
    cluster_method: str = "agglomerative"
    distance_threshold: float = 0.4
    dbscan_eps: float = 0.38
    dbscan_min_samples: int = 3
    min_cluster_size: int = 3
    min_score: float = 0.85
    batch_size: int = 16
    max_size: int = 2048
    copy_originals: bool = False
    detect_only: bool = False
    save_annots: bool = False
    bench: bool = False
    json_path: Optional[Path] = None
    resume: bool = False
    incremental: bool = False
    people_dir: Optional[Path] = None
    assign_threshold: float = 0.6
    crop_size: Optional[int] = None
    near_dup_threshold: float = 0.98
    allow_coreml: bool = True
    verbose: bool = False


@dataclass
class PipelineResult:
    images_processed: int
    faces_detected: int
    clusters_created: int
    unknown_faces: int
    elapsed: float


def load_yaml_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    if yaml is None:
        LOGGER.warning("PyYAML not installed; skipping config file %s", path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect, embed, and group faces.")
    parser.add_argument("--images", type=Path, required=True, help="Input image directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Configuration YAML path")
    parser.add_argument("--backend", choices=["insightface", "mediapipe", "simple"], help="Detector backend")
    parser.add_argument("--embedder", choices=["insightface", "simple"], help="Embedding backend")
    parser.add_argument("--cluster", choices=["agglomerative", "dbscan"], help="Clustering algorithm")
    parser.add_argument("--distance-threshold", type=float, help="Agglomerative distance threshold")
    parser.add_argument("--dbscan-eps", type=float, help="DBSCAN eps")
    parser.add_argument("--min-score", type=float, help="Detector confidence minimum")
    parser.add_argument("--min-cluster-size", type=int, help="Minimum faces per identity")
    parser.add_argument("--assign-threshold", type=float, help="Incremental assignment cosine threshold")
    parser.add_argument("--batch-size", type=int, help="Embedding batch size")
    parser.add_argument("--max-size", type=int, help="Resize longest edge before detection")
    parser.add_argument("--copy-originals", action="store_true", help="Copy original images into clusters")
    parser.add_argument("--detect-only", action="store_true", help="Only run detection and emit JSON")
    parser.add_argument("--json", type=Path, help="Detection JSON output path")
    parser.add_argument("--save-annots", action="store_true", help="Save annotated detections")
    parser.add_argument("--bench", action="store_true", help="Print timing information")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run state")
    parser.add_argument("--incremental", action="store_true", help="Assign faces into existing people directory")
    parser.add_argument("--people-dir", type=Path, help="Existing people directory for incremental mode")
    parser.add_argument("--crop-size", type=int, help="Resize saved crops to square size")
    parser.add_argument("--near-dup-threshold", type=float, help="Prune near duplicates above cosine threshold")
    parser.add_argument("--allow-coreml", action="store_true", help="Allow CoreML/Metal execution providers")
    parser.add_argument("--no-allow-coreml", action="store_true", help="Disable CoreML/Metal providers")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    yaml_config = load_yaml_config(args.config)
    cfg = PipelineConfig(
        images=args.images,
        out_dir=args.out,
        backend=yaml_config.get("detector_backend", "insightface"),
        embedder_backend=yaml_config.get("embedder_backend", "insightface"),
        cluster_method=yaml_config.get("cluster_method", "agglomerative"),
        distance_threshold=float(yaml_config.get("distance_threshold", 0.4)),
        dbscan_eps=float(yaml_config.get("dbscan_eps", 0.38)),
        dbscan_min_samples=int(yaml_config.get("dbscan_min_samples", 3)),
        min_cluster_size=int(yaml_config.get("min_cluster_size", 3)),
        min_score=float(yaml_config.get("min_score", 0.85)),
        batch_size=int(yaml_config.get("batch_size", 16)),
        max_size=int(yaml_config.get("max_size", 2048)),
        copy_originals=bool(yaml_config.get("copy_originals", False)),
        detect_only=bool(yaml_config.get("detect_only", False)),
        save_annots=bool(yaml_config.get("save_annots", False)),
        bench=bool(yaml_config.get("bench", False)),
        json_path=Path(yaml_config["detections_json"]) if yaml_config.get("detections_json") else None,
        resume=bool(yaml_config.get("resume", False)),
        incremental=bool(yaml_config.get("incremental", False)),
        people_dir=Path(yaml_config["people_dir"]) if yaml_config.get("people_dir") else None,
        assign_threshold=float(yaml_config.get("assign_threshold", 0.6)),
        crop_size=yaml_config.get("crop_size"),
        near_dup_threshold=float(yaml_config.get("near_dup_threshold", 0.98)),
        allow_coreml=bool(yaml_config.get("allow_coreml", True)),
        verbose=bool(yaml_config.get("verbose", False)),
    )

    if args.backend:
        cfg.backend = args.backend
    if args.embedder:
        cfg.embedder_backend = args.embedder
    if args.cluster:
        cfg.cluster_method = args.cluster
    if args.distance_threshold is not None:
        cfg.distance_threshold = args.distance_threshold
    if args.dbscan_eps is not None:
        cfg.dbscan_eps = args.dbscan_eps
    if args.min_score is not None:
        cfg.min_score = args.min_score
    if args.min_cluster_size is not None:
        cfg.min_cluster_size = args.min_cluster_size
    if args.assign_threshold is not None:
        cfg.assign_threshold = args.assign_threshold
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_size is not None:
        cfg.max_size = args.max_size
    if args.copy_originals:
        cfg.copy_originals = True
    if args.detect_only:
        cfg.detect_only = True
    if args.json:
        cfg.json_path = args.json
    if args.save_annots:
        cfg.save_annots = True
    if args.bench:
        cfg.bench = True
    if args.resume:
        cfg.resume = True
    if args.incremental:
        cfg.incremental = True
    if args.people_dir:
        cfg.people_dir = args.people_dir
    if args.crop_size is not None:
        cfg.crop_size = args.crop_size
    if args.near_dup_threshold is not None:
        cfg.near_dup_threshold = args.near_dup_threshold
    if args.allow_coreml:
        cfg.allow_coreml = True
    if args.no_allow_coreml:
        cfg.allow_coreml = False
    if args.verbose:
        cfg.verbose = True
    return cfg


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    start_time = time.time()
    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ensure_dir(config.out_dir)
    detections_json = config.json_path
    paths = build_paths(config.out_dir, detections_json)
    state_path = config.out_dir / "resume.json"
    state = PipelineState.load(state_path) if config.resume else PipelineState()
    processed = set(state.processed_images)

    images = discover_images(config.images)
    if not images:
        LOGGER.warning("No images found in %s", config.images)

    detector = build_detector(
        DetectorConfig(
            min_score=config.min_score,
            max_size=config.max_size,
            backend=config.backend,
            simple_padding=4,
            allow_coreml=config.allow_coreml,
        )
    )
    embedder: Optional[BaseEmbedder] = None
    if not config.detect_only:
        embedder = build_embedder(
            EmbeddingConfig(
                batch_size=config.batch_size,
                backend=config.embedder_backend,
                allow_coreml=config.allow_coreml,
            )
        )

    detection_records: List[Dict] = []
    samples: List[FaceSample] = []
    crops: List[np.ndarray] = []
    metadata: List[Dict] = []
    total_faces = 0
    successfully_processed_images: List[str] = []  # Track images that complete processing

    with Progress(console=console) as progress:
        task_id = progress.add_task("Processing", total=len(images))
        for image_path in images:
            rel_key = str(image_path.relative_to(config.images))
            progress.update(task_id, advance=1, description=f"{rel_key}")
            if config.resume and rel_key in processed:
                continue
            try:
                image = load_image(image_path, config.max_size)
            except Exception as exc:
                LOGGER.warning("Failed to load %s: %s", image_path, exc)
                continue
            detections = detector.detect(image, image_path)
            detection_records.append(
                {
                    "image_path": str(image_path),
                    "detections": [
                        {
                            "bbox": list(det.bbox),
                            "score": det.score,
                            "landmarks": {k: list(v) for k, v in det.landmarks.items()},
                        }
                        for det in detections
                    ],
                }
            )
            total_faces += len(detections)

            if config.save_annots and detections:
                annot = draw_detections(image, detections)
                annot_path = config.out_dir / "annots" / rel_key
                ensure_dir(annot_path.parent)
                out_path = annot_path.with_suffix(".jpg")
                Image.fromarray(annot.astype("uint8")).save(out_path)

            if not config.detect_only and detections:
                per_image = [(det, crop_from_bbox(image, det.bbox, config.crop_size)) for det in detections]
                for start in range(0, len(per_image), config.batch_size):
                    batch = per_image[start : start + config.batch_size]
                    face_arrays = [item[1] for item in batch]
                    if not face_arrays:
                        continue
                    results = embedder.embed_batch(face_arrays) if embedder else []
                    if len(results) != len(face_arrays):
                        LOGGER.warning("Embedder returned %s results for %s inputs", len(results), len(face_arrays))
                    for (det, crop), embedding in zip(batch, results):
                        crops.append(crop)
                        sample = FaceSample(
                            image_path=image_path,
                            detection_index=det.index,
                            bbox=det.bbox,
                            score=det.score,
                            embedding=embedding.vector,
                            crop_index=len(crops) - 1,
                        )
                        samples.append(sample)
                        metadata.append(
                            {
                                "image_path": str(image_path),
                                "bbox": list(det.bbox),
                                "score": det.score,
                                "landmarks": {k: list(v) for k, v in det.landmarks.items()},
                            }
                        )
            # Track this image as successfully processed (will mark as complete after persistence)
            successfully_processed_images.append(rel_key)

    if config.detect_only:
        if paths.detections_json:
            ensure_dir(paths.detections_json.parent)
            save_json(detection_records, paths.detections_json)
        
        # Mark images as processed after detection data is saved
        if config.resume:
            state.processed_images.extend(successfully_processed_images)
            state.save(state_path)
            
        elapsed = time.time() - start_time
        return PipelineResult(
            images_processed=len(images),
            faces_detected=total_faces,
            clusters_created=0,
            unknown_faces=0,
            elapsed=elapsed,
        )

    if not samples:
        elapsed = time.time() - start_time
        return PipelineResult(
            images_processed=len(images),
            faces_detected=total_faces,
            clusters_created=0,
            unknown_faces=0,
            elapsed=elapsed,
        )

    embeddings_matrix = np.stack([sample.embedding for sample in samples]).astype("float32")
    save_embeddings(embeddings_matrix, metadata, paths)

    existing = load_existing_clusters(config.people_dir or paths.people_dir) if config.incremental else None
    cluster_config = ClusterConfig(
        method=config.cluster_method,
        distance_threshold=config.distance_threshold,
        dbscan_eps=config.dbscan_eps,
        dbscan_min_samples=config.dbscan_min_samples,
        min_cluster_size=config.min_cluster_size,
        assign_threshold=config.assign_threshold,
        near_dup_threshold=config.near_dup_threshold,
    )
    cluster_result = cluster_faces(samples, cluster_config, existing)
    _write_clusters(cluster_result, crops, paths, config)
    
    # Now that all data is persisted, mark images as processed for resume functionality
    if config.resume:
        state.processed_images.extend(successfully_processed_images)
        state.save(state_path)

    elapsed = time.time() - start_time
    if config.bench:
        console.print(f"Processed {len(images)} images in {elapsed:.2f}s ({total_faces} faces)")

    return PipelineResult(
        images_processed=len(images),
        faces_detected=total_faces,
        clusters_created=len(cluster_result.clusters),
        unknown_faces=len(cluster_result.unknown),
        elapsed=elapsed,
    )


def _write_clusters(result: ClusterResult, crops: List[np.ndarray], paths: PipelinePaths, config: PipelineConfig) -> None:
    ensure_dir(paths.people_dir)
    ensure_dir(paths.unknown_dir)
    for cluster in result.clusters:
        cluster_dir = paths.people_dir / cluster.label
        ensure_dir(cluster_dir)
        faces_dir = cluster_dir / "faces"
        ensure_dir(faces_dir)
        originals_dir = cluster_dir / "originals"
        seen_originals = set()
        rep_image: Optional[np.ndarray] = None
        rep_path = cluster_dir / "rep.jpg"
        for member_idx, member in enumerate(cluster.members):
            if member.crop_index is None:
                continue
            crop = crops[member.crop_index]
            basename = safe_filename(f"{member.image_path.stem}_{member_idx:04d}")
            filename = f"{basename}.jpg"
            crop_path = save_face_crop(crop, faces_dir, filename)
            member.crop_path = crop_path
            if config.copy_originals and member.image_path not in seen_originals:
                copy_or_link(member.image_path, originals_dir / member.image_path.name)
                seen_originals.add(member.image_path)
            if member_idx == cluster.representative_index:
                rep_image = crop
        if rep_image is None and cluster.members:
            first = cluster.members[0]
            if first.crop_index is not None:
                rep_image = crops[first.crop_index]
        if rep_image is not None:
            Image.fromarray(rep_image.astype("uint8")).save(rep_path)
        save_cluster_manifest(cluster, cluster_dir)

    if result.unknown:
        unknown_faces_dir = paths.unknown_dir / "faces"
        ensure_dir(unknown_faces_dir)
        for idx, member in enumerate(result.unknown):
            if member.crop_index is None:
                continue
            crop = crops[member.crop_index]
            basename = safe_filename(f"unknown_{idx:04d}")
            filename = f"{basename}.jpg"
            member.crop_path = save_face_crop(crop, unknown_faces_dir, filename)
        save_unknown_manifest(result.unknown, paths.unknown_dir)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    result = run_pipeline(config)
    console.print(
        f"Processed {result.images_processed} images, detected {result.faces_detected} faces, "
        f"formed {result.clusters_created} clusters, {result.unknown_faces} unknown faces."
    )


if __name__ == "__main__":
    main()

