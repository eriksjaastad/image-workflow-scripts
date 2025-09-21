"""Clustering utilities for face identities."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN

LOGGER = logging.getLogger(__name__)


@dataclass
class FaceSample:
    """Face sample carrying embedding and metadata."""

    image_path: Path
    detection_index: int
    bbox: tuple[int, int, int, int]
    score: float
    embedding: np.ndarray
    crop_index: Optional[int] = None
    crop_path: Optional[Path] = None


@dataclass
class ClusterEntry:
    """Clustered identity."""

    label: str
    centroid: np.ndarray
    members: List[FaceSample] = field(default_factory=list)
    representative_index: int = 0
    similarity_matrix: Optional[np.ndarray] = None


@dataclass
class ClusterConfig:
    """Parameters for clustering."""

    method: str = "agglomerative"
    distance_threshold: float = 0.4
    dbscan_eps: float = 0.38
    dbscan_min_samples: int = 3
    min_cluster_size: int = 3
    assign_threshold: float = 0.6
    near_dup_threshold: float = 0.98


@dataclass
class ClusterResult:
    """Outcome of clustering."""

    clusters: List[ClusterEntry]
    unknown: List[FaceSample]
    assignments: Dict[str, List[FaceSample]]


@dataclass
class ClusterSummary:
    """Summary persisted for incremental runs."""

    label: str
    centroid: np.ndarray


def cluster_faces(
    samples: List[FaceSample],
    config: ClusterConfig,
    existing: Optional[List[ClusterSummary]] = None,
) -> ClusterResult:
    """Cluster embeddings into identities."""

    if not samples:
        return ClusterResult(clusters=[], unknown=[], assignments={})

    embeddings = np.stack([sample.embedding for sample in samples]).astype("float32")
    labels = _initialise_labels(samples, embeddings, existing, config)

    remaining_indices = [i for i, lbl in enumerate(labels) if lbl == -1]
    if remaining_indices:
        remaining_embeddings = embeddings[remaining_indices]
        if config.method == "dbscan":
            clustering = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples, metric="cosine")
            new_labels = clustering.fit_predict(remaining_embeddings)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=config.distance_threshold,
            )
            new_labels = clustering.fit_predict(remaining_embeddings)
        next_label = _next_label(labels, existing)
        for idx, sample_index in enumerate(remaining_indices):
            if new_labels[idx] == -1:
                continue
            labels[sample_index] = next_label + new_labels[idx]

    cluster_map: Dict[int, List[FaceSample]] = {}
    for label, sample in zip(labels, samples):
        cluster_map.setdefault(label, []).append(sample)

    clusters: List[ClusterEntry] = []
    unknown: List[FaceSample] = []
    assignments: Dict[str, List[FaceSample]] = {}

    existing_label_map = {idx: summary.label for idx, summary in enumerate(existing or [])}
    next_person_index = _starting_person_index(existing or [])

    sorted_labels = sorted(cluster_map.keys())
    for numeric_label in sorted_labels:
        members = cluster_map[numeric_label]
        if numeric_label == -1 or len(members) < config.min_cluster_size:
            unknown.extend(members)
            continue
        centroid = _compute_centroid(members)
        if numeric_label in existing_label_map:
            label_name = existing_label_map[numeric_label]
        else:
            label_name = f"person_{next_person_index:04d}"
            next_person_index += 1
        pruned_members = _prune_duplicates(members, config.near_dup_threshold)
        sim_matrix = _similarity_matrix(pruned_members)
        representative_index = int(np.argmax(sim_matrix.mean(axis=1))) if sim_matrix.size else 0
        entry = ClusterEntry(
            label=label_name,
            centroid=centroid,
            members=pruned_members,
            representative_index=representative_index,
            similarity_matrix=sim_matrix if sim_matrix.size else None,
        )
        clusters.append(entry)
        assignments[label_name] = pruned_members

    return ClusterResult(clusters=clusters, unknown=unknown, assignments=assignments)


def _initialise_labels(
    samples: List[FaceSample], embeddings: np.ndarray, existing: Optional[List[ClusterSummary]], config: ClusterConfig
) -> List[int]:
    labels = [-1] * len(samples)
    if not existing:
        return labels
    centroids = np.stack([summary.centroid for summary in existing])
    for idx, vector in enumerate(embeddings):
        sims = _cosine_similarity(vector[None, :], centroids)[0]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= config.assign_threshold:
            labels[idx] = best_idx
    return labels


def _next_label(labels: List[int], existing: Optional[List[ClusterSummary]]) -> int:
    max_label = max(labels) if labels else -1
    if existing:
        max_label = max(max_label, len(existing) - 1)
    return max_label + 1


def _starting_person_index(existing: List[ClusterSummary]) -> int:
    max_idx = 0
    for summary in existing:
        try:
            suffix = int(summary.label.split("_")[-1])
        except ValueError:
            continue
        max_idx = max(max_idx, suffix)
    return max_idx + 1 if max_idx else 1


def _compute_centroid(members: List[FaceSample]) -> np.ndarray:
    matrix = np.stack([member.embedding for member in members])
    centroid = matrix.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    return centroid.astype("float32")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.clip(a_norm @ b_norm.T, -1.0, 1.0)


def _similarity_matrix(members: List[FaceSample]) -> np.ndarray:
    if not members:
        return np.zeros((0, 0), dtype="float32")
    matrix = np.stack([member.embedding for member in members])
    return _cosine_similarity(matrix, matrix)


def _prune_duplicates(members: List[FaceSample], threshold: float) -> List[FaceSample]:
    if len(members) <= 1:
        return members
    kept: List[FaceSample] = []
    embeddings = np.stack([m.embedding for m in members])
    sims = _cosine_similarity(embeddings, embeddings)
    removed = set()
    for i, member in enumerate(members):
        if i in removed:
            continue
        kept.append(member)
        for j in range(i + 1, len(members)):
            if sims[i, j] > threshold:
                removed.add(j)
    return kept


def load_existing_clusters(people_dir: Path) -> List[ClusterSummary]:
    summaries: List[ClusterSummary] = []
    if not people_dir.exists():
        return summaries
    for manifest_path in sorted(people_dir.glob("person_*")):
        manifest = manifest_path / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            LOGGER.warning("Skipping invalid manifest: %s", manifest)
            continue
        centroid = np.array(data.get("centroid", []), dtype="float32")
        if centroid.size == 0:
            continue
        summaries.append(ClusterSummary(label=manifest_path.name, centroid=centroid))
    return summaries


def save_cluster_manifest(cluster: ClusterEntry, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "label": cluster.label,
        "centroid": cluster.centroid.tolist(),
        "members": [
            {
                "image_path": str(sample.image_path),
                "bbox": list(sample.bbox),
                "score": sample.score,
                "crop_path": str(sample.crop_path) if sample.crop_path else None,
            }
            for sample in cluster.members
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(data, indent=2))


def save_unknown_manifest(samples: List[FaceSample], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "image_path": str(sample.image_path),
            "bbox": list(sample.bbox),
            "score": sample.score,
            "crop_path": str(sample.crop_path) if sample.crop_path else None,
        }
        for sample in samples
    ]
    (directory / "unknown.json").write_text(json.dumps(data, indent=2))

