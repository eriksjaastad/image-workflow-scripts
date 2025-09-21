"""Face grouping pipeline package."""

from .detector import DetectorConfig, FaceDetection, build_detector
from .embeddings import EmbeddingConfig, EmbeddingResult, build_embedder
from .cluster import (
    ClusterConfig,
    ClusterEntry,
    ClusterResult,
    FaceSample,
    cluster_faces,
    load_existing_clusters,
)
from .io import PipelinePaths, PipelineState

__all__ = [
    "DetectorConfig",
    "FaceDetection",
    "build_detector",
    "EmbeddingConfig",
    "EmbeddingResult",
    "build_embedder",
    "ClusterConfig",
    "ClusterEntry",
    "ClusterResult",
    "FaceSample",
    "cluster_faces",
    "load_existing_clusters",
    "PipelinePaths",
    "PipelineState",
]
