"""Face embedding backends."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Embedding metadata."""

    vector: np.ndarray
    norm: float


@dataclass
class EmbeddingConfig:
    """Configuration for embedders."""

    batch_size: int = 16
    backend: str = "insightface"
    allow_coreml: bool = True


class BaseEmbedder:
    """Interface for embedding extraction."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    def embed_batch(self, faces: Iterable[np.ndarray]) -> List[EmbeddingResult]:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources."""


class InsightFaceEmbedder(BaseEmbedder):
    """ArcFace embeddings via insightface."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        try:
            import insightface  # type: ignore
        except Exception as exc:  # pragma: no cover - import guarded
            raise RuntimeError("insightface is required for the ArcFace embedder") from exc

        providers = None
        if config.allow_coreml:
            from .detector import _preferred_providers  # local import

            providers = _preferred_providers()
            LOGGER.debug("InsightFace embedder providers: %s", providers)
        self._model = insightface.model_zoo.get_model("arcface_r100_v1", providers=providers)
        self._model.prepare(ctx_id=0)

    def embed_batch(self, faces: Iterable[np.ndarray]) -> List[EmbeddingResult]:
        results: List[EmbeddingResult] = []
        for face in faces:
            if face.size == 0:
                continue
            if face.ndim == 2:
                face_rgb = np.stack([face] * 3, axis=-1)
            else:
                face_rgb = face
            face_bgr = face_rgb[:, :, ::-1].copy()
            embedding = self._model.get_embedding(face_bgr)
            vec = np.asarray(embedding).astype("float32")
            norm = float(np.linalg.norm(vec) + 1e-12)
            vec /= norm
            results.append(EmbeddingResult(vector=vec, norm=norm))
        return results


class SimpleEmbedder(BaseEmbedder):
    """Lightweight, deterministic embedding fallback."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)

    def embed_batch(self, faces: Iterable[np.ndarray]) -> List[EmbeddingResult]:
        results: List[EmbeddingResult] = []
        for face in faces:
            if face.size == 0:
                continue
            vec = _simple_descriptor(face)
            norm = float(np.linalg.norm(vec) + 1e-12)
            vec = vec / norm
            results.append(EmbeddingResult(vector=vec.astype("float32"), norm=norm))
        return results


def build_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    backend = config.backend.lower()
    LOGGER.debug("Building embedder backend: %s", backend)
    if backend == "insightface":
        try:
            return InsightFaceEmbedder(config)
        except Exception as exc:  # pragma: no cover - best effort fallback
            LOGGER.warning("InsightFace embedder unavailable (%s); using simple embedder.", exc)
            return SimpleEmbedder(config)
    if backend == "simple":
        return SimpleEmbedder(config)
    raise ValueError(f"Unsupported embedder backend: {backend}")


def _simple_descriptor(face: np.ndarray, size: int = 64) -> np.ndarray:
    rgb = face.astype("float32") / 255.0 if face.ndim == 3 else np.stack([face] * 3, axis=-1) / 255.0
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    hist_r, _ = np.histogram(rgb[..., 0], bins=128, range=(0.0, 1.0))
    hist_g, _ = np.histogram(rgb[..., 1], bins=128, range=(0.0, 1.0))
    hist_b, _ = np.histogram(rgb[..., 2], bins=128, range=(0.0, 1.0))
    hist_gray, _ = np.histogram(gray, bins=128, range=(0.0, 1.0))
    descriptor = np.concatenate([hist_r, hist_g, hist_b, hist_gray]).astype("float32")
    return descriptor

