"""Face detection backends."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - fallback when OpenCV missing
    cv2 = None

LOGGER = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Detected face metadata."""

    image_path: Path
    bbox: tuple[int, int, int, int]
    score: float
    landmarks: Dict[str, tuple[float, float]]
    index: int
    face_patch: np.ndarray


@dataclass
class DetectorConfig:
    """Configuration shared across detector implementations."""

    min_score: float = 0.85
    max_size: int = 2048
    backend: str = "insightface"
    mediapipe_model: str = "short"
    simple_padding: int = 0
    allow_coreml: bool = True


class BaseDetector:
    """Abstract detector interface."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def detect(self, image: np.ndarray, image_path: Path) -> List[FaceDetection]:
        raise NotImplementedError

    def close(self) -> None:
        """Release detector resources."""


class InsightFaceDetector(BaseDetector):
    """RetinaFace detector via insightface."""

    def __init__(self, config: DetectorConfig) -> None:  # noqa: D401
        super().__init__(config)
        try:
            import insightface  # type: ignore
        except Exception as exc:  # pragma: no cover - import guarded
            raise RuntimeError("insightface is required for this backend") from exc

        providers: Optional[List[str]] = None
        if config.allow_coreml:
            providers = _preferred_providers()
            LOGGER.debug("InsightFace providers: %s", providers)

        self._app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, image: np.ndarray, image_path: Path) -> List[FaceDetection]:
        inference_image = image
        if inference_image.ndim == 3:
            inference_image = inference_image[:, :, ::-1].copy()
        faces = self._app.get(inference_image)
        detections: List[FaceDetection] = []
        for idx, face in enumerate(faces):
            score = float(getattr(face, "det_score", 0.0))
            if score < self.config.min_score:
                continue
            bbox = tuple(int(v) for v in face.bbox.astype(int))
            landmarks = {
                str(i): (float(face.landmark_2d[i][0]), float(face.landmark_2d[i][1]))
                for i in range(face.landmark_2d.shape[0])
            }
            patch = _crop_face(image, bbox, self.config.simple_padding)
            detections.append(
                FaceDetection(
                    image_path=image_path,
                    bbox=bbox,
                    score=score,
                    landmarks=landmarks,
                    index=idx,
                    face_patch=patch,
                )
            )
        return detections


class MediaPipeDetector(BaseDetector):
    """MediaPipe CPU fallback."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        try:
            from mediapipe.tasks import python  # type: ignore
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover - import guarded
            raise RuntimeError("mediapipe is not installed") from exc

        base_options = python.BaseOptions(model_asset_path=self._model_path())
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=config.min_score)
        self._detector = vision.FaceDetector.create_from_options(options)

    def _model_path(self) -> str:
        model_name = "face_detection_short_range.tflite" if self.config.mediapipe_model == "short" else "face_detection_full_range.tflite"
        default_path = Path(__file__).resolve().parent / "models" / model_name
        if default_path.exists():
            return str(default_path)
        raise FileNotFoundError(
            "MediaPipe model weights are not bundled. Please download the TFLite model and place it under faces/models/."
        )

    def detect(self, image: np.ndarray, image_path: Path) -> List[FaceDetection]:
        from mediapipe.tasks.python.vision import FaceDetectorResult  # type: ignore

        mp_image = _np_to_mp(image)
        result: FaceDetectorResult = self._detector.detect(mp_image)
        detections: List[FaceDetection] = []
        for idx, detection in enumerate(result.detections):
            score = float(detection.categories[0].score)
            if score < self.config.min_score:
                continue
            bbox = detection.bounding_box
            x1 = max(int(bbox.origin_x), 0)
            y1 = max(int(bbox.origin_y), 0)
            x2 = min(int(bbox.origin_x + bbox.width), image.shape[1])
            y2 = min(int(bbox.origin_y + bbox.height), image.shape[0])
            patch = _crop_face(image, (x1, y1, x2, y2), self.config.simple_padding)
            landmarks = {str(i): (lm.x * image.shape[1], lm.y * image.shape[0]) for i, lm in enumerate(detection.keypoints)}
            detections.append(
                FaceDetection(
                    image_path=image_path,
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    landmarks=landmarks,
                    index=idx,
                    face_patch=patch,
                )
            )
        return detections


class SimpleDetector(BaseDetector):
    """Fallback detector tuned for synthetic fixtures and debugging."""

    def detect(self, image: np.ndarray, image_path: Path) -> List[FaceDetection]:
        if float(np.std(image)) < 5.0:
            return []
        if cv2 is not None:
            bgr = image[:, :, ::-1] if image.ndim == 3 else image.copy()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else bgr.copy()
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(contour) for contour in contours]
        else:
            boxes = _numpy_bounding_boxes(image)
        detections: List[FaceDetection] = []
        idx = 0
        for x, y, w, h in boxes:
            if w < 32 or h < 32:
                continue
            score = float(min(0.99, 0.5 + (w * h) / (image.shape[0] * image.shape[1] + 1e-6)))
            if score < self.config.min_score:
                score = self.config.min_score
            patch = _crop_face(image, (x, y, x + w, y + h), self.config.simple_padding)
            center = (x + w / 2.0, y + h / 2.0)
            landmarks = {"center": center}
            detections.append(
                FaceDetection(
                    image_path=image_path,
                    bbox=(x, y, x + w, y + h),
                    score=score,
                    landmarks=landmarks,
                    index=idx,
                    face_patch=patch,
                )
            )
            idx += 1
        detections.sort(key=lambda det: det.bbox[0])
        return detections


def build_detector(config: DetectorConfig) -> BaseDetector:
    """Factory returning the desired detector backend."""

    backend = config.backend.lower()
    LOGGER.debug("Building detector backend: %s", backend)
    if backend == "insightface":
        try:
            return InsightFaceDetector(config)
        except Exception as exc:  # pragma: no cover - best effort fallback
            LOGGER.warning("InsightFace unavailable (%s). Falling back to simple detector.", exc)
            return SimpleDetector(config)
    if backend == "mediapipe":
        return MediaPipeDetector(config)
    if backend == "simple":
        return SimpleDetector(config)
    raise ValueError(f"Unsupported detector backend: {config.backend}")


def _crop_face(image: np.ndarray, bbox: tuple[int, int, int, int], padding: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    if padding:
        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, image.shape[1])
        y2 = min(y2 + padding, image.shape[0])
    return image[y1:y2, x1:x2].copy()


def _np_to_mp(image: np.ndarray):  # pragma: no cover - exercised when mediapipe installed
    from mediapipe.python import solutions

    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image
    return solutions.Image(image_format=solutions.ImageFormat.SRGB, data=rgb)


def _preferred_providers() -> List[str]:  # pragma: no cover - hardware-specific
    providers: List[str] = []
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        LOGGER.debug("onnxruntime not available; defaulting to CPU")
        return providers

    available = [p[0] for p in ort.get_available_providers()]
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    if "MetalExecutionProvider" in available:
        providers.append("MetalExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


def run_detector(detector: BaseDetector, images: Iterable[np.ndarray], paths: Iterable[Path]) -> List[List[FaceDetection]]:
    """Utility to batch-run detections."""

    results: List[List[FaceDetection]] = []
    for image, path in zip(images, paths):
        detections = detector.detect(image, path)
        results.append(detections)
    return results


def _numpy_bounding_boxes(image: np.ndarray) -> List[tuple[int, int, int, int]]:
    if image.ndim == 3:
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = image.astype(float)
    threshold = np.percentile(gray, 40)
    mask = gray <= threshold
    visited = np.zeros(mask.shape, dtype=bool)
    boxes: List[tuple[int, int, int, int]] = []
    height, width = mask.shape
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    min_component = max(256, int(0.005 * height * width))
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            ys: List[int] = []
            xs: List[int] = []
            count = 0
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                count += 1
                for dy, dx in neighbours:
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if not xs or count < min_component:
                continue
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            boxes.append((x1, y1, x2 - x1 + 1, y2 - y1 + 1))
    if not boxes:
        return boxes
    max_area = max(w * h for _, _, w, h in boxes)
    area_threshold = max(1024, max_area * 0.2)
    filtered = [box for box in boxes if box[2] * box[3] >= area_threshold]
    return filtered

