# Face Grouping Pipeline

A modular face detection, embedding, and clustering pipeline optimised for Apple Silicon (macOS Sequoia, M-series). It uses **insightface** (RetinaFace + ArcFace) by default and automatically falls back to CPU execution when CoreML/Metal execution providers are unavailable. An optional MediaPipe detector backend is also exposed behind a CLI flag, and a lightweight "simple" backend is bundled for testing and troubleshooting.

## Features
- Recursive image discovery with EXIF orientation handling.
- RetinaFace detection + ArcFace embeddings via `onnxruntime` (CoreML/Metal preferred on Apple Silicon).
- Configurable clustering (Agglomerative or DBSCAN) with sensible cosine-distance defaults.
- Incremental assignment into existing identity folders with configurable similarity thresholds.
- Deterministic per-person directories containing representative crops, member crops, manifests, and optional original images (hard-linked when possible).
- Embedding/metadata persistence (`embeddings.npy` + `metadata.json`) for downstream analytics or resume.
- Resume support (`--resume`) and progress reporting.
- Optional annotation overlays, montages, and detection-only JSON output.

## Installation (macOS Sequoia on M-series)
1. Install Python 3.10–3.12 (e.g. via [pyenv](https://github.com/pyenv/pyenv)).
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Optional: Enable CoreML / Metal execution
`onnxruntime` ships CPU wheels by default. To leverage CoreML/Metal on Apple Silicon:
```bash
pip install onnxruntime-silicon
```
The pipeline automatically detects the CoreML and Metal execution providers when available and falls back to CPU otherwise, logging the chosen provider when `--verbose` is set.

### Optional: MediaPipe detector
MediaPipe is behind the `--backend mediapipe` flag. Install it only if you intend to use that backend:
```bash
pip install mediapipe
# Download the required TFLite detector and place it under faces/models/
#  faces/models/face_detection_short_range.tflite
```

## Configuration
Default values live in [`config.yaml`](config.yaml). All options are overridable via CLI flags. Key thresholds:
- `min_score`: 0.85 (detector confidence)
- `max_size`: 2048 (resize longest side before detection)
- `distance_threshold`: 0.40 (Agglomerative cosine distance)
- `dbscan_eps`: 0.38, `dbscan_min_samples`: 3
- `min_cluster_size`: 3
- `assign_threshold`: 0.60 (incremental cosine similarity)
- `near_dup_threshold`: 0.98 (prune near-duplicates inside a cluster)

## Quick Start
### Detect + embed + cluster
```bash
python main.py --images ./photos --out ./out --save-annots --bench
```

### Use DBSCAN with a custom distance threshold
```bash
python main.py --images ./photos --out ./out --cluster dbscan --dbscan-eps 0.35
```

### Incremental assignment into an existing people directory
```bash
python main.py --images ./new_batch --out ./out --incremental --people-dir ./out/people
```

### Detection-only JSON export
```bash
python main.py --images ./photos --out ./out --detect-only --json ./out/detections.json
```

### Troubleshooting low recall
- Lower `--min-score`.
- Increase `--max-size` to allow larger inputs before resizing.
- Switch detectors (`--backend mediapipe` or `--backend simple` for debugging).
- Try DBSCAN with a looser `--dbscan-eps` when agglomerative merges identities too aggressively.

## Development
Run tests:
```bash
pytest
```

The synthetic fixtures in `tests/` ensure:
- A blank 1024×1024 canvas yields zero faces.
- Two distinct synthetic faces yield two clusters (DBSCAN with min cluster size of 1).

## Outputs
```
out/
├── embeddings.npy          # stacked 512-D embeddings
├── metadata.json           # bbox, score, landmarks per face
├── people/
│   ├── person_0001/
│   │   ├── faces/*.jpg     # face crops
│   │   ├── rep.jpg         # best representative crop
│   │   └── manifest.json   # members + centroid
│   └── ...
├── unknown/
│   ├── faces/*.jpg         # outliers + small clusters
│   └── unknown.json
└── detections.json         # optional detection-only export
```

Use `--copy-originals` to hard-link/copy source photos into each identity folder under `originals/`.

## Logging
Enable verbose logging with `--verbose` to print detector/backend selection, execution providers, batch sizes, and summary statistics. Progress bars provide per-image status, and `--bench` reports total runtime.

