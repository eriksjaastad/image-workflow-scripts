"""Smoke tests for the face grouping pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import PipelineConfig, run_pipeline


def _make_blank(tmp_path: Path) -> Path:
    img = Image.new("RGB", (1024, 1024), color=(255, 255, 255))
    path = tmp_path / "blank.png"
    img.save(path)
    return path


def _make_face(tmp_path: Path, name: str, color: tuple[int, int, int]) -> Path:
    img = Image.new("RGB", (512, 512), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 60, 452, 452), fill=color)
    draw.ellipse((180, 200, 230, 250), fill=(0, 0, 0))
    draw.ellipse((280, 200, 330, 250), fill=(0, 0, 0))
    draw.arc((200, 280, 320, 380), start=0, end=180, fill=(0, 0, 0), width=6)
    path = tmp_path / f"{name}.png"
    img.save(path)
    return path


def test_blank_image_has_no_faces(tmp_path):
    images_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    images_dir.mkdir()
    _make_blank(images_dir)

    config = PipelineConfig(
        images=images_dir,
        out_dir=out_dir,
        backend="simple",
        embedder_backend="simple",
        detect_only=False,
        min_score=0.5,
        min_cluster_size=1,
    )
    result = run_pipeline(config)
    assert result.faces_detected == 0
    assert result.clusters_created == 0


def test_two_synthetic_faces_form_distinct_clusters(tmp_path):
    images_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    images_dir.mkdir()
    _make_face(images_dir, "face_red", (220, 80, 80))
    _make_face(images_dir, "face_blue", (80, 120, 220))

    config = PipelineConfig(
        images=images_dir,
        out_dir=out_dir,
        backend="simple",
        embedder_backend="simple",
        cluster_method="dbscan",
        dbscan_eps=0.05,
        dbscan_min_samples=1,
        min_cluster_size=1,
        min_score=0.5,
    )
    result = run_pipeline(config)
    assert result.faces_detected == 2
    assert result.clusters_created == 2

    metadata_path = out_dir / "metadata.json"
    assert metadata_path.exists()
    data = metadata_path.read_text()
    assert "face_red" in data and "face_blue" in data

    people_dir = out_dir / "people"
    clusters = sorted(p for p in people_dir.glob("person_*"))
    assert len(clusters) == 2
    for cluster in clusters:
        rep = cluster / "rep.jpg"
        assert rep.exists()
        manifest = cluster / "manifest.json"
        assert manifest.exists()

