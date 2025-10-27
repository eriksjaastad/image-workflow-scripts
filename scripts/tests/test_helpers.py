import shutil
import tempfile
from pathlib import Path


def make_temp_images(count: int = 3):
    base = Path(tempfile.mkdtemp())
    src = base / "src"
    src.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (src / f"20250101_00000{i}_stage1_generated.png").write_bytes(b"fake")
        (src / f"20250101_00000{i}_stage1_generated.yaml").write_text("meta: 1")
    dest_selected = base / "selected"
    dest_crop = base / "crop"
    dest_selected.mkdir(exist_ok=True)
    dest_crop.mkdir(exist_ok=True)
    return base, src, dest_selected, dest_crop


def cleanup_temp(base: Path):
    shutil.rmtree(base, ignore_errors=True)
