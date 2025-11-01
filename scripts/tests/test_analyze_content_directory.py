import shutil
import tempfile
from pathlib import Path


def test_analyze_helpers_smoke():
    # Import helpers without executing CLI
    import importlib.util
    import sys

    target = Path("scripts/tools/analyze_content_directory.py")
    spec = importlib.util.spec_from_file_location("analyze_content_directory", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_content_directory"] = mod
    spec.loader.exec_module(mod)  # type: ignore

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # create tiny structure
        (tmpdir / "a.png").write_bytes(b"0")
        (tmpdir / "a.yaml").write_text("x: 1")
        (tmpdir / "b.caption").write_text("caption")
        (tmpdir / "sub").mkdir()
        (tmpdir / "sub" / "c.png").write_bytes(b"0")

        exts = mod.analyze_file_extensions(tmpdir)
        assert exts.get(".png", 0) == 2

        comps = mod.analyze_companion_files(tmpdir)
        assert comps["total_images"] == 2

        struct = mod.analyze_directory_structure(tmpdir)
        assert struct["total_files"] >= 3
    finally:
        shutil.rmtree(tmpdir)
