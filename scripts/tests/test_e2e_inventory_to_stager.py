import json
import unittest
from pathlib import Path
from zipfile import ZipFile


class TestEndToEndInventoryToStager(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("scripts/tests/tmp_e2e")
        self.tmp.mkdir(parents=True, exist_ok=True)
        # Build content tree
        self.content = self.tmp / "content"
        (self.content / "images").mkdir(parents=True, exist_ok=True)
        (self.content / "notes").mkdir(parents=True, exist_ok=True)  # no images here
        # Files under images
        (self.content / "images" / "x1.png").write_bytes(b"png")
        (self.content / "images" / "x1.yaml").write_text("meta: 1")
        (self.content / "images" / "x2.png").write_bytes(b"png")
        # Non-image under root should not affect allowlist
        (self.content / "readme.txt").write_text("txt")
        # Hidden dir/file
        (self.content / ".hidden").mkdir()
        (self.content / ".hidden" / "whatever.png").write_bytes(b"png")

        # Allowlist file (derive from this tree)
        self.allowlist = self.tmp / "proj_allowed_ext.json"
        self.allowlist.write_text(
            json.dumps(
                {
                    "projectId": "proj",
                    "snapshotAt": "2025-10-06T00:00:00Z",
                    "sourcePath": str(self.content),
                    "allowedExtensions": ["png", "yaml"],
                    "clientWhitelistOverrides": [],
                }
            ),
            encoding="utf-8",
        )

        self.out_zip = self.tmp / "out" / "final.zip"

    def tearDown(self):
        def _rm(p: Path):
            if not p.exists():
                return
            if p.is_file() or p.is_symlink():
                p.unlink()
                return
            for c in p.iterdir():
                _rm(c)
            p.rmdir()

        _rm(self.tmp)

    def test_e2e_dryrun_then_commit(self):
        from scripts.tools.prezip_stager import StagerConfig, prezip_stage

        # Dry-run
        cfg = StagerConfig(
            project_id="proj",
            content_dir=self.content.resolve(),
            output_zip=self.out_zip.resolve(),
            allowlist_json=self.allowlist.resolve(),
            bans_json=None,
            recent_mins=0,
            require_full=False,
            commit=False,
        )
        report = prezip_stage(cfg)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["dryRun"])
        # Expect x1.png + x1.yaml + x2.png
        self.assertEqual(report["eligibleCount"], 3)

        # Commit
        cfg.commit = True
        report2 = prezip_stage(cfg)
        self.assertEqual(report2["status"], "ok")
        self.assertFalse(report2["dryRun"])
        self.assertTrue(Path(report2["zip"]).exists())
        # Check zip contents
        with ZipFile(report2["zip"], "r") as zf:
            names = set(zf.namelist())
        # Relative to staging root mirrors content
        self.assertIn("images/x1.png", names)
        self.assertIn("images/x1.yaml", names)
        self.assertIn("images/x2.png", names)
        self.assertNotIn("readme.txt", names)


if __name__ == "__main__":
    unittest.main()
