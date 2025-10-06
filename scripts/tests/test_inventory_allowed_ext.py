import json
import os
from pathlib import Path
import unittest


class TestInventoryAllowedExt(unittest.TestCase):
    def setUp(self):
        self.tmp = Path('scripts/tests/tmp_inv')
        self.tmp.mkdir(parents=True, exist_ok=True)
        # content root
        self.content = self.tmp / 'content'
        self.content.mkdir()
        # dirs: images/, notes/ (no images)
        (self.content / 'images').mkdir()
        (self.content / 'notes').mkdir()
        # files
        (self.content / 'images' / 'a.png').write_bytes(b'PNG')
        (self.content / 'images' / 'a.yaml').write_text('meta: 1')
        (self.content / 'notes' / 'readme.txt').write_text('txt')

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

    def test_collect_skips_non_image_topdir(self):
        from scripts.tools.inventory_allowed_ext import collect_extensions
        exts = collect_extensions(self.content, {"png", "jpg"})
        # yaml counts only if present under an image-bearing top-level dir (images/)
        # readme.txt under notes/ should not cause notes/ to be scanned
        self.assertIn('png', exts)
        self.assertNotIn('txt', exts)


if __name__ == '__main__':
    unittest.main()


