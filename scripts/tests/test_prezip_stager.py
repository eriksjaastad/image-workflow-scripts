import json
import os
from pathlib import Path

import unittest


class TestPrezipStager(unittest.TestCase):
    def setUp(self):
        self.tmp = Path('scripts/tests/tmp_prezip')
        self.tmp.mkdir(parents=True, exist_ok=True)
        # Create fake content tree
        self.content = self.tmp / 'content'
        (self.content / 'a').mkdir(parents=True, exist_ok=True)
        (self.content / 'b').mkdir(parents=True, exist_ok=True)
        (self.content / 'a' / 'img1.png').write_bytes(b'PNG')
        (self.content / 'a' / 'img1.yaml').write_text('meta: 1')
        (self.content / 'b' / 'img2.png').write_bytes(b'PNG')
        # Hidden file should be excluded
        (self.content / '.DS_Store').write_text('')

        # Allowlist JSON
        self.allowlist = self.tmp / 'mojo1_allowed_ext.json'
        self.allowlist.write_text(json.dumps({
            'projectId': 'mojo1',
            'snapshotAt': '2025-10-06T00:00:00Z',
            'sourcePath': str(self.content),
            'allowedExtensions': ['png', 'yaml'],
            'clientWhitelistOverrides': []
        }), encoding='utf-8')

        # Output zip path
        self.out_zip = self.tmp / 'out' / 'final.zip'

    def tearDown(self):
        # Cleanup tmp directory
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

    def test_dry_run(self):
        from scripts.tools.prezip_stager import StagerConfig, prezip_stage
        cfg = StagerConfig(
            project_id='mojo1',
            content_dir=self.content.resolve(),
            output_zip=self.out_zip.resolve(),
            allowlist_json=self.allowlist.resolve(),
            bans_json=None,
            recent_mins=0,
            require_full=False,
            allow_unknown=False,
            commit=False,
        )
        report = prezip_stage(cfg)
        self.assertEqual(report['status'], 'ok')
        self.assertTrue(report['dryRun'])
        self.assertEqual(report['excludedCounts']['hidden'], 1)
        # Both png files plus one yaml should be eligible (3)
        self.assertEqual(report['eligibleCount'], 3)

    def test_commit_creates_zip(self):
        from scripts.tools.prezip_stager import StagerConfig, prezip_stage
        cfg = StagerConfig(
            project_id='mojo1',
            content_dir=self.content.resolve(),
            output_zip=self.out_zip.resolve(),
            allowlist_json=self.allowlist.resolve(),
            bans_json=None,
            recent_mins=0,
            require_full=False,
            allow_unknown=False,
            commit=True,
            update_manifest=False,
            strict_companions=False,
        )
        report = prezip_stage(cfg)
        self.assertEqual(report['status'], 'ok')
        self.assertFalse(report['dryRun'])
        self.assertTrue(Path(report['zip']).exists())


if __name__ == '__main__':
    unittest.main()


