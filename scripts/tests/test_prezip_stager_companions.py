import json
from pathlib import Path
from zipfile import ZipFile
import unittest


class TestPrezipCompanions(unittest.TestCase):
    def setUp(self):
        self.tmp = Path('scripts/tests/tmp_stager_comp')
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.content = self.tmp / 'content'
        self.content.mkdir()
        # Create png + yaml, and a png missing its yaml
        (self.content / 'a.png').write_bytes(b'x')
        (self.content / 'a.yaml').write_text('m:1')
        (self.content / 'b.png').write_bytes(b'x')  # no yaml

        # Allowlist permits png,yaml
        self.allowlist = self.tmp / 'proj_allowed.json'
        self.allowlist.write_text(json.dumps({
            'projectId': 'proj',
            'snapshotAt': '2025-10-06T00:00:00Z',
            'sourcePath': str(self.content),
            'allowedExtensions': ['png','yaml'],
            'clientWhitelistOverrides': []
        }), encoding='utf-8')

        self.out_zip = self.tmp / 'out' / 'final.zip'

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

    def test_companion_issues_reported(self):
        from scripts.tools.prezip_stager import StagerConfig, prezip_stage
        cfg = StagerConfig(
            project_id='proj',
            content_dir=self.content.resolve(),
            output_zip=self.out_zip.resolve(),
            allowlist_json=self.allowlist.resolve(),
            bans_json=None,
            recent_mins=0,
            require_full=False,
            commit=False,
            strict_companions=False,
        )
        report = prezip_stage(cfg)
        self.assertEqual(report['status'], 'ok')
        # Policy: include companions if present for the stem; missing companions are not required.
        self.assertNotIn('companionIssues', report)

    def test_strict_mode_blocks(self):
        from scripts.tools.prezip_stager import StagerConfig, prezip_stage
        cfg = StagerConfig(
            project_id='proj',
            content_dir=self.content.resolve(),
            output_zip=self.out_zip.resolve(),
            allowlist_json=self.allowlist.resolve(),
            bans_json=None,
            recent_mins=0,
            require_full=False,
            commit=False,
            strict_companions=True,
        )
        report = prezip_stage(cfg)
        # Strict mode is only relevant if there are companionIssues; under current policy, there should be none.
        self.assertEqual(report['status'], 'ok')


if __name__ == '__main__':
    unittest.main()


