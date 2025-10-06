#!/usr/bin/env python3
"""
prezip_stager
--------------
Copy-only staging and zip for client deliverables using an allowlist-by-inventory
and global bans. Default is dry-run; require --commit to write outputs.

Usage:
  python scripts/tools/prezip_stager.py \
    --project-id mojo1 \
    --content-dir /abs/path/to/mojo1 \
    --output-zip /abs/path/to/out/mojo1_final.zip \
    [--allowlist-json /abs/path/to/allowed_ext.json] \
    [--bans-json /abs/path/to/bans.json] \
    [--recent-mins 10] [--no-require-full] [--allow-unknown] [--commit]

The module exposes prezip_stage(...) for programmatic use (tests).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Local import: scan_dir_state
try:
    from scripts.tools.scan_dir_state import scan as scan_dir_state_scan  # type: ignore
except Exception:
    # Fallback relative import when run from repo root
    try:
        from tools.scan_dir_state import scan as scan_dir_state_scan  # type: ignore
    except Exception:
        scan_dir_state_scan = None  # tests may disable require-full


DEFAULT_BANNED_EXT: Set[str] = {
    'project.json', 'project.yml', 'manifest.json', 'md', 'log', 'csv', 'json',
    'sqlite', 'db', 'lock', 'cropped'
}

# Note: 'json' is banned by default for internal files; allowlist overrides can whitelist specific cases per project.

DEFAULT_BANNED_PATTERNS: List[str] = [
    r".*\.project\.(json|yml)$",
    r".*\.cropped$",
    r"^\."  # dotfiles
]


@dataclass
class StagerConfig:
    project_id: str
    content_dir: Path
    output_zip: Path
    allowlist_json: Optional[Path]
    bans_json: Optional[Path]
    recent_mins: int = 10
    require_full: bool = True
    allow_unknown: bool = False
    commit: bool = False
    update_manifest: bool = False
    log_level: str = "info"
    strict_companions: bool = False


def load_allowlist(project_id: str, allowlist_json: Optional[Path]) -> Tuple[Set[str], Set[str]]:
    if allowlist_json is None:
        allowlist_json = Path('data/projects') / f"{project_id}_allowed_ext.json"
    if not allowlist_json.exists():
        raise FileNotFoundError(f"Allowlist JSON not found: {allowlist_json}")
    data = json.loads(allowlist_json.read_text(encoding='utf-8'))
    allowed = set(e.lower() for e in data.get('allowedExtensions', []) if isinstance(e, str))
    overrides = set(e.lower() for e in data.get('clientWhitelistOverrides', []) if isinstance(e, str))
    return allowed, overrides


def load_bans(bans_json: Optional[Path]) -> Tuple[Set[str], List[str]]:
    banned_ext = set(DEFAULT_BANNED_EXT)
    banned_patterns = list(DEFAULT_BANNED_PATTERNS)
    if bans_json and bans_json.exists():
        data = json.loads(bans_json.read_text(encoding='utf-8'))
        for e in data.get('bannedExtensions', []) or []:
            banned_ext.add(str(e).lower())
        for p in data.get('bannedPatterns', []) or []:
            banned_patterns.append(str(p))
    return banned_ext, banned_patterns


def is_hidden(name: str) -> bool:
    return name.startswith('.')


def matches_banned_patterns(name: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if re.match(pat, name):
            return True
    return False


def relpath_under(root: Path, p: Path) -> Path:
    try:
        return p.relative_to(root)
    except ValueError:
        return Path(p.name)


def stem_of(p: Path) -> str:
    return p.stem


def prezip_stage(cfg: StagerConfig) -> Dict:
    # Precondition: require FULL unless disabled
    if cfg.require_full and scan_dir_state_scan is not None:
        state = scan_dir_state_scan(cfg.content_dir, cfg.recent_mins)
        if state.get('state') != 'FULL':
            return {
                'status': 'error',
                'message': f"Directory not FULL (state={state.get('state')}). Use --no-require-full or wait.",
                'state': state
            }

    allowed, overrides = load_allowlist(cfg.project_id, cfg.allowlist_json)
    banned_ext, banned_patterns = load_bans(cfg.bans_json)
    eligible: List[Path] = []
    excluded: Dict[str, List[str]] = {
        'hidden': [], 'banned_ext': [], 'banned_pattern': [], 'not_allowed': [], 'no_extension': []
    }
    by_ext_included: Dict[str, int] = {}
    # Track companions by stem for validation
    stem_to_included_exts: Dict[str, Set[str]] = {}
    stem_to_all_exts: Dict[str, Set[str]] = {}

    for dirpath, dirnames, filenames in os.walk(cfg.content_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for name in filenames:
            p = Path(dirpath) / name
            rel = relpath_under(cfg.content_dir, p)
            if is_hidden(name):
                excluded['hidden'].append(str(rel))
                continue
            ext = p.suffix.lower().lstrip('.')
            if not ext:
                excluded['no_extension'].append(str(rel))
                continue
            if ext in banned_ext:
                excluded['banned_ext'].append(str(rel))
                continue
            if matches_banned_patterns(name, banned_patterns):
                excluded['banned_pattern'].append(str(rel))
                continue
            if ext not in allowed and ext not in overrides:
                excluded['not_allowed'].append(str(rel))
                continue
            eligible.append(p)
            by_ext_included[ext] = by_ext_included.get(ext, 0) + 1
            st = stem_of(p)
            stem_to_included_exts.setdefault(st, set()).add(ext)

            # Discover companions in the same directory for validation
            try:
                for sib in p.parent.iterdir():
                    if sib.is_file() and sib.stem == st:
                        se = sib.suffix.lower().lstrip('.')
                        if se:
                            stem_to_all_exts.setdefault(st, set()).add(se)
            except Exception:
                pass

    report = {
        'status': 'ok',
        'projectId': cfg.project_id,
        'scannedAt': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'contentDir': str(cfg.content_dir),
        'eligibleCount': len(eligible),
        'byExtIncluded': by_ext_included,
        'excludedCounts': {k: len(v) for k, v in excluded.items()},
    }

    # Companion integrity validation: ensure that for each stem, all allowed companions present
    companion_issues: List[Dict[str, object]] = []
    allow_set = allowed.union(overrides)
    for st, all_exts in stem_to_all_exts.items():
        included_exts = stem_to_included_exts.get(st, set())
        should_have = {e for e in all_exts if e in allow_set and e not in banned_ext}
        missing = sorted(list(should_have - included_exts))
        if missing:
            companion_issues.append({'stem': st, 'missingAllowedCompanions': missing})
    if companion_issues:
        report['companionIssues'] = companion_issues
        if cfg.strict_companions:
            return report | {'status': 'error', 'message': 'Companion integrity check failed'}

    if not cfg.commit:
        return report | {'staging': None, 'zip': None, 'dryRun': True}

    # Commit: copy to staging and create zip
    out_zip = cfg.output_zip.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = out_zip.with_suffix('').as_posix() + '.staging'
    staging_path = Path(staging_dir)
    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True, exist_ok=True)

    for p in eligible:
        rel = relpath_under(cfg.content_dir, p)
        dest = staging_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

    # Create zip
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(staging_path):
            for f in files:
                fp = Path(root) / f
                arc = relpath_under(staging_path, fp)
                zf.write(fp, arcname=str(arc))

    report.update({'staging': str(staging_path), 'zip': str(out_zip), 'dryRun': False})

    # Optional manifest update
    if cfg.update_manifest:
        try:
            manifest_path = Path('data/projects') / f"{cfg.project_id}.project.json"
            if manifest_path.exists():
                m = json.loads(manifest_path.read_text(encoding='utf-8'))
                now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                m.setdefault('metrics', {})
                m['metrics']['stager'] = {
                    'zip': str(out_zip),
                    'eligibleCount': report['eligibleCount'],
                    'byExtIncluded': report['byExtIncluded'],
                    'excludedCounts': report['excludedCounts']
                }
                if not m.get('finishedAt'):
                    m['finishedAt'] = now
                # Backup then write
                backup = manifest_path.with_suffix('.project.json.bak')
                try:
                    shutil.copy2(manifest_path, backup)
                except Exception:
                    pass
                manifest_path.write_text(json.dumps(m, indent=2), encoding='utf-8')
                report['manifestUpdated'] = True
            else:
                report['manifestUpdated'] = False
        except Exception as e:
            report['manifestUpdated'] = False
            report['manifestError'] = str(e)
    return report


def parse_args() -> StagerConfig:
    ap = argparse.ArgumentParser(description='Pre-zip stager (copy-only)')
    ap.add_argument('--project-id', required=True)
    ap.add_argument('--content-dir', required=True)
    ap.add_argument('--output-zip', required=True)
    ap.add_argument('--allowlist-json')
    ap.add_argument('--bans-json', default=str(Path('data/projects/global_bans.json').resolve()))
    ap.add_argument('--recent-mins', type=int, default=10)
    ap.add_argument('--no-require-full', action='store_true')
    ap.add_argument('--allow-unknown', action='store_true')
    ap.add_argument('--commit', action='store_true')
    ap.add_argument('--update-manifest', action='store_true')
    ap.add_argument('--log-level', default='info', choices=['info','debug'])
    ap.add_argument('--strict-companions', action='store_true')
    args = ap.parse_args()
    return StagerConfig(
        project_id=args.project_id,
        content_dir=Path(args.content_dir).resolve(),
        output_zip=Path(args.output_zip).resolve(),
        allowlist_json=Path(args.allowlist_json).resolve() if args.allowlist_json else None,
        bans_json=Path(args.bans_json).resolve() if args.bans_json else None,
        recent_mins=args.recent_mins,
        require_full=not args.no_require_full,
        allow_unknown=args.allow_unknown,
        commit=args.commit,
        update_manifest=args.update_manifest,
        log_level=args.log_level,
        strict_companions=args.strict_companions,
    )


def main() -> None:
    cfg = parse_args()
    try:
        report = prezip_stage(cfg)
    except Exception as e:
        print(json.dumps({'status': 'error', 'message': str(e)}))
        raise SystemExit(1)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()


