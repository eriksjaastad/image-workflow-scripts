#!/usr/bin/env python3
"""
prezip_stager
--------------
Copy-only staging and zip for client deliverables using an allowlist-by-inventory
and global bans. Default is dry-run; require --commit to write outputs.

Usage (from repo root, relative paths):
  # Dry run (default), show progress
  python scripts/tools/prezip_stager.py \
    --project-id mojo1 \
    --content-dir mojo1 \
    --output-zip mojo1_final.zip

  # Create zip, update manifest, minimal options
  python scripts/tools/prezip_stager.py \
    --project-id mojo1 \
    --content-dir mojo1 \
    --output-zip mojo1_final.zip \
    --commit --update-manifest

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
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

# String patterns are kept for readability; they are compiled at import time
DEFAULT_BANNED_PATTERNS: List[str] = [
    r".*\.project\.(json|yml)$",
    r".*\.cropped$",
    r"^\."  # dotfiles
]

# Pre-compile default regex patterns once
DEFAULT_BANNED_REGEX = [re.compile(p) for p in DEFAULT_BANNED_PATTERNS]


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
    quiet: bool = False
    strict_companions: bool = False
    check_companions: bool = False
    compression: str = "stored"  # stored|deflated|bzip2|lzma
    compress_level: Optional[int] = None


def load_allowlist(project_id: str, allowlist_json: Optional[Path]) -> Tuple[Set[str], Set[str]]:
    if allowlist_json is None:
        allowlist_json = Path('data/projects') / f"{project_id}_allowed_ext.json"
    if not allowlist_json.exists():
        raise FileNotFoundError(f"Allowlist JSON not found: {allowlist_json}")
    data = json.loads(allowlist_json.read_text(encoding='utf-8'))
    allowed = set(e.lower() for e in data.get('allowedExtensions', []) if isinstance(e, str))
    overrides = set(e.lower() for e in data.get('clientWhitelistOverrides', []) if isinstance(e, str))
    return allowed, overrides


def load_bans(bans_json: Optional[Path]) -> Tuple[Set[str], List[re.Pattern], Set[str]]:
    banned_ext = set(DEFAULT_BANNED_EXT)
    # Start from precompiled defaults
    banned_regex: List[re.Pattern] = list(DEFAULT_BANNED_REGEX)
    banned_basenames: Set[str] = set()
    if bans_json and bans_json.exists():
        data = json.loads(bans_json.read_text(encoding='utf-8'))
        for e in data.get('bannedExtensions', []) or []:
            banned_ext.add(str(e).lower())
        for p in data.get('bannedPatterns', []) or []:
            try:
                banned_regex.append(re.compile(str(p)))
            except Exception:
                # Skip invalid patterns from config
                pass
        for n in data.get('bannedBasenames', []) or []:
            if isinstance(n, str) and n.strip():
                banned_basenames.add(n.strip().lower())
    return banned_ext, banned_regex, banned_basenames


def is_hidden(name: str) -> bool:
    return name.startswith('.')


def matches_banned_patterns(name: str, patterns: List[re.Pattern]) -> bool:
    for pat in patterns:
        # Use search to allow unanchored patterns naturally
        if pat.search(name):
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
    banned_ext, banned_patterns, banned_basenames = load_bans(cfg.bans_json)
    eligible: List[Path] = []
    # Track counts only to reduce memory overhead; also track unique filenames for reporting
    excluded: Dict[str, int] = {
        'hidden': 0, 'banned_name': 0, 'banned_ext': 0, 'banned_pattern': 0, 'not_allowed': 0, 'no_extension': 0
    }
    excluded_unique_names: Dict[str, Set[str]] = {
        'hidden': set(), 'banned_name': set(), 'banned_ext': set(), 'banned_pattern': set(), 'not_allowed': set(), 'no_extension': set()
    }
    by_ext_included: Dict[str, int] = {}
    # Pre-filter inventory of what exists by extension (non-hidden files only)
    incoming_by_ext: Dict[str, int] = {}
    # Track companions by stem for validation
    stem_to_included_exts: Dict[str, Set[str]] = {}
    stem_to_all_exts: Dict[str, Set[str]] = {}
    do_companion_checks = bool(cfg.strict_companions or cfg.check_companions)
    scanned_files = 0

    for dirpath, dirnames, filenames in os.walk(cfg.content_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        # If doing companion checks, aggregate per-directory once from filenames
        if do_companion_checks and filenames:
            per_dir_map: Dict[str, Set[str]] = {}
            for name in filenames:
                st_local = Path(name).stem
                ext_local = Path(name).suffix.lower().lstrip('.')
                if ext_local:
                    per_dir_map.setdefault(st_local, set()).add(ext_local)
            # Merge into global map
            for st_key, ext_set in per_dir_map.items():
                acc = stem_to_all_exts.setdefault(st_key, set())
                acc.update(ext_set)
        for name in filenames:
            p = Path(dirpath) / name
            rel = relpath_under(cfg.content_dir, p)
            if is_hidden(name):
                excluded['hidden'] += 1
                excluded_unique_names['hidden'].add(Path(name).name)
                continue
            ext = p.suffix.lower().lstrip('.')
            # Record incoming inventory for any non-hidden file with an extension
            if ext:
                incoming_by_ext[ext] = incoming_by_ext.get(ext, 0) + 1
            if not ext:
                excluded['no_extension'] += 1
                excluded_unique_names['no_extension'].add(Path(name).name)
                continue
            base_name = Path(name).name.lower()
            if base_name in banned_basenames:
                excluded['banned_name'] += 1
                excluded_unique_names['banned_name'].add(Path(name).name)
                continue
            if ext in banned_ext:
                excluded['banned_ext'] += 1
                excluded_unique_names['banned_ext'].add(Path(name).name)
                continue
            if matches_banned_patterns(name, banned_patterns):
                excluded['banned_pattern'] += 1
                excluded_unique_names['banned_pattern'].add(Path(name).name)
                continue
            if ext not in allowed and ext not in overrides:
                excluded['not_allowed'] += 1
                excluded_unique_names['not_allowed'].add(Path(name).name)
                continue
            eligible.append(p)
            by_ext_included[ext] = by_ext_included.get(ext, 0) + 1
            st = stem_of(p)
            stem_to_included_exts.setdefault(st, set()).add(ext)
            scanned_files += 1
            if not cfg.quiet and scanned_files % 5000 == 0:
                print(f"[scan] scanned={scanned_files} eligible={len(eligible)}", file=sys.stderr, flush=True)

    report = {
        'status': 'ok',
        'projectId': cfg.project_id,
        'scannedAt': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'contentDir': str(cfg.content_dir),
        'eligibleCount': len(eligible),
        'byExtIncluded': by_ext_included,
        'excludedCounts': {k: int(v) for k, v in excluded.items()},
        'excludedUniqueNames': {k: sorted(list(v)) for k, v in excluded_unique_names.items()},
        'incomingByExt': incoming_by_ext,
    }

    # Companion integrity validation: ensure that for each stem, all allowed companions present
    if do_companion_checks:
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

    # Commit: stream directly into zip (no staging directory)
    out_zip = cfg.output_zip.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    # Resolve compression method
    comp_map = {
        'stored': zipfile.ZIP_STORED,
        'deflated': zipfile.ZIP_DEFLATED,
        'bzip2': zipfile.ZIP_BZIP2,
        'lzma': zipfile.ZIP_LZMA,
    }
    comp_method = comp_map.get(cfg.compression.lower(), zipfile.ZIP_DEFLATED)
    zip_kwargs = {'compression': comp_method}
    if cfg.compress_level is not None:
        zip_kwargs['compresslevel'] = cfg.compress_level

    zipped_count = 0
    total_to_zip = len(eligible)
    with zipfile.ZipFile(out_zip, 'w', **zip_kwargs) as zf:
        for p in eligible:
            arc = relpath_under(cfg.content_dir, p)
            zf.write(p, arcname=str(arc))
            zipped_count += 1
            if not cfg.quiet and zipped_count % 1000 == 0:
                print(f"[zip] {zipped_count}/{total_to_zip}", file=sys.stderr, flush=True)

    report.update({'staging': None, 'zip': str(out_zip), 'dryRun': False})

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
                    'excludedCounts': report['excludedCounts'],
                    'incomingByExt': report.get('incomingByExt', {})
                }
                if not m.get('finishedAt'):
                    m['finishedAt'] = now
                # Ensure status reflects completion
                m['status'] = 'finished'
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
    ap.add_argument('--quiet', action='store_true', help='Suppress progress output')
    ap.add_argument('--strict-companions', action='store_true')
    ap.add_argument('--check-companions', action='store_true')
    ap.add_argument('--compression', default='deflated', choices=['stored','deflated','bzip2','lzma'])
    ap.add_argument('--compress-level', type=int)
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
        quiet=args.quiet,
        strict_companions=args.strict_companions,
        check_companions=args.check_companions,
        compression=args.compression,
        compress_level=args.compress_level,
    )


def main() -> None:
    cfg = parse_args()
    try:
        report = prezip_stage(cfg)
    except Exception as e:
        print(json.dumps({'status': 'error', 'message': str(e)}))
        raise SystemExit(1)
    # Post-success hint for uploads (stderr so JSON stdout stays clean)
    try:
        if not report.get('dryRun') and report.get('zip'):
            print(
                f"[next] Upload tip (Google Drive via rclone): rclone copy \"{cfg.output_zip}\" gdrive:Deliveries/<ProjectName> --drive-chunk-size 256M --progress",
                file=sys.stderr,
                flush=True,
            )
    except Exception:
        pass
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()


