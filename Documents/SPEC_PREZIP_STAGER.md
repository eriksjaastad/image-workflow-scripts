# Spec: prezip_stager

Goal: Safely produce the client zip by copying only eligible deliverables into a temporary staging directory, validating invariants, and zipping the staging tree.

---

## CLI
```
python scripts/tools/prezip_stager.py \
  --project-id mojo1 \
  --content-dir /abs/path/to/mojo1 \
  --output-zip /abs/path/to/out/mojo1_final.zip \
  [--allow-unknown] [--commit]
```

- Default is dry-run (no writes). `--commit` performs copy + zip.
- Reads: `data/projects/<project_id>_allowed_ext.json` + global bans.

## Logic
1) Load allowlist (allowed extensions + overrides) and global banned types/patterns.
2) Scan `--content-dir` recursively; for each file:
   - Lowercase extension; skip directories.
   - If hidden (dotfile) → exclude + report.
   - If extension in banned set or matches banned patterns → exclude + report.
   - If extension not in allowed ∪ overrides → exclude + report.
   - Else include (eligible).
3) Companion integrity (optional): if client initially had companions, verify same‑stem companions included/excluded consistently.
4) Build differences report (stdout + JSON alongside zip path).
5) If `--commit`: copy eligible files to staging dir (mirroring structure) and create zip from staging.

## Outputs
- Dry-run summary: totals by extension, excluded counts by reason.
- JSON report: `output-zip + .report.json` with file lists and reasons.
- On commit: `*.zip` + delete staging on success.

## Invariants
- No hidden files in staging.
- No global-banned types/patterns.
- Optional: forbid empty directories in staging.

## Failure modes
- Fail closed if any disallowed/banned types encountered (unless `--allow-unknown`).
- Nonexistent allowlist file → instruct to run inventory first.

## Integration hooks
- Project manifest: set `finishedAt` after successful zip; write counts under `metrics`.
- Dashboard: emit a compact summary (eligible totals, excluded by reason) for audit.

## Future
- Per-client policy presets; signed artifact manifest; checksum verification.
