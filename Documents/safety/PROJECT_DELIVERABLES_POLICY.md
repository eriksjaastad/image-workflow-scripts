# Deliverables Policy (Allowed vs Banned Types)
**Status:** Active
**Audience:** Developers, Operators

**Last Updated:** 2025-10-26


Purpose: Ensure only client-eligible files are included in the final zip while preventing any internal files (manifests, logs, markers) from shipping. Policy is per-project and enforced by a pre-zip stager.

---

## Core principles
- Inventory first: record which file extensions exist in the original `content/` at project start; this becomes the per-project allowed set.
- Least privilege: include only files whose extensions are in the allowed set AND not in the banned set.
- Internal files never ship: any types we introduce (manifests, logs, CSV/JSON, `.cropped`, etc.) are globally banned by default.
- Explicit overrides: if a client needs a type not present at start, add a per-project whitelist override.

---

## Per-project allowed extensions snapshot
- Created once at project start (or first scan of `content/`).
- Stored at: `data/projects/<project_id>_allowed_ext.json`
- Example:
```json
{
  "projectId": "mojo1",
  "snapshotAt": "2025-10-06T16:15:00Z",
  "allowedExtensions": ["png", "yaml"],
  "clientWhitelistOverrides": [],
  "notes": "Initial inventory of content/ extensions"
}
```
Notes:
- Extensions are lowercase, no leading dot.
- If the client’s source did not include `.yaml`/`.caption`, we exclude any we produced later unless explicitly whitelisted.

---

## Globally banned internal types
- Project/control/docs/logs: `*.project.json`, `*.project.yml`, `*.manifest.json`, `*.md`, `*.log`, `*.csv`, `*.json` (non-client data), `*.sqlite`, `*.db`, `*.lock`
- Markers/temp: `*.cropped`, temporary files, dotfiles
- Training/metrics: anything under `data/`, `scripts/`, or generated reports unless explicitly whitelisted
- Banned filename patterns:
  - `.*\.project\.(json|yml)$`
  - `.*\.cropped$`
  - `^\.` (no dotfiles)

---

## Pre-zip inclusion rule
Eligible if and only if:
1) extension ∈ `allowedExtensions ∪ clientWhitelistOverrides`, and
2) extension ∉ global banned set, and
3) filename does not match banned patterns.

Companions: If the client’s source included companions (e.g., `.yaml`, `.caption`), include same‑stem companions that pass the rule. If not present initially, exclude by default.

---

## Validation & reporting
- Differences report: list excluded files and reasons (not allowed, banned, hidden).
- Summary: totals by extension; any disallowed extensions encountered.
- Copy-only: stager never modifies `content/`.

---

## Operational flow
1) Project start → write `data/projects/<id>_allowed_ext.json` (inventory of `content/`).
2) Normal work proceeds.
3) Pre-zip → read allowlist + bans → scan final `content/` → copy eligible files to staging → build zip from staging.

---

## Safety
- Default dry-run; require `--commit` to write staging/zip.
- Fail-closed: error if banned/unknown types unless `--allow-unknown` is passed.
- Keep `data/` out of deliverables by design.
