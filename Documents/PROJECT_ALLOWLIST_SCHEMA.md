# Per-Project Allowed Extensions Schema

Purpose: Capture the exact set of file extensions present in the original `content/` at project start to form an allowlist for pre-zip deliverables.

---

## Storage location
- File: `data/projects/<project_id>_allowed_ext.json`
- Created: at project start (or first inventory of `content/`).

## JSON schema (logical)
```json
{
  "projectId": "string",
  "snapshotAt": "ISO-8601 UTC",
  "sourcePath": "absolute-or-relative path to content/ at inventory time",
  "allowedExtensions": ["lowercase extensions without dot"],
  "clientWhitelistOverrides": ["lowercase extensions without dot"],
  "notes": "optional"
}
```

### Field rules
- `allowedExtensions`: derived strictly from scanning the original `content/`. Lowercase, no leading dot, unique, sorted.
- `clientWhitelistOverrides`: optional explicit list for types client requests that were not present at start.

### Extraction rules
- Consider only regular files under `content/` (skip directories and dotfiles).
- Normalize extension by splitting on the last dot, lowercasing, and stripping the dot.
- If a file has no extension, exclude it by default (log separately).

### Safety
- Immutable once written (append-only mindset). If re-inventory is needed, write a new file with an incremented suffix or update `snapshotAt` with a retained prior copy.

---

## Creation flow
1) Inventory `content/` recursively and collect unique extensions.
2) Sort and write `data/projects/<project_id>_allowed_ext.json` with timestamp.
3) Optionally print a summary by extension with counts for audit.

---

## Usage in pre-zip stager
- Eligible extension set = `allowedExtensions ∪ clientWhitelistOverrides`.
- Still must pass global bans and filename pattern checks.

---

## Example
```json
{
  "projectId": "mojo1",
  "snapshotAt": "2025-10-06T16:20:00Z",
  "sourcePath": "../../mojo1",
  "allowedExtensions": ["png", "yaml"],
  "clientWhitelistOverrides": [],
  "notes": "Initial extensions from original content"
}
```
