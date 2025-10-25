# Baseline Templates & README

This doc provides CSV templates and concise instructions for building historical throughput baselines from archives and timesheets.

---

## CSV A: Archives Ledger (archives_ledger.csv)
Columns:
```
archive_name,project_slug,date_archived,image_count
```
Notes:
- `archive_name`: filename of the ZIP/tar you created
- `project_slug`: e.g., mojo1
- `date_archived`: ISO-8601 date (UTC if known)
- `image_count`: number of images in the archive (extensions you care about)

---

## CSV B: Timesheet Ledger (timesheet_ledger.csv)
Columns:
```
date,project_slug,hours,notes
```
Notes:
- `date`: ISO-8601 date
- `project_slug`: matches archives ledger
- `hours`: decimal hours spent that day on that project
- `notes`: optional tags, e.g., select|crop|cleanup (pipe-separated)

---

## Reconstruct Daily Throughput (CSV C)
Output columns (produced by join script):
```
date,project_slug,hours,images_estimated,images_per_hour
```
Rules:
- If an archive spans multiple prior workdays, spread `image_count` backward across those days pro‑rata by `hours`.
- Example: ZIP has 1,800 images; prior 3 workdays total 9 hours ⇒ 200 iph ⇒ 600 per day.

---

## Workflow
1) Fill CSV A and CSV B for your historical projects.
2) Run the join script (future): `scripts/tools/join_throughput_baseline.py` → CSV C.
3) Compute baseline (median `images_per_hour`) and p25/p75 for ahead/behind bands.
4) Optionally tag phases via `notes` and compute per-phase medians.

---

## What to count as images
- Use a fixed extension list per project (png, jpg, jpeg, webp, heic, tif, tiff, …). Confirm before counting.

---

## Example Rows
CSV A:
```
mojo1_2025-10-03.zip,mojo1,2025-10-03,1250
mojo1_2025-10-05.zip,mojo1,2025-10-05,950
```
CSV B:
```
2025-10-01,mojo1,3.5,select|cleanup
2025-10-02,mojo1,2.0,crop
2025-10-03,mojo1,3.5,select
2025-10-04,mojo1,0.8,cleanup
2025-10-05,mojo1,3.2,select
```
