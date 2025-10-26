# Automation Reduction Experiments
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26


## Goal
Reduce total files requiring manual review/cropping by automatically selecting one best image per stage-run (from stage 1 → 1.5 → 2 → 3 variants) and separating images into needs-crop vs no-crop with high accuracy, operating strictly inside a sandbox.

## Constraints
- Sandbox-only: no moves/deletes outside sandbox.
- Companion-safety: all operations use shared companion utilities.
- Reversible: dry-run first; commits gated.

## Pipelines to Trial

1) Character processor first, then auto-pick winners
- Group by stage/time using `character_processor`.
- For each filename stem-run, keep one winner using priority: 3 > 2 > 1.5 > 1.
- Tie-breakers (in order): file size, sharpness (Laplacian variance), brightness/contrast sanity.
- Mark non-winners for delete (sandbox trash). Dry-run first.

2) Dedupe-first, then character processor and winners
- Run duplicate finder to eliminate near-identicals early.
- Then run pipeline (1). Compare volume reduction and accuracy vs (1).

3) Similarity clusters for crop-need triage
- Build quick embeddings/similarity for within-run alternatives.
- If winner is from stage 3 and passes framing heuristics (see below), auto-mark no-crop.
- Else flag as needs-crop.

## Crop-Need Heuristic (prototype)
- Aspect ratio and face framing windows (centered face/top rule of thirds where applicable).
- Edge cutoff detection (salient content near borders) via simple Sobel/Laplacian edge density in border bands.
- Sharpness threshold (reject too-blurry as needs-crop).
- Over/under exposure quick check (mean + std bands).

Notes: Heuristic is conservative; prefer false negatives (send to crop) over false positives.

## Experiment Matrix (order, variants, settings)

Run in this order; stop early if success criteria are met with lower risk:

1) A-conservative
- Variant: CharacterProcessor → WinnerSelect (3>2>1.5>1) → CropNeed (conservative)
- Settings: stage-strict; sharpness p95 cutoff; border-edge low tolerance; exposure ±2σ

2) B-conservative
- Variant: DedupeFirst → A-conservative
- Settings: dedupe threshold high (few merges), exact stem neighbors only

3) A-balanced
- Variant: same as (1) but thresholds relaxed
- Settings: sharpness p85; border-edge medium; exposure ±2.5σ

4) B-balanced
- Variant: DedupeFirst → A-balanced
- Settings: dedupe threshold medium

5) C-balanced (similarity assist)
- Variant: Similarity within-run to validate winner; if conflict, prefer higher-stage unless quality gap > X%
- Settings: cosine ≥ 0.8 for near-duplicate; size/blur gap ≥ 15%

6) A-aggressive (pilot on small subset only)
- Variant: like A, but auto no-crop for stage3 passing framing; tighter runtime goals
- Settings: border-edge very low; sharpness p75

Settings grid to sweep (where relevant):
- Stage priority: strict vs lenient (allow 2 > 3 if 3 is blurry by >25%)
- [Deprecated for AI images] Sharpness/Border-edge/Exposure thresholds (see Excluded metrics below)
- Dedupe similarity: 0.90, 0.85, 0.80
- Group time window (if used): 2 min, 5 min, 10 min

## Metrics to Log (per variant)
- Input file count, groups detected, winners kept, percent reduction.
- Needs-crop vs no-crop counts; heuristic thresholds used.
- Time to run (wall-clock), errors.
- Sample-based accuracy: manual spot-check on N random groups (TP/FP/FN rates).

### Excluded metrics for AI-generated images (decision-making)
- Sharpness (e.g., LapVar/Tenengrad): Not used for pass/fail; AI prompts may intend blur.
- Border-edge density: Not used as a negative signal; composition near borders is common/intentional.
- Exposure (mean/std bands): Not used as a negative signal; exposure deviations are often prompt-driven.
Note: These may be collected as optional telemetry for offline analysis, but are not decision features.

Sampling/evaluation protocol:
- Stratified sample of N=200 winner decisions across characters/stages.
- Annotate FP (bad keeps), FN (missed better alternative), and crop-need FP/FN.
- Wilson CI for error rates; flag variants with overlap to choose simpler option.

## Procedure
- Prepare sandbox copy.
- Run Variant A (char-processor → winners → crop-need heuristic) [dry-run → commit].
- Run Variant B (dedupe-first → Variant A) [dry-run → commit].
- Optionally run similarity-assisted triage.
- Record metrics into `data/projects/<project>_automation_metrics.jsonl`.

## Success Criteria (initial)
- ≥60% immediate reduction via winners selection with ≤2% false-positive discards (on sampled review).
- ≥20% of remaining confidently marked no-crop with ≤5% false positives.
- End-to-end speedup measured in hours saved per 10k images.

## Safety & Logging
- All deletes to sandbox-local Trash via `safe_delete_image_and_yaml`.
- All moves via `move_file_with_all_companions`.
- FileTracker logs + JSON metrics for reproducibility.
 - Experiments logging policy: For harness dry-runs, do not record to FileTracker or any tracking programs; write sandbox-local artifacts only.

### Ground rules addendum (clarifications)
- Sandbox-only: All reads/writes under `sandbox/mojo2/`. No writes outside.
- Tracking disabled: Harness dry-runs never write to FileTracker or global logs. Only sandbox-local `metrics/`, `reports/`, `runs/`, `logs/`.
- Modes: Default `dry-run`. “Commit” requires explicit written approval here or in terminal comments.
- Commit gates: Winners ≥60% reduction with ≤2% FP; auto no-crop ≥20% with ≤5% FP (sampled).
- Metrics: reduction = 1 − winners/input. FP (winners) = true-best wrongly discarded. FP (no-crop) = marked no-crop but needs crop.
- Sampling: Stratified N=200, fixed seed (1337) for reproducibility; record run-id and seed.
- Hygiene: Kill any servers/processes after runs.
- Safety: Never modify files outside sandbox; never alter ZIP contents.

### Watchdog/Heartbeat quick usage (status + safety)
- New flags (runner `scripts/tools/reducer.py`):
  - `--max-runtime <sec>`: hard wall timeout (abort on exceed)
  - `--watchdog-threshold <sec>`: no-progress stall threshold (abort on exceed)
  - `--progress-interval <sec>`: prints progress to terminal at this cadence
  - `--no-stack-dump`: disable stack dump on abort (json report still written)
  - `--simulate-hang`: test hook to verify watchdog aborts
- On abort: prints `ABORT <run-id> reason=<...>`; writes sandbox-only logs:
  - `sandbox/mojo2/logs/error_<run-id>.json`
  - `sandbox/mojo2/logs/stack_<run-id>.txt` (unless `--no-stack-dump`)

Examples (terminal):
- Simulated hang test (expect abort in ~2–5s):
  - `python3 scripts/tools/reducer.py run --variant A --profile conservative --dry-run --sandbox-root sandbox/mojo2 --simulate-hang --max-runtime 5 --watchdog-threshold 2 --progress-interval 0.5 --no-stack-dump`
- Normal dry-run with visible progress (no real moves):
  - `python3 scripts/tools/reducer.py run --variant A --profile conservative --dry-run --sandbox-root sandbox/mojo2 --max-runtime 120 --watchdog-threshold 60 --progress-interval 2`

## AI Training Integration (when/how)
- Now: collect sandbox-local feature/label rows for two tasks:
  - Winner selection: features (stage, sharpness, size, exposure, similarity); label = chosen/not.
  - Crop-need: features (AR, border-edge, sharpness, exposure, face-box if available); label = needs_crop.
- Storage: write to sandbox-local `metrics/` only; do not write to global logs (sandbox mode on).
- Later: train lightweight classifiers (logistic/XGBoost) after we have ≥5–10k labeled examples; compare vs heuristics.
- Gate: deploy model only if it beats conservative heuristic with ≤2% FP on winners and ≤5% FP on no-crop.

## Output Artifacts
- Per-variant JSONL metrics, settings snapshot, and sample evaluation CSV in sandbox `metrics/`.
- A consolidated markdown summary with reduction, accuracy, runtime, and recommendation.

## Results & Diary

Variant A (timestamp exact groups → winner by stage)
- Expectation: moderate reduction if stage variants share timestamps (30–50%).
- Run: sandbox/mojo2, dry-run; exact timestamp grouping.
- Results:
  - Total PNGs: 17,935
  - Groups: 17,707; Images in groups: 17,934; Winners: 17,707
  - Reduction: 1.27%
  - Winner stages: 1.0=5,951, 1.5=5,391, 2.0=5,983, 3.0=382
- Notes: Variants rarely share exact timestamps → tiny reduction. Next: list-sort stage-run grouping.

Variant B (list-sort stage runs → winner by stage, conservative)
- Expectation: high reduction by collapsing runs (≥50%).
- Run: sandbox/mojo2, dry-run; sort by timestamp+stage, group consecutive runs.
- Results:
  - Total PNGs: 17,935
  - Stage-run groups: 5,985; Images in groups: 17,735; Winners: 5,985
  - Reduction: 66.25%
  - Winner stages: 1.5=5, 2.0=5,598, 3.0=382
- Notes: Matches target; many runs pick stage 2; sample-check next to validate.

Composition Triage (border-edge heuristic)
- Full run (bf=2.5%, edge≥0.35), winners=5,985:
  - Flagged: 5 (stage2=4, stage3=1), Skipped: 0
- Sample grid sweep (n=1,000 winners):
  - bf=1.0% → edge≥0.30/0.40/0.50: flagged 0, skipped 0
  - bf=2.5% → edge≥0.30: flagged 2; edge≥0.40/0.50: flagged 0; skipped 0
  - bf=5.0% → edge≥0.30: flagged 59; edge≥0.40: flagged 11; edge≥0.50: flagged 2; skipped 0
- Notes: Conservative settings (≤2.5% bands, ≥0.40 threshold) barely flag; 5.0% @ 0.30 begins to catch more edge-cutoff cases.

Harness dry-run scaffolds (2025-10-07 UTC)
- Variant A (dry-run harness initialization)
  - run-id: 20251007T220336Z_A-conservative
  - artifacts:
    - sandbox/mojo2/metrics/automation_metrics.jsonl (appended placeholder line)
    - sandbox/mojo2/metrics/samples_20251007T220336Z_A-conservative.csv
    - sandbox/mojo2/reports/summary_20251007T220336Z_A-conservative.md
    - sandbox/mojo2/logs/filetracker_20251007T220336Z_A-conservative.log
    - sandbox/mojo2/runs/20251007T220336Z_A-conservative/manifest.json
  - note: Same variant name as earlier; this was a CLI harness dry-run only (no selection/triage executed). Metrics are placeholders; no sandbox moves/deletes.

- Variant B (dry-run harness initialization)
  - run-id: 20251007T220502Z_B-conservative
  - artifacts:
    - sandbox/mojo2/metrics/samples_20251007T220502Z_B-conservative.csv
    - sandbox/mojo2/reports/summary_20251007T220502Z_B-conservative.md
    - sandbox/mojo2/logs/filetracker_20251007T220502Z_B-conservative.log
    - sandbox/mojo2/runs/20251007T220502Z_B-conservative/manifest.json
  - note: Same variant label as prior experiments; dry-run harness only; results differ from earlier real runs by design (placeholders).

- Variant C (balanced, dry-run harness initialization)
  - run-id: 20251007T220540Z_C-balanced
  - artifacts:
    - sandbox/mojo2/metrics/samples_20251007T220540Z_C-balanced.csv
    - sandbox/mojo2/reports/summary_20251007T220540Z_C-balanced.md
    - sandbox/mojo2/logs/filetracker_20251007T220540Z_C-balanced.log
    - sandbox/mojo2/runs/20251007T220540Z_C-balanced/manifest.json
  - note: New variant in diary; this run established the reporting/logging scaffold only. No winner/crop logic executed.

Watchdog validation (2025-10-08 UTC)
- Variant A (conservative) — run-id: 20251008T191244Z_A-conservative
  - Expectation: visible terminal progress (files/groups/items), no stalls, sandbox-only artifacts, tracking off.
  - Command: `python3 scripts/tools/reducer.py run --variant A --profile conservative --dry-run --sandbox-root sandbox/mojo2 --max-runtime 300 --watchdog-threshold 60 --progress-interval 2`
  - Result: progress printed; no abort; artifacts written under sandbox/mojo2 (metrics/report/run manifest). Stats still N/A (analysis not wired yet).
  - Notes → Next: proceed to B-conservative watchdog run; then wire real analysis for stats.

- Variant B (conservative) — run-id: 20251008T191624Z_B-conservative
  - Expectation: same as A; ensure watchdog/progress hold under B label.
  - Command: `python3 scripts/tools/reducer.py run --variant B --profile conservative --dry-run --sandbox-root sandbox/mojo2 --max-runtime 300 --watchdog-threshold 60 --progress-interval 2`
  - Result: progress printed; no abort; artifacts written under sandbox/mojo2. Stats N/A (analysis not wired yet).
  - Notes → Next: C-balanced watchdog run; then implement real analysis to produce reduction/FP stats before further sweeps.

Stage selection run (2025-10-08 UTC, full sandbox move, one-winner-per-stage group)
- Variant B (conservative) — run-id: 20251008T235939Z_B-conservative
  - Settings: centralized grouping via `sort_image_files_by_timestamp_and_stage` + `find_consecutive_stage_groups`; winner rule = highest stage (3>2>1.5>1); companions preserved; watchdog on; commit on; sandbox-only
  - Command: `python3 scripts/tools/reducer.py run --variant B --profile conservative --sandbox-root sandbox/mojo2 --commit --max-runtime 3600 --watchdog-threshold 120 --progress-interval 2`
  - Result counts:
    - SELECTED_PNG: 5,985 (expected = total stage groups)
    - DELETE_PNG: 11,750
    - CROP_PNG: 0
    - Residual in `sandbox/mojo2`: 200 PNGs (likely tail of final window; sweep next run)
  - Notes: Terminal progress remained active; no explicit end-of-run error was emitted. Outcome matches goal: one winner per stage group routed to `selected/`, remainder to `delete/`. A short sweep will clear the final 200 PNGs.

<!-- Daily journal content moved to AI Journal; removing from this experiments doc to keep scope clean. -->

### Abbreviated results (side-by-side)

| Variant | Profile | Run-id | Reduction | Winners | Notes |
|---|---|---|---:|---:|---|
| A | conservative | earlier run | 1.27% | 17,707 | Timestamp-exact grouping; minimal overlap across stages |
| B | conservative | earlier run | 66.25% | 5,985 | List-sort stage-run grouping; stage-priority 3>2>1.5>1 |
| A | conservative | 20251007T220336Z_A-conservative | N/A | N/A | Harness scaffold only (dry-run); no selection/triage executed |
| B | conservative | 20251007T220502Z_B-conservative | N/A | N/A | Harness scaffold only (dry-run); no selection/triage executed |
| C | balanced | 20251007T220540Z_C-balanced | N/A | N/A | Harness scaffold only (dry-run); similarity-assisted variant placeholder |

#### Quick chart: Reduction vs FP targets

```
Target (winners FP discards): ≤2%

Variant  Reduction (20 = 100%)             FP est
A (earlier)  [....................]  1.27%  TBD
B (earlier)  [#############.......] 66.25%  TBD
C (dry-run)  [....................]   N/A   TBD
```

Planned Variations (10)
1) B-conservative + dedupe-first (similarity ≥0.90)
2) B-balanced (sharpen threshold p90; exposure ±2.5σ)
3) B + dedupe 0.85
4) B + prefer 2 over weak 3 (blur gap ≥25%)
5) C-balanced (similarity validation of winner; cosine ≥0.8)
6) A-aggressive on subset (auto no-crop for stage3 passing framing)
7) Group time window sensitivity (2/5/10 min) before list-sort
8) Stage priority lenient (allow 1.5 over weak 2 with big quality gap)
9) Crop-need heuristic thresholds sweep (border 1%/2.5%/5%)
10) Dedupe-first only vs B (to isolate dedupe impact)

## Future Experiments & Expected Learnings

- Winner selection validation (B-conservative vs B-balanced)
  - Expectation: Balanced sharpness cutoff (p85/p90) may increase 3.0 winners if 3 is only slightly blurrier; measure change in reduction and sampled FP discards.
- Dedupe sensitivity (0.90 vs 0.85 vs 0.80)
  - Expectation: Lower thresholds merge more near-dupes; quantify added reduction vs risk of merging non-dupes (manual spot-check sample).
- Stage priority leniency (allow 2 > weak 3 when blur gap ≥25%)
  - Expectation: Slight increase in perceived quality of winners; minimal impact on reduction; validate via sample FN/FP on winners.
- Crop-need band sweep (border 1%/2.5%/5% with edge density 0.30/0.40)
  - Expectation: 5% bands at 0.30 catch more edge-cutoff; tune to keep FP ≤5% on auto no-crop.
- Similarity-assisted C-balanced (cosine ≥0.8 validation)
  - Expectation: Reduce winner FPs by flagging conflicts; trade small runtime for improved no-regret keeps.
- Group window sensitivity (2m/5m/10m)
  - Expectation: Larger window increases grouping and reduction; ensure runs don’t conflate distinct prompts.

- Face-box–aware framing for auto no-crop (when face boxes available)
  - Expectation: Safely expand auto no-crop share with ≤5% FP by enforcing centered/rule-of-thirds face constraints.
- Exposure normalization before sharpness scoring
  - Expectation: Stabilize sharpness ranking across under/over-exposed images; reduce misranking FPs.
- Entropy/texture fallback for blur edge cases
  - Expectation: Catch low-frequency blur that LapVar misses; compare Tenengrad vs entropy cutoff.
- Confidence thresholding with abstain
  - Expectation: Use calibrated scores + Wilson CI to abstain when uncertain; minimize FPs with small hit to reduction.


### Review Queue + Sidecar Marks (rejected for production; directory-driven is source of truth)

- Decision: Do not introduce a sidecar-backed queue for cropping. The current directory-driven pipeline is the source of truth and is more robust to out-of-band edits.
- Rationale:
  - Web image selector already routes files to `selected/` vs `crop/` (acts as the queue).
  - `character_processor` and `character_sorter` operate on directories; they automatically reflect any manual file moves/deletes done outside tools.
  - `multi_crop_tool` scans `crop/` and moves processed items with companions to `*_cropped/`; no list drift risk.
  - A sidecar list could desync if files are renamed/moved externally; directory scans won’t.
- If we need acceleration later, prefer directory-based helpers (e.g., glob filters or tag-to-subdir moves) that keep the directory as the single source of truth.
- Optional (non-SOT use): a sidecar can be used for analytics/telemetry only, never to drive processing order or inclusion.

## Next Steps
- Implement stage-run winner selector (dry-run first).
- Implement crop-need heuristic scorer.
- Wire character_processor grouping and sandbox guards.
- Run both variants on the sandbox and evaluate.
