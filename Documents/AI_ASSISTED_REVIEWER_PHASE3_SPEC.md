AI-Assisted Reviewer — Phase 3 (Two-Action Crop Flow)

Status: Proposed (ready to implement incrementally)
Audience: AI
Tags: [ai, reviewer, cropping, phase3]

PLAN
1) Keep current fast path (approve + aiCropAccepted) routing to `crop_auto/` for later tweaks.
2) Introduce two distinct actions:
   - approve_ai_suggestion: Accept AI’s image; crop will be adjusted later → route to `crop_auto/` with `.decision` including `ai_crop_coords` and `ai_route: "suggestion"`.
   - approve_ai_crop (future): AI’s crop is perfect → auto-apply crop (via a safe headless mode) and move to `selected/`.
3) UX mapping (now):
   - 1–4 with overlay ON → approve_ai_suggestion (today’s behavior; no auto-crop).
   - 1–4 with overlay OFF → approve (no crop).
4) UX mapping (future):
   - Shift+1–4 (or explicit button) → approve_ai_crop (auto-crop), after safety review.
5) Safety gates: no auto-crop until we add a dedicated, headless crop path guarded by file safety rules.

SCHEMA
- Current DB (v3) constraint: CHECK(user_action IN ('approve','crop','reject')).
- Today (no schema change):
  - Log `user_action='approve'` for approve_ai_suggestion.
  - Sidecar in `crop_auto/`: add `ai_route: 'suggestion'` for intent and `ai_crop_coords`.
- Future (Phase 3B) migration:
  - Extend enum: `user_action IN ('approve','crop','reject','approve_ai_suggestion','approve_ai_crop')`.
  - Add view to detect “perfect crop”: compare final_crop to ai_crop with 5% tolerance.

API SPEC
- Client payload (unchanged + flag):
  - `{ groupId, selectedImage, crop: boolean, aiCropAccepted: boolean }`
- Server routing (current):
  - If `crop==true` → manual crop → `crop/`.
  - Else if `aiCropAccepted==true` and `ai_crop_coords` exist → approve_ai_suggestion → `crop_auto/` with `.decision` { ai_crop_coords, ai_route: 'suggestion' }.
  - Else → approve → `selected/`.
- Future approve_ai_crop:
  - New hotkey (Shift+1–4) or button.
  - Server calls safe headless crop path, then moves to `selected/` upon success; logs accordingly.

DATA HANDLING
- Today:
  - DB: `user_action='approve'` for approve_ai_suggestion; route captured by sidecar (`ai_route='suggestion'`).
  - Desktop AI crop tool (`04_ai_desktop_multi_crop.py`) preloads AI rectangles from sidecars; final crop is written and DB updated by the crop tool as usual.
- Future:
  - With enum extension, analytics can filter/user_action directly without relying on sidecars.
  - Add view `ai_perfect_crops` to surface matches between AI and final crops.

NOTES
- Keeps current speed-first reviewer flow intact.
- Adds explicit intent signal without breaking the v3 DB constraint.
- Auto-crop path is deferred until safety-approved.


