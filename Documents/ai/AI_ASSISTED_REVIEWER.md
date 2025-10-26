# AI-Assisted Reviewer

**Last Updated:** 2025-10-26
**Status:** Active
**Audience:** Developers, Operators
**Estimated Reading Time:** 12 minutes

## Feature Overview
A fast web-based reviewer for image groups that performs selection, optional crop staging, and file routing with full audit logging. Future versions integrate trained models.

## Usage
```bash
source .venv311/bin/activate
python scripts/01_ai_assisted_reviewer.py <path-to-image-groups>
```

## Implementation Details
- Grouping & stage detection reused from existing utilities.
- Decisions logged: selection-only now; crop logging when crop is performed.
- File operations go through companion-aware moves and FileTracker logging.

### File Routing Specification
| Action | Crop Proposal | Routing |
|---|---|---|
| Approve | none | winner → `__selected/`; others → `__delete_staging/` |
| Approve | proposed | perform crop → `__selected/`; others → `__delete_staging/` |
| Override | any | chosen → `__selected/`; others → `__delete_staging/` |
| Manual Crop | any | chosen → `__crop/`; others → `__delete_staging/` |
| Reject | any | all → `__delete_staging/` |
| Skip | any | no moves |

### Directories
```
project/
  __selected/
  __crop/
  __delete_staging/
  data/training/
  data/file_operations_logs/
```

### Training Data Logging
- Selection: `log_selection_only_entry()` → `data/training/selection_only_log.csv`
- Crop: `log_select_crop_entry()` → `data/training/select_crop_log.csv`

### Keyboard Shortcuts
- A Approve, 1–4 Override, C Manual Crop, R Reject, S Skip, ↑/↓/Enter Navigate

## Known Limitations
- Current tool is rule-based; integrate `train_ranker_v3.py` and crop proposer when ready.
- Delete staging uses fast move; Trash step is manual.

## Related Documents
- Phase 3 spec: `Documents/AI_ASSISTED_REVIEWER_PHASE3_SPEC.md`
- File routing spec: `Documents/AI_ASSISTED_REVIEWER_FILE_ROUTING_SPEC.md`
- Training Guide: `Documents/ai/AI_TRAINING_GUIDE.md`
