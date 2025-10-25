# AI Training Decisions v3 - Complete Implementation Plan
**Date:** October 21, 2025  
**Status:** Implemented (migrated on 2025-10-24)  
**Goal:** Link AI recommendations with user corrections for superior training data

---

## 🎯 Executive Summary

### The Problem
Current training data logs crops in isolation, without context:
- ❌ We don't know what images were in the group
- ❌ We don't know what AI recommended
- ❌ We don't know if user agreed or corrected
- ❌ We can't track AI's improvement over time

### The Solution
**Group-based decision logging** that tracks the complete decision chain:
- ✅ All images in the group (context)
- ✅ AI's recommendation (what it thought)
- ✅ User's selection (what was correct)
- ✅ AI's crop proposal (where it thought to crop)
- ✅ User's final crop (where to actually crop)
- ✅ Match flags (was AI right?)

### Key Innovation
**Two-stage logging with group_id linkage:**
1. **AI Reviewer:** Logs AI recommendation + user selection → creates `group_id`
2. **Desktop Multi-Crop:** Looks up `group_id`, adds final crop coordinates
3. **Result:** Complete before/after training data!

---

## 📊 Schema Design v3 - SQLite Database

### ✅ **Why SQLite Instead of CSV?**

**Benefits:**
- ✅ **Data Integrity:** ACID compliant, no corruption from concurrent access
- ✅ **Built-in Validation:** Constraints prevent invalid data at write time
- ✅ **Fast Lookups:** Indexed queries (find by group_id instantly)
- ✅ **JSON Support:** Flexible arrays for images and coordinates
- ✅ **Zero Installation:** Built into Python, works on all platforms
- ✅ **File-Based:** One `.db` file per project (easy to backup/archive)
- ✅ **Queryable:** SQL queries for analysis and debugging

### File: `data/training/ai_training_decisions/{project_id}.db`

**Table: `ai_decisions`**

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | TEXT PRIMARY KEY | Unique identifier: `{project}_group_{timestamp}_batch{N}_img{N}` |
| `timestamp` | TEXT NOT NULL | ISO 8601 UTC when decision made |
| `project_id` | TEXT NOT NULL | Project name (e.g., "mojo3") |
| `directory` | TEXT | Source directory path |
| `batch_number` | INTEGER | Batch number within project |
| `images` | TEXT NOT NULL | JSON array: `["img1.png", "img2.png", ...]` |
| `ai_selected_index` | INTEGER | Which image AI picked (0-3) |
| `ai_crop_coords` | TEXT | JSON array: `[x1, y1, x2, y2]` normalized [0,1] |
| `ai_confidence` | REAL | Model confidence (0.0-1.0) |
| `user_selected_index` | INTEGER NOT NULL | Which image user picked (0-3) |
| `user_action` | TEXT NOT NULL | `'approve'` \| `'crop'` \| `'reject'` |
| `ai_crop_accepted` | BOOLEAN | TRUE if user explicitly accepted AI crop suggestion; FALSE if declined; NULL if not applicable |
| `final_crop_coords` | TEXT | JSON array: `[x1, y1, x2, y2]` (filled later) |
| `crop_timestamp` | TEXT | ISO 8601 UTC when crop completed |
| `image_width` | INTEGER NOT NULL | Original image width in pixels |
| `image_height` | INTEGER NOT NULL | Original image height in pixels |
| `selection_match` | BOOLEAN | TRUE if AI picked same image as user |
| `crop_match` | BOOLEAN | TRUE if AI crop within tolerance of user crop |

**Constraints:**
```sql
CHECK(user_action IN ('approve', 'crop', 'reject'))
CHECK(ai_selected_index >= 0 AND ai_selected_index <= 3)
CHECK(user_selected_index >= 0 AND user_selected_index <= 3)
CHECK(ai_confidence IS NULL OR (ai_confidence >= 0.0 AND ai_confidence <= 1.0))
CHECK(image_width > 0 AND image_height > 0)
```

**Indexes:**
- `idx_project_id` - Fast queries by project
- `idx_timestamp` - Chronological queries
- `idx_selection_match` - Filter by AI correctness
- `idx_crop_match` - Filter by crop quality
- `idx_user_action` - Filter by action type
- `idx_batch_number` - Batch-based queries

**Views:**
- `ai_performance` - Aggregated accuracy stats per project
- `incomplete_crops` - Images marked for crop but not yet done
- `ai_mistakes` - Decisions where AI was wrong (for training)

---

## 🔄 How It Works in Practice
Update 2025-10-24
- Reviewer now logs `ai_crop_accepted` when user approves AI-picked image with crop overlay ON.
- Databases migrated to add `ai_crop_accepted`. Optional backfill applied to rows meeting strict approval criteria.
- Reviewer loads `crop_proposer_v3.pt`; crop overlays sized correctly (no CLIP L2-normalization at inference).

### Phase 1: AI Reviewer (Image Selection)

**User Flow:**
1. Load batch of groups (e.g., 100 groups)
2. For each group:
   - AI analyzes all images → picks best + suggests crop
   - User sees AI's choice highlighted
   - User presses hotkey (1-4 for approve, Q-R for manual crop)
3. User clicks "Finalize selections"
4. **LOGGING HAPPENS HERE**

**Data Written:**
```python
{
    'group_id': 'mojo3_group_20251021_234530',
    'timestamp': '2025-10-21T23:45:30Z',
    'project_id': 'mojo3',
    'directory': 'mojo3/faces',
    'batch_number': 1,
    
    'image_1_filename': '20250705_220230_stage3_enhanced.png',
    'image_2_filename': '20250705_220604_stage3_enhanced.png',
    'image_3_filename': '20250705_221332_stage3_enhanced.png',
    'image_4_filename': None,
    
    'ai_selected_index': 1,                    # AI picked image 2
    'ai_crop_x1': 0.1, 'ai_crop_y1': 0.0,
    'ai_crop_x2': 0.9, 'ai_crop_y2': 0.8,
    'ai_confidence': 0.87,
    
    'user_selected_index': 2,                  # USER picked image 3!
    'user_action': 'crop',                     # Needs manual cropping
    'ai_crop_accepted': True,                  # New: user left AI crop ON (if same image approved)
    
    'final_crop_x1': None,                     # Not cropped yet
    'final_crop_y1': None,
    'final_crop_x2': None,
    'final_crop_y2': None,
    'crop_timestamp': None,
    
    'image_width': 3072,
    'image_height': 3072,
    
    'selection_match': False,                  # AI was WRONG!
    'crop_match': None                         # Can't calculate yet
}
```

**Files Moved:**
- Images marked "crop" → `crop/` directory
- Images marked "approve" → `selected/` directory  
- Images marked "reject" → `delete_staging/` directory

**Decision File Created:**
For each moved file, create a `.decision` sidecar file:
```json
// crop/20250705_221332_stage3_enhanced.decision
{
    "group_id": "mojo3_group_20251021_234530",
    "project_id": "mojo3",
    "needs_crop": true
}
```

---

### Phase 2: Desktop Multi-Crop (Final Cropping)

**User Flow:**
1. Open `crop/` directory in multi-crop tool
2. For each image:
   - Load image
   - **Read `.decision` file → get `group_id`**
   - User draws crop rectangle
   - Press Enter to submit
3. Image cropped and moved to `crop_cropped/`

**Data Updated:**
```python
# Look up the row by group_id and user_selected_index filename
# Update ONLY the crop fields:
{
    'final_crop_x1': 0.2,       # What user actually cropped
    'final_crop_y1': 0.0,
    'final_crop_x2': 0.7,
    'final_crop_y2': 0.6,
    'crop_timestamp': '2025-10-21T23:52:15Z',
    
    'crop_match': False          # AI crop was 5%+ different
}
```

### ai_crop_accepted Semantics
Update 2025-10-24: Field is live in all active DBs; legacy rows may be NULL.

- ai_crop_accepted = True: User approved the AI-picked image and left the AI crop overlay ON. The file is routed to `crop_auto/` with a `.decision` sidecar containing `ai_crop_coords`.
- ai_crop_accepted = False: User approved the AI-picked image but toggled the AI crop OFF (no crop needed). The file is routed to `selected/` and no `.decision` is created.
- ai_crop_accepted = None: No AI crop was proposed, the user approved a different image than AI picked, or the decision predates this field.

Backfill guidance (optional):
- For rows where `ai_crop_coords IS NOT NULL AND user_action='approve' AND ai_selected_index=user_selected_index`:
  - If file is in `crop_auto/` with a sidecar → set `ai_crop_accepted=True`.
  - If file is in `selected/` and no sidecar in `crop_auto/` → set `ai_crop_accepted=False`.
  - Else keep `NULL`.

---

### Phase 3: Training the AI

**What the Model Learns:**

#### **Selection Model (Ranker):**
```python
INPUT:  [image_1_embedding, image_2_embedding, image_3_embedding]
OUTPUT: user_selected_index (0, 1, or 2)
LOSS:   CrossEntropyLoss on correct index
```

**Feedback loop:** `selection_match` field shows when model is correct!

#### **Crop Model (Crop Proposer):**
```python
INPUT:  [selected_image_embedding, width_norm, height_norm]
OUTPUT: [final_crop_x1, final_crop_y1, final_crop_x2, final_crop_y2]
LOSS:   IoU Loss between AI crop and final crop
```

**Feedback loop:** `crop_match` field shows when crop is good!

#### **Combined Learning:**
The model can learn:
- "When I see images like THIS group, humans prefer THIS one"
- "When I see a portrait with hands, humans crop to THIS region"
- "When I'm wrong about selection, humans usually pick the image with THESE features"

---

## 🗂️ File Organization

### Per-Project Decision Databases

**Structure:**
```
data/training/ai_training_decisions/
├── mojo1.db              # Historical (if we backfill)
├── mojo2.db              # Historical (if we backfill)
├── mojo3.db              # NEW! Active project
└── mojo4.db              # Future projects
```

**Benefits:**
- ✅ One database per project (manageable size, ~1-5MB each)
- ✅ Easy to analyze single project's data (SQL queries)
- ✅ Can archive old projects (copy .db file)
- ✅ Clean separation of concerns
- ✅ Fast queries (indexed, optimized per project)

**Naming Convention:**
- File: `{project_id}.db`
- Group ID: `{project_id}_group_{timestamp}_batch{batch:03d}_img{index:03d}`
  - Example: `mojo3_group_20251021T234530Z_batch001_img002`

---

## 📝 Implementation Checklist

### 1. `scripts/00_start_project.py`

**Changes Needed:**
- ✅ Create new decision log file: `data/training/ai_training_decisions/{project_id}_decisions.csv`
- ✅ Write CSV header on project initialization
- ✅ Document decision log location in project manifest

**Implementation:**
```python
def initialize_ai_training_logs(project_id: str):
    """Create decision log for new project."""
    decisions_dir = Path("data/training/ai_training_decisions")
    decisions_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = decisions_dir / f"{project_id}_decisions.csv"
    
    if not csv_path.exists():
        header = [
            "group_id", "timestamp", "project_id", "directory", "batch_number",
            "image_1_filename", "image_2_filename", "image_3_filename", "image_4_filename",
            "ai_selected_index", "ai_crop_x1", "ai_crop_y1", "ai_crop_x2", "ai_crop_y2", "ai_confidence",
            "user_selected_index", "user_action",
            "final_crop_x1", "final_crop_y1", "final_crop_x2", "final_crop_y2", "crop_timestamp",
            "image_width", "image_height",
            "selection_match", "crop_match"
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    return csv_path
```

**Replaces:** Nothing (new functionality)

**Testing:**
- Create new test project
- Verify decision log file created
- Verify header is correct

---

### 2. `scripts/01_ai_assisted_reviewer.py` (AI Reviewer)

**Changes Needed:**
- ✅ Generate unique `group_id` for each group
- ✅ Log AI recommendation when page loads
- ✅ Log user selection when batch processed
- ✅ Calculate `selection_match` flag
- ✅ Create `.decision` sidecar files for images moved to `crop/`
- ✅ Write decision row to project-specific CSV

**New Functions:**
```python
def generate_group_id(project_id: str, timestamp: str, batch: int, index: int) -> str:
    """Generate unique group ID."""
    return f"{project_id}_group_{timestamp}_batch{batch:03d}_img{index:03d}"

def log_ai_decision(group_id, group_images, ai_rec, user_selection, project_id):
    """Log complete decision to CSV."""
    csv_path = Path(f"data/training/ai_training_decisions/{project_id}_decisions.csv")
    # ... write row with all fields ...

def create_decision_sidecar(image_path, group_id, project_id, needs_crop):
    """Create .decision file next to image."""
    decision_path = image_path.with_suffix('.decision')
    data = {
        'group_id': group_id,
        'project_id': project_id,
        'needs_crop': needs_crop
    }
    with open(decision_path, 'w') as f:
        json.dump(data, f, indent=2)
```

**Modified Functions:**
- `index()`: Generate group_ids, pass to template
- `process_batch()`: Log decisions, create sidecar files
- `get_ai_recommendation()`: Return confidence score

**Replaces:**
- Old `log_crop_decision()` calls (minimal schema)
- Replace with `log_ai_decision()` (full schema)

**Testing:**
- Process 10 groups through AI Reviewer
- Verify decision log has 10 rows
- Verify `.decision` files created for cropped images
- Verify `selection_match` calculated correctly

---

### 3. `scripts/01_web_image_selector.py` (Web Image Selector)

**Changes Needed:**
- ⚠️ **This tool does NOT use AI recommendations!**
- ✅ Can optionally log "manual selection" decisions
- ✅ Create decision log with `ai_selected_index=None`

**Implementation Decision:**
- **Option A:** Don't log anything (tool is for non-AI workflows)
- **Option B:** Log as "manual decisions" with AI fields NULL
- **Recommendation:** Option A (keep it simple, AI Reviewer is the AI tool)

**Replaces:** Nothing

**Testing:** None needed (no changes)

---

### 4. `scripts/02_character_processor.py` (Character Processor)

**Changes Needed:**
- ❌ **No changes needed**
- This tool organizes files into character directories
- Runs BEFORE AI Reviewer in the workflow
- Does not make selection or crop decisions

**Replaces:** Nothing

**Testing:** None needed (no changes)

---

### 5. `scripts/03_web_character_sorter.py` (Character Sorter)

**Changes Needed:**
- ❌ **No changes needed**
- This tool sorts images into character groups
- Runs AFTER cropping in the workflow
- Does not make selection or crop decisions

**Replaces:** Nothing

**Testing:** None needed (no changes)

---

### 6. `scripts/02_ai_desktop_multi_crop.py` (AI Desktop Multi-Crop)

**Changes Needed:**
- ✅ **CRITICAL: This is Phase 2 of the two-stage logging!**
- ✅ Read `.decision` sidecar file to get `group_id`
- ✅ Look up decision row in CSV by `group_id`
- ✅ Update row with final crop coordinates
- ✅ Calculate `crop_match` flag (if AI made a crop suggestion)
- ✅ Delete `.decision` file after updating

**New Functions:**
```python
def read_decision_sidecar(image_path: Path) -> Optional[Dict]:
    """Read .decision file if it exists."""
    decision_path = image_path.with_suffix('.decision')
    if decision_path.exists():
        with open(decision_path) as f:
            return json.load(f)
    return None

def update_decision_with_crop(group_id: str, project_id: str, crop_coords: Tuple, width: int, height: int):
    """Update decision log with final crop coordinates."""
    csv_path = Path(f"data/training/ai_training_decisions/{project_id}_decisions.csv")
    
    # Read all rows
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in rows:
            if row['group_id'] == group_id:
                # Update this row
                row['final_crop_x1'] = crop_coords[0]
                row['final_crop_y1'] = crop_coords[1]
                row['final_crop_x2'] = crop_coords[2]
                row['final_crop_y2'] = crop_coords[3]
                row['crop_timestamp'] = datetime.utcnow().isoformat() + 'Z'
                row['crop_match'] = calculate_crop_match(row)  # Compare AI vs user crop
            rows.append(row)
    
    # Write back
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def calculate_crop_match(row: Dict) -> Optional[bool]:
    """Check if AI crop was within 5% of user crop."""
    if not row.get('ai_crop_x1'):
        return None  # AI didn't make crop suggestion
    
    # Calculate IoU or simple distance metric
    ai_crop = [float(row['ai_crop_x1']), float(row['ai_crop_y1']), 
               float(row['ai_crop_x2']), float(row['ai_crop_y2'])]
    user_crop = [float(row['final_crop_x1']), float(row['final_crop_y1']), 
                 float(row['final_crop_x2']), float(row['final_crop_y2'])]
    
    # Simple threshold: all corners within 5% tolerance
    tolerance = 0.05
    for ai, user in zip(ai_crop, user_crop):
        if abs(ai - user) > tolerance:
            return False
    return True
```

**Modified Functions:**
- `crop_and_save()`: Read decision sidecar, update CSV, delete sidecar

**Replaces:**
- Old `log_select_crop_entry()` call
- Replace with `update_decision_with_crop()`

**Testing:**
- Process 5 images with `.decision` files
- Verify decision log updated with crop coords
- Verify `crop_match` calculated correctly
- Verify `.decision` files deleted after processing

---

### 7. `scripts/05_web_multi_directory_viewer.py` (Multi-Directory Viewer)

**Changes Needed:**
- ❌ **No changes needed**
- This tool is for viewing/reviewing completed work
- Does not make decisions or log training data

**Replaces:** Nothing

**Testing:** None needed (no changes)

---

### 8. `scripts/99_web_duplicate_finder.py` (Duplicate Finder)

**Changes Needed:**
- ❌ **No changes needed**
- This tool finds duplicate images
- Does not make selection or crop decisions

**Replaces:** Nothing

**Testing:** None needed (no changes)

---

### 9. `scripts/07_finish_project.py` (Finish Project)

**Changes Needed:**
- ✅ Archive decision log to project's final directory
- ✅ Verify all decision rows have `final_crop_x1` filled (no incomplete rows)
- ✅ Generate summary statistics (selection accuracy, crop accuracy)

**New Functionality:**
```python
def archive_decision_log(project_id: str, project_final_dir: Path):
    """Copy decision log to project archive."""
    source = Path(f"data/training/ai_training_decisions/{project_id}_decisions.csv")
    dest = project_final_dir / "ai_decisions.csv"
    shutil.copy(source, dest)
    print(f"✓ Archived decision log to {dest}")

def generate_ai_performance_report(project_id: str):
    """Analyze AI performance on this project."""
    csv_path = Path(f"data/training/ai_training_decisions/{project_id}_decisions.csv")
    
    total = 0
    selection_correct = 0
    crop_correct = 0
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row['selection_match'] == 'True':
                selection_correct += 1
            if row['crop_match'] == 'True':
                crop_correct += 1
    
    print(f"\n📊 AI Performance for {project_id}:")
    print(f"   Selection Accuracy: {selection_correct}/{total} ({selection_correct/total*100:.1f}%)")
    print(f"   Crop Accuracy: {crop_correct}/{total} ({crop_correct/total*100:.1f}%)")
```

**Replaces:** Nothing (new functionality)

**Testing:**
- Finish test project
- Verify decision log archived
- Verify performance report generated

---

## 🧹 Cleanup Guide

### What This Replaces

#### **1. Old Crop Training Schema (19 columns)**
**File:** `data/training/select_crop_log.csv`

**Status:** Keep for historical data, but NEW crops go to new schema

**Migration Strategy:**
- ✅ Keep old file as-is (historical record)
- ✅ Mark as "Legacy - October 2025" in docs
- ✅ New crops use `ai_training_decisions/{project}_decisions.csv`
- ✅ Can optionally backfill old data into new schema (future project)

#### **2. Minimal 8-Column Schema**
**File:** `data/training/crop_training_data.csv`

**Status:** Never used! Remove references.

**Cleanup:**
- ❌ Remove `log_crop_decision()` function from `companion_file_utils.py`
- ❌ Remove import from AI Reviewer
- ✅ This schema was designed but never implemented (good timing!)

#### **3. AI Data Collection (snapshots)**
**Files:** `data/ai_data/training_snapshots/`

**Status:** Keep for now, evaluate later

**These capture:**
- Image thumbnails
- Crop decisions
- Session metadata

**Decision:** Keep as supplementary data, but decision log is primary source

---

### Files to Delete

**None!** We're not deleting anything - just adding new functionality.

**Old files to mark as "Legacy":**
- `data/training/select_crop_log.csv` → Rename to `select_crop_log_LEGACY.csv`
- Add `_LEGACY_SCHEMAS.md` document explaining old formats

---

## 🧪 Testing Strategy

### Unit Tests

**File:** `scripts/tests/test_ai_training_decisions_v3.py`

**Test Cases:**
1. `test_generate_group_id()` - Verify unique IDs
2. `test_log_ai_decision()` - Write decision row
3. `test_read_decision_sidecar()` - Read `.decision` file
4. `test_update_decision_with_crop()` - Update CSV row
5. `test_calculate_selection_match()` - Verify boolean logic
6. `test_calculate_crop_match()` - Verify IoU/tolerance
7. `test_decision_log_per_project()` - Verify file isolation

### Integration Tests

**Test Scenario 1: Happy Path**
1. Start new project → decision log created
2. Process 10 groups in AI Reviewer → 10 rows logged
3. Crop 5 images in Desktop Multi-Crop → 5 rows updated
4. Finish project → decision log archived, stats generated

**Test Scenario 2: AI Agrees with User**
1. AI picks image 2, user picks image 2 → `selection_match=True`
2. AI crops (0.1, 0.1, 0.9, 0.9), user crops (0.11, 0.09, 0.91, 0.88) → `crop_match=True`

**Test Scenario 3: AI Disagrees with User**
1. AI picks image 1, user picks image 3 → `selection_match=False`
2. AI crops (0.1, 0.1, 0.9, 0.9), user crops (0.3, 0.0, 0.7, 0.6) → `crop_match=False`

**Test Scenario 4: Missing Decision File**
1. Image in `crop/` has no `.decision` file
2. Desktop Multi-Crop logs warning, skips decision update
3. Image still gets cropped and saved (workflow continues)

### Manual Testing Checklist

- [ ] Create new project, verify decision log file created
- [ ] Process 5 groups in AI Reviewer, verify 5 rows in CSV
- [ ] Verify `.decision` files created for images moved to `crop/`
- [ ] Open Desktop Multi-Crop, crop 3 images
- [ ] Verify decision log updated with crop coordinates
- [ ] Verify `.decision` files deleted after cropping
- [ ] Finish project, verify decision log archived
- [ ] Check AI performance report shows correct stats

---

## 🎓 Training the AI Models

### Using the New Schema

**Benefits for Training:**

#### **1. Contextual Learning**
Old approach:
```python
# Just one image, no context
train_crop_model(image_embedding, crop_coords)
```

New approach:
```python
# Model sees the CHOICE that was made
train_selection_model(
    [img1_emb, img2_emb, img3_emb],  # All options
    user_selected_index=2              # Which was chosen
)
```

**Why better:** Model learns "given THESE options, pick THIS one"

#### **2. Correction Learning**
```python
# When AI is wrong, we know what it SHOULD have done
if not selection_match:
    # This is a hard negative example!
    train_with_higher_weight(
        images=group_images,
        ai_pick=ai_selected_index,      # Wrong
        correct_pick=user_selected_index # Right
    )
```

**Why better:** Model learns from its mistakes explicitly

#### **3. Progressive Accuracy Tracking**
```python
# Track model improvement over time
epoch_1_accuracy = evaluate_on_mojo1_decisions()  # 45%
epoch_5_accuracy = evaluate_on_mojo1_decisions()  # 67%
epoch_10_accuracy = evaluate_on_mojo1_decisions() # 82%
```

**Why better:** We can see if retraining is actually helping!

---

### Training Script Updates

**File:** `scripts/ai/train_ranker_model.py`

**Changes:**
```python
def load_training_data_v3(project_ids: List[str]):
    """Load from new decision logs."""
    all_data = []
    
    for project_id in project_ids:
        csv_path = f"data/training/ai_training_decisions/{project_id}_decisions.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Build training example from group
                group_images = [
                    row['image_1_filename'],
                    row['image_2_filename'],
                    row['image_3_filename'],
                    row['image_4_filename']
                ]
                group_images = [img for img in group_images if img]  # Remove NULLs
                
                # Label is the user's choice
                label = int(row['user_selected_index'])
                
                # Weight examples where AI was wrong more heavily
                weight = 2.0 if row['selection_match'] == 'False' else 1.0
                
                all_data.append({
                    'images': group_images,
                    'label': label,
                    'weight': weight
                })
    
    return all_data
```

**File:** `scripts/ai/train_crop_model.py`

**Changes:**
```python
def load_crop_training_data_v3(project_ids: List[str]):
    """Load crop decisions from new schema."""
    all_data = []
    
    for project_id in project_ids:
        csv_path = f"data/training/ai_training_decisions/{project_id}_decisions.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only use rows where cropping actually happened
                if not row['final_crop_x1']:
                    continue
                
                # Get the image that was actually selected
                selected_img = row[f"image_{int(row['user_selected_index'])+1}_filename"]
                
                crop_coords = [
                    float(row['final_crop_x1']),
                    float(row['final_crop_y1']),
                    float(row['final_crop_x2']),
                    float(row['final_crop_y2'])
                ]
                
                # Weight corrections more heavily
                weight = 2.0 if row['crop_match'] == 'False' else 1.0
                
                all_data.append({
                    'image': selected_img,
                    'crop': crop_coords,
                    'width': int(row['image_width']),
                    'height': int(row['image_height']),
                    'weight': weight
                })
    
    return all_data
```

---

## 📅 Implementation Timeline

### Phase 1: Foundation (Day 1)
- [ ] Create schema documentation (this file)
- [ ] Create `data/training/ai_training_decisions/` directory
- [ ] Write utility functions (`generate_group_id`, `log_ai_decision`, etc.)
- [ ] Write unit tests for utility functions

### Phase 2: AI Reviewer Integration (Day 2)
- [ ] Update AI Reviewer to generate group IDs
- [ ] Update AI Reviewer to log decisions
- [ ] Update AI Reviewer to create `.decision` sidecar files
- [ ] Test with 10-20 groups manually

### Phase 3: Desktop Multi-Crop Integration (Day 3)
- [ ] Update Desktop Multi-Crop to read `.decision` files
- [ ] Update Desktop Multi-Crop to update decision log
- [ ] Implement `crop_match` calculation
- [ ] Test with 5-10 images manually

### Phase 4: Project Lifecycle Integration (Day 4)
- [ ] Update `00_start_project.py` to create decision log
- [ ] Update `07_finish_project.py` to archive and report
- [ ] Test full project lifecycle

### Phase 5: Training Scripts (Day 5)
- [ ] Update `train_ranker_model.py` to use new schema
- [ ] Update `train_crop_model.py` to use new schema
- [ ] Retrain models on combined old + new data
- [ ] Evaluate performance improvement

### Phase 6: Cleanup & Documentation (Day 6)
- [ ] Mark old schemas as legacy
- [ ] Update all documentation
- [ ] Create migration guide for old data (optional)
- [ ] Final testing and validation

---

## 🚨 Risks & Mitigation

### Risk 1: CSV File Corruption
**Problem:** Multiple processes writing to same CSV

**Mitigation:**
- Use file locking when writing
- Atomic write pattern (temp file + rename)
- Regular backups of decision logs

### Risk 2: Missing Decision Files
**Problem:** `.decision` file deleted or lost

**Mitigation:**
- Desktop Multi-Crop continues without it (logs warning)
- Can manually reconstruct from filenames + timestamps
- Not critical for workflow to continue

### Risk 3: Schema Evolution
**Problem:** Need to add columns later

**Mitigation:**
- CSV format easily extendable (add columns to right)
- Utility functions handle missing fields gracefully
- Version number in filename if major changes needed

### Risk 4: Performance with Large CSVs
**Problem:** Reading entire CSV to update one row

**Mitigation:**
- For now: acceptable (CSVs stay <10MB per project)
- Future: migrate to SQLite if needed
- Keep per-project logs small

---

## 📊 Success Metrics

### Immediate Metrics (After Implementation)
- ✅ Decision log created for each new project
- ✅ Every AI Reviewer batch logs decisions
- ✅ Every Desktop Multi-Crop crop updates decision log
- ✅ No workflow interruptions or errors

### Training Data Quality Metrics
- ✅ 100% of cropped images have complete decision rows
- ✅ `selection_match` and `crop_match` flags calculated
- ✅ Decision logs archived with finished projects

### AI Performance Metrics (After Retraining)
- 📈 Selection accuracy improves over baseline
- 📈 Crop IoU improves over baseline
- 📈 User intervention rate decreases (more auto-approvals)

### Long-Term Metrics (After Multiple Projects)
- 📈 AI Reviewer approval rate increases (less manual cropping)
- 📈 Time per image decreases (faster workflow)
- 📈 Model confidence correlates with correctness

---

## 🎯 Key Takeaways

### What Makes This Schema Better

**1. Complete Context**
- Old: Crop coordinates in isolation
- New: Full group context + AI recommendation + user correction

**2. Explicit Feedback Loop**
- Old: No way to know if AI was right or wrong
- New: `selection_match` and `crop_match` flags

**3. Two-Stage Linking**
- Old: Desktop Multi-Crop logs independently
- New: Links back to AI Reviewer decision via `group_id`

**4. Scalable Organization**
- Old: One giant CSV for all projects
- New: Per-project CSVs (manageable, archivable)

**5. Training-Ready**
- Old: Requires complex preprocessing
- New: Direct training data format

---

## 📚 Related Documents

- `AI_TRAINING_DATA_STRUCTURE.md` - Original training data design
- `archives/ai/AI_TRAINING_CROP_AND_RANKING.md` - Model architecture docs (archived)
- `COMPANION_FILE_SYSTEM_GUIDE.md` - File handling patterns
- `FILE_SAFETY_SYSTEM.md` - Data integrity rules
- `TECHNICAL_KNOWLEDGE_BASE.md` - General system knowledge

---

## ✅ Implementation Approval

**Status:** ⏳ Awaiting Erik's approval

**Questions for Erik:**
1. Is the schema complete? Any fields missing?
2. Is per-project organization correct?
3. Should we backfill old data or start fresh?
4. Any concerns about the two-stage logging approach?

**Next Steps After Approval:**
1. Create utility functions
2. Write unit tests
3. Integrate into AI Reviewer (Phase 2)
4. Integrate into Desktop Multi-Crop (Phase 3)
5. Test end-to-end workflow
6. Retrain models with new data

---

**Document Version:** 1.0  
**Last Updated:** October 21, 2025  
**Author:** Claude (Sonnet 4.5) with Erik Sjaastad  
**Status:** Design Complete, Implementation Pending

