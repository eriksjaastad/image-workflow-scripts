# Phase 2 Quick Start Guide
*Moving from Data Collection to AI Training*

---

## 📚 Quick Navigation - All AI Documents

**Current Document:** AI_TRAINING_PHASE2_QUICKSTART.md (You are here)

### Core Workflow Documents:
1. **[AI_TRAINING_PHASE2_QUICKSTART.md](AI_TRAINING_PHASE2_QUICKSTART.md)** ⭐ **START HERE**
   - Practical step-by-step implementation guide
   - Install dependencies → Extract features → Train models
   - **Status:** Training data ready (17k examples), models not yet trained

2. **[AUTOMATION_REVIEWER_SPEC.md](archives/misc/AUTOMATION_REVIEWER_SPEC.md)** ⭐ **BUILD NEXT**
   - Complete spec for the review UI (`scripts/07_automation_reviewer.py`)
   - Web interface for approving/rejecting AI recommendations
   - **Status:** Fully documented, ready to build

3. **[AI_TRAINING_CROP_AND_RANKING.md](archives/ai/AI_TRAINING_CROP_AND_RANKING.md)**
   - Comprehensive technical training plan
   - Details: CLIP embeddings, saliency, anomaly detection, crop optimization
   - **Status:** Reference document for deep dives

### Supporting Documents:
4. **[AI_ANOMALY_DETECTION_OPTIONS.md](AI_ANOMALY_DETECTION_OPTIONS.md)**
   - Hand/foot anomaly detection options (MediaPipe, OpenPose, etc.)
   - **Status:** Future enhancement reference

5. **[CURRENT_TODO_LIST.md](CURRENT_TODO_LIST.md)** - Section: "Automation Pipeline Plan"
   - Lines 251-490: Full automation pipeline details
   - Data formats, safety rules, testing plan
   - **Status:** Living document, updated continuously

6. **[scripts/ai/README.md](../scripts/ai/README.md)**
   - Documents existing AI training scripts
   - Safety guarantees and usage instructions
   - **Status:** Current for existing scripts

### Implementation Order:
```
Phase 1: ✅ Data Collection (COMPLETE - 17k examples)
         ↓
Phase 2: ⏳ Train AI Models (THIS DOCUMENT)
         ↓
Phase 3: ⏳ Build Automation Reviewer UI (AUTOMATION_REVIEWER_SPEC.md)
         ↓
Phase 4: ⏳ Test in Sandbox + Iterate (CURRENT_TODO_LIST.md)
```

---

## 📊 Current Training Data Status (Updated: Oct 20, 2025)

### ✅ Data Collection Complete: 17,032 Training Examples

**Selection Data (AI-Assisted Reviewer):**
- Mojo 1: 5,244 selections
- Mojo 2: 4,594 selections
- **Total: 9,838 selection decisions** ✅

**Crop Data (Multi Crop Tool):**
- Total crop decisions: 7,194 ✅
- Dates: Oct 4, 8, 16, 19, 2025
- Source directories: `crop/` and character subdirectories

**Combined Total: 17,032 training examples**

---

## 🚨 CRITICAL: Training Data Structure & Project Boundaries

### **RULE #1: Projects NEVER Mix**

Each project (Mojo 1, Mojo 2, Eleni, etc.) is a **separate character**. Training data from different projects **MUST NEVER be compared or mixed**.

**Why This Matters:**
- Mojo 1 is Character A, Mojo 2 is Character B
- Your preferences for Character A don't inform preferences for Character B
- Comparing images across projects is meaningless and will corrupt the model

### **RULE #2: Each Selection is an Image Set Within One Project**

Every row in `selection_only_log.csv` is a **complete image set** from ONE project:

```csv
session_id,set_id,chosen_path,neg_paths,timestamp
20251004_213737,group_8,/Users/.../mojo1/20250708_060711_stage2_upscaled.png,"[""/Users/.../mojo1/20250708_060558_stage1_generated.png""]",2025-10-05T01:43:47Z
```

**What this means:**
- `chosen_path` = winner image YOU picked
- `neg_paths` = loser images YOU rejected
- **ALL images in this set are from the SAME project (mojo1)**
- **ALL images in this set have the SAME timestamp/group** (different stages of the same generation)

### **RULE #3: Match by Filename Within Project Context**

When loading training data:

1. **Extract project from path:**
   - `/Users/.../mojo1/20250708_060711_stage2.png` → project = `mojo1`
   - `/Users/.../mojo2/_mixed/20250801_123456_stage3.png` → project = `mojo2`

2. **Match by filename only (ignore full path):**
   - Embeddings cache may have: `training data/mojo1/faces/20250708_060711_stage2.png`
   - CSV may have: `/Users/.../mojo1/20250708_060711_stage2.png`
   - **Match by:** `20250708_060711_stage2.png` (filename only)

3. **Constrain matches to same project:**
   - Only look for `20250708_060711_stage2.png` within `mojo1` embeddings
   - **NEVER** match a mojo1 CSV entry to a mojo2 embedding

### **RULE #4: Training Pairs Must Be From Same Set**

When creating training pairs for the ranker:

```python
# ✅ CORRECT: Both from same set (group_8, mojo1)
pair1 = (winner: mojo1/.../stage2.png, loser1: mojo1/.../stage1.png)
pair2 = (winner: mojo1/.../stage2.png, loser2: mojo1/.../stage3.png)

# ❌ WRONG: Mixed projects
pair_bad = (winner: mojo1/.../stage2.png, loser: mojo2/.../stage1.png)
```

### **Implementation Guidelines**

**When loading training data:**

```python
# 1. Parse project from path
def get_project_id(path: str) -> str:
    """Extract project ID from file path"""
    # /Users/.../mojo1/file.png → 'mojo1'
    # /Users/.../mojo2/_mixed/file.png → 'mojo2'
    parts = Path(path).parts
    for part in parts:
        if part.startswith('mojo') or part in ['eleni', 'aiko', 'dalia']:
            return part
    return 'unknown'

# 2. Match by filename within project
def find_embedding(csv_path: str, embeddings_cache: dict) -> Optional[np.ndarray]:
    """Find embedding for a CSV path entry"""
    project_id = get_project_id(csv_path)
    filename = Path(csv_path).name
    
    # Search embeddings cache for this project + filename
    for cache_path, embedding in embeddings_cache.items():
        if project_id in cache_path and filename in cache_path:
            return embedding
    return None

# 3. Verify all images in set are from same project
def validate_set(chosen_path: str, neg_paths: List[str]) -> bool:
    """Ensure all paths in a set are from the same project"""
    chosen_project = get_project_id(chosen_path)
    neg_projects = [get_project_id(p) for p in neg_paths]
    
    if not all(p == chosen_project for p in neg_projects):
        raise ValueError(f"Mixed projects in set: {chosen_project} vs {neg_projects}")
    return True
```

### **Project Manifest Reference**

Each project has a manifest at `data/projects/<projectId>.project.json`:

```json
{
  "projectId": "mojo1",
  "title": "Mojo1",
  "startedAt": "2025-10-01T00:00:00Z",
  "finishedAt": "2025-10-11T17:29:49Z",
  "counts": {
    "initialImages": 19183,
    "finalImages": 6453
  }
}
```

**Use project dates to:**
- Determine which files belong to which project (based on file modification times)
- Filter training data by project if needed
- Track training data distribution across projects

### **Training Strategy**

**Option A: Unified Model (Recommended)**
- Train ONE model on ALL projects combined
- Model learns your **general aesthetic preferences**
- BUT: Only compare images within same project during training
- Pairs always respect project boundaries

**Option B: Per-Project Models**
- Train separate model for each project
- Model learns project-specific preferences
- More complex to maintain
- Only use if Option A performs poorly

**Start with Option A.** Most aesthetic preferences (composition, lighting, expression) transfer across characters.

---

## Current Status
- ✅ Phase 1 Complete: 17,032 training examples collected (Mojo 1 + Mojo 2)
- ⏳ Phase 2 Ready: Install dependencies and train first models

## What Phase 2 Achieves
By the end of Phase 2, the AI will:
1. **Suggest which image is "best"** from a set (based on your 9,838 selections)
2. **Propose crop coordinates** for images (based on your 7,194 crops)
3. Show **confidence scores** so you know when to trust it

You still approve/reject everything - it's just making suggestions.

---

## Step-by-Step Implementation

### Step 1: Install ML Dependencies (~30 min)

**Check current environment:**
```bash
source .venv311/bin/activate
python --version  # Should be 3.11+
```

**Install PyTorch with Apple Silicon support:**
```bash
# Core ML framework
pip install torch torchvision torchaudio

# Training utilities
pip install lightning accelerate

# Models & image processing
pip install open_clip_torch  # For CLIP embeddings
pip install opencv-python pillow numpy scipy
pip install imagehash        # For duplicate detection
pip install mediapipe        # For hand/pose detection

# Optional but useful
pip install pandas pyarrow    # For efficient data handling
pip install tqdm             # Progress bars
```

**Verify Apple GPU works:**
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Should print: MPS available: True
```

---

### Step 2: Compute Features (~2-4 hours one-time)

These scripts need to be created. They process your existing images once to extract features the AI will use.

#### 2A. CLIP Embeddings (semantic understanding)
**Purpose:** Convert images to 512-dimensional vectors that capture "what's in the image"

**Script to create:** `scripts/ai_training/compute_embeddings.py`
```python
#!/usr/bin/env python3
"""Extract CLIP embeddings for all images"""
import torch
import open_clip
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Use Apple GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load lightweight CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# Process all images
embeddings = []
for img_path in tqdm(list(Path("data/training/images/").glob("*.png"))):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    embeddings.append({
        'image': img_path.name,
        'embedding': embedding.cpu().numpy().flatten().tolist()
    })

# Save
df = pd.DataFrame(embeddings)
df.to_parquet("data/sidecar/embeddings.parquet")
print(f"Saved {len(embeddings)} embeddings")
```

**Run:**
```bash
python scripts/ai/compute_embeddings.py
# Takes ~1-2 hours for thousands of images
```

#### 2B. Image Hashes (duplicate detection)
**Purpose:** Detect nearly-identical images

**Script to create:** `scripts/ai_training/compute_hashes.py`
```python
#!/usr/bin/env python3
"""Compute perceptual hashes for duplicate detection"""
import imagehash
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

hashes = []
for img_path in tqdm(list(Path("data/training/images/").glob("*.png"))):
    img = Image.open(img_path)
    phash = str(imagehash.phash(img))
    hashes.append({'image': img_path.name, 'phash': phash})

df = pd.DataFrame(hashes)
df.to_parquet("data/sidecar/hashes.parquet")
print(f"Saved {len(hashes)} hashes")
```

**Run:**
```bash
# Planned utility; not currently implemented as a standalone script
# python scripts/ai_training/compute_hashes.py
```

#### 2C. Saliency & Hand Detection (optional but recommended)
These help the AI understand:
- Where the subject is (saliency)
- Where hands/fingers are (for detecting anomalies)

**Can skip for initial Phase 2 and add later if needed.**

---

### Step 3: Train Ranking Model (~1 hour)

**Purpose:** Learn which images you prefer from sets

**Script to create:** `scripts/ai_training/train_ranker.py`
```python
#!/usr/bin/env python3
"""Train a model to rank images by preference"""
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load training data
selections = pd.read_csv("data/training/selection_only_log.csv")
embeddings = pd.read_parquet("data/sidecar/embeddings.parquet")

# Simple ranking model (Bradley-Terry style)
class RankingModel(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Score per image
        )
    
    def forward(self, x):
        return self.net(x)

# Training loop (simplified)
model = RankingModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ... training code here ...
# Uses pairwise comparisons from your selections

# Save trained model
torch.save(model.state_dict(), "models/ranker_v1.pt")
print("Ranking model trained!")
```

**Run:**
```bash
python scripts/ai/train_ranker.py
# Takes ~30-60 minutes
```

---

### Step 4: Train Crop Proposer (~1 hour)

**Purpose:** Learn where you crop images

**Script to create:** `scripts/ai_training/train_crop_proposer.py`
```python
#!/usr/bin/env python3
"""Train a model to propose crop boxes"""
import pandas as pd
import torch
import torch.nn as nn

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load crop data
crops = pd.read_csv("data/training/select_crop_log.csv")
embeddings = pd.read_parquet("data/sidecar/embeddings.parquet")

# Crop prediction model
class CropProposer(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # x1, y1, x2, y2
        )
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))  # Normalize to 0-1

# Training loop
model = CropProposer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ... training code here ...
# Predicts crop coordinates from image embeddings

# Save
torch.save(model.state_dict(), "models/crop_proposer_v1.pt")
print("Crop proposer trained!")
```

**Run:**
```bash
python scripts/ai/train_crop_proposer.py
# Takes ~30-60 minutes
```

---

### Step 5: Test Inference

**Script to create:** `scripts/ai/test_models.py` (exists; extend as needed)
```python
#!/usr/bin/env python3
"""Test AI predictions on new images"""
import torch
from train_ranker import RankingModel
from train_crop_proposer import CropProposer

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load models
ranker = RankingModel().to(device)
ranker.load_state_dict(torch.load("models/ranker_v1.pt"))
ranker.eval()

cropper = CropProposer().to(device)
cropper.load_state_dict(torch.load("models/crop_proposer_v1.pt"))
cropper.eval()

# Test on a set of images
test_images = ["img1.png", "img2.png", "img3.png"]
# ... compute embeddings for test images ...
# ... get scores ...

print("Rankings:")
for img, score in zip(test_images, scores):
    print(f"  {img}: {score:.3f}")

print("\nSuggested winner:", test_images[best_idx])
print("Suggested crop:", crop_box)
```

---

## What You'll Need to Build Next

### Phase 3: Review UI with AI Suggestions

**Not yet built - needs development:**

A modified version of your current tools that:
1. Shows AI's suggested winner + crop
2. Lets you approve/edit/reject
3. Logs your feedback for continuous learning

**Options:**
- Modify existing `01_ai_assisted_reviewer.py` to show AI suggestions
- Create new `01_ai_assisted_selector.py` script
- Add "AI mode" toggle to current tools

This is ~4-8 hours of coding work.

---

## Quick Assessment

**Can you code Python/PyTorch?**
- ✅ **Yes:** Follow this guide, create the scripts above
- ❌ **No:** You'll need help from someone who can

**Realistic Timeline:**
- **Week 1:** Install deps + compute features (mostly "run and wait")
- **Week 2:** Train initial models (need Python/ML knowledge)
- **Week 3:** Build review UI (need Python/web dev knowledge)
- **Week 4:** Test and iterate

**Alternative:** Hire an ML engineer for 1-2 weeks to implement Phase 2-3.

---

## Decision Point

**Should you do Phase 2 now?**

**Pros:**
- You have 14k training examples ready
- M4 Pro is perfect for this
- Could save hundreds of hours of manual work long-term

**Cons:**
- Requires ML/Python expertise (or hiring help)
- 1-3 weeks implementation time
- No guarantee of 99%+ accuracy initially

**My recommendation:**
1. **Install dependencies now** (Step 1, 30 min, easy)
2. **Run feature extraction** (Step 2, automated, 2-4 hours)
3. **Then decide:** Try training yourself OR get help

This way you're 50% done with minimal risk, and you can see if it's worth continuing.

---

## Resources

- **Training plan:** `Documents/image_cropping_ranking_training_plan.md`
- **PyTorch MPS guide:** https://pytorch.org/docs/stable/notes/mps.html
- **OpenCLIP docs:** https://github.com/mlfoundations/open_clip
- **Your data:** `data/training/*.csv` (14k examples)

**Questions?** Let me know if you want me to:
- Scaffold these scripts for you
- Explain any concepts in more detail
- Help assess if this is worth the investment

