# Phase 2 Quick Start Guide
*Moving from Data Collection to AI Training*

## Current Status
- ✅ Phase 1 Complete: 14,232 training examples collected
- ⏳ Phase 2 Ready: Install dependencies and train first models

## What Phase 2 Achieves
By the end of Phase 2, the AI will:
1. **Suggest which image is "best"** from a set (based on your 9,839 selections)
2. **Propose crop coordinates** for images (based on your 4,393 crops)
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
python scripts/ai_training/compute_embeddings.py
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
python scripts/ai_training/compute_hashes.py
# Takes ~10-30 minutes
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
python scripts/ai_training/train_ranker.py
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
python scripts/ai_training/train_crop_proposer.py
# Takes ~30-60 minutes
```

---

### Step 5: Test Inference

**Script to create:** `scripts/ai_training/test_inference.py`
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
- Modify existing `01_web_image_selector.py` to show AI suggestions
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

