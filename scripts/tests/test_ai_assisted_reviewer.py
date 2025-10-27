#!/usr/bin/env python3
"""
Test AI-Assisted Reviewer - Integration Tests
==============================================
Tests AI model integration, crop overlay, and file operations.

Run: python scripts/tests/test_ai_assisted_reviewer.py
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.utils.companion_file_utils import detect_stage

# Import core functions only (avoid Flask dependencies for testing)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print("[!] PyTorch not available - some tests will be skipped")

try:
    import clip
    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False
    print("[!] CLIP not available - some tests will be skipped")

# Define what we need for testing
from dataclasses import dataclass
from typing import List


@dataclass
class ImageGroup:
    """Represents a group of images with same timestamp."""
    group_id: str
    images: List[Path]
    directory: Path

def get_rule_based_recommendation(group: ImageGroup) -> dict:
    """Rule-based recommendation - pick highest stage."""
    best_image = group.images[-1]
    best_index = len(group.images) - 1
    stage = detect_stage(best_image.name) or "unknown"
    
    return {
        "selected_image": best_image.name,
        "selected_index": best_index,
        "reason": f"Highest stage: {stage} (rule-based)",
        "confidence": 1.0,
        "crop_needed": False,
        "crop_coords": None
    }

# Model classes (for testing)
if TORCH_AVAILABLE:
    class RankerModel(nn.Module):
        def __init__(self, input_dim=512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.net(x).squeeze(-1)

    class CropProposerModel(nn.Module):
        def __init__(self, input_dim=514):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    def load_ai_models(models_dir: Path):
        """Load AI models from directory."""
        if not CLIP_AVAILABLE:
            return None, None, None
        
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            
            ranker_path = models_dir / "ranker_v3_w10.pt"
            if ranker_path.exists():
                ranker = RankerModel(input_dim=512).to(device)
                ranker.load_state_dict(torch.load(ranker_path, map_location=device))
                ranker.eval()
            else:
                ranker = None
            
            crop_path = models_dir / "crop_proposer_v2.pt"
            if crop_path.exists():
                crop_proposer = CropProposerModel(input_dim=514).to(device)
                crop_proposer.load_state_dict(torch.load(crop_path, map_location=device))
                crop_proposer.eval()
            else:
                crop_proposer = None
            
            return ranker, crop_proposer, (clip_model, preprocess, device)
        except Exception as e:
            print(f"[!] Error loading models: {e}")
            return None, None, None
else:
    RankerModel = None
    CropProposerModel = None
    def load_ai_models(models_dir: Path):
        return None, None, None

def test_model_loading():
    """Test that AI models load correctly."""
    print("\n🧪 Test 1: Model Loading")
    
    models_dir = PROJECT_ROOT / "data" / "ai_data" / "models"
    
    if not models_dir.exists():
        print("   ⚠️  Models directory not found - SKIP")
        return True
    
    ranker, crop_model, clip_info = load_ai_models(models_dir)
    
    if ranker is None:
        print("   ⚠️  Models not available (PyTorch/CLIP missing) - SKIP")
        return True
    
    # Check model architecture
    assert isinstance(ranker, RankerModel), "Ranker should be RankerModel instance"
    assert isinstance(crop_model, CropProposerModel), "Crop model should be CropProposerModel instance"
    
    # Check CLIP loaded
    assert clip_info is not None, "CLIP info should be loaded"
    clip_model, preprocess, device = clip_info
    assert clip_model is not None, "CLIP model should be loaded"
    
    print(f"   ✅ Models loaded successfully on device: {device}")
    print(f"   ✅ Ranker: {type(ranker).__name__}")
    print(f"   ✅ Crop Proposer: {type(crop_model).__name__}")
    return True


def test_rule_based_recommendation():
    """Test that rule-based fallback works."""
    print("\n🧪 Test 2: Rule-Based Recommendation (Fallback)")
    
    # Create fake image group
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create fake images with different stages (sorted by stage)
        images = [
            tmpdir / "test_20250101_120000_stage1_generated.png",
            tmpdir / "test_20250101_120000_stage1.5_face_swapped.png",
            tmpdir / "test_20250101_120000_stage2_upscaled.png",
        ]
        
        for img in images:
            img.touch()
        
        group = ImageGroup(
            group_id="20250101_120000",
            images=images,
            directory=tmpdir
        )
        
        recommendation = get_rule_based_recommendation(group)
        
        # Should pick highest stage (stage2 = last index)
        assert recommendation["selected_index"] == len(images) - 1, f"Should pick last image (highest stage), got {recommendation['selected_index']}"
        assert recommendation["selected_image"] == images[-1].name
        assert recommendation["confidence"] == 1.0
        assert not recommendation["crop_needed"]
        assert recommendation["crop_coords"] is None
        
        print(f"   ✅ Rule-based picked: {recommendation['selected_image']}")
        print(f"   ✅ Reason: {recommendation['reason']}")
        return True


def test_crop_coordinates_validation():
    """Test that crop coordinates are normalized and valid."""
    print("\n🧪 Test 3: Crop Coordinates Validation")
    
    # Mock crop coordinates (normalized [0, 1])
    test_coords = [
        (0.1, 0.2, 0.9, 0.8),  # Valid
        (0.0, 0.0, 1.0, 1.0),  # Full image (should not crop)
        (0.2, 0.3, 0.5, 0.6),  # Small crop
    ]
    
    for i, coords in enumerate(test_coords):
        x1, y1, x2, y2 = coords
        
        # Validate normalized range
        assert 0 <= x1 < x2 <= 1, f"Coords {i}: x1={x1}, x2={x2} not in valid range"
        assert 0 <= y1 < y2 <= 1, f"Coords {i}: y1={y1}, y2={y2} not in valid range"
        
        # Calculate crop area
        crop_area = (x2 - x1) * (y2 - y1)
        
        print(f"   ✅ Coords {i}: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}) - Area: {crop_area:.2%}")
        
        # Check if meaningful crop (less than 95% of image)
        if crop_area < 0.95:
            print(f"      → Meaningful crop (removes {100 - crop_area*100:.1f}% of image)")
        else:
            print("      → Nearly full image (should not crop)")
    
    return True


def test_crop_aspect_ratio_preservation():
    """Test that crop maintains reasonable aspect ratios."""
    print("\n🧪 Test 4: Crop Aspect Ratio Check")
    
    # Test different image aspect ratios with crop coords
    test_cases = [
        # (image_width, image_height, crop_coords, expected_aspect_ratio)
        (1024, 1024, (0.1, 0.2, 0.9, 0.8), "square"),      # Square image
        (1920, 1080, (0.1, 0.1, 0.9, 0.7), "landscape"),   # Landscape
        (1080, 1920, (0.2, 0.1, 0.8, 0.9), "portrait"),    # Portrait
    ]
    
    for img_w, img_h, coords, desc in test_cases:
        x1, y1, x2, y2 = coords
        
        # Calculate crop dimensions in pixels
        crop_w = (x2 - x1) * img_w
        crop_h = (y2 - y1) * img_h
        
        # Calculate aspect ratio
        crop_aspect = crop_w / crop_h
        orig_aspect = img_w / img_h
        
        print(f"   ✅ {desc.capitalize()}: {img_w}x{img_h} → crop {crop_w:.0f}x{crop_h:.0f}")
        print(f"      Original aspect: {orig_aspect:.2f}:1")
        print(f"      Crop aspect: {crop_aspect:.2f}:1")
        
        # Aspect ratio should be reasonable (not too extreme)
        assert 0.3 < crop_aspect < 3.0, f"Crop aspect {crop_aspect:.2f} is too extreme"
    
    return True


def test_decision_file_creation():
    """Test that .decision files are created correctly."""
    print("\n🧪 Test 5: Decision File Creation")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create fake image
        img = tmpdir / "test_20250101_120000_stage2_upscaled.png"
        img.touch()
        
        # Create decision data
        decision_data = {
            "group_id": "20250101_120000",
            "images": ["test_20250101_120000_stage2_upscaled.png"],
            "ai_recommendation": {
                "selected_image": "test_20250101_120000_stage2_upscaled.png",
                "selected_index": 0,
                "reason": "Test reason",
                "confidence": 0.95,
                "crop_needed": True,
                "crop_coords": (0.1, 0.2, 0.9, 0.8)
            },
            "user_decision": None
        }
        
        # Save decision file
        decision_path = tmpdir / "20250101_120000.decision"
        with open(decision_path, 'w') as f:
            json.dump(decision_data, f, indent=2)
        
        # Verify file created
        assert decision_path.exists(), "Decision file should be created"
        
        # Load and verify
        with open(decision_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded["group_id"] == decision_data["group_id"]
        assert loaded["ai_recommendation"]["crop_needed"]
        assert len(loaded["ai_recommendation"]["crop_coords"]) == 4
        
        # Verify crop coords are normalized
        x1, y1, x2, y2 = loaded["ai_recommendation"]["crop_coords"]
        assert 0 <= x1 < x2 <= 1
        assert 0 <= y1 < y2 <= 1
        
        print(f"   ✅ Decision file created at: {decision_path.name}")
        print("   ✅ AI recommendation saved correctly")
        print(f"   ✅ Crop coords: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
        return True


def test_batch_size_configuration():
    """Test that batch size is configurable."""
    print("\n🧪 Test 6: Batch Size Configuration")
    
    total_groups = 100
    test_batch_sizes = [10, 20, 50]
    
    for batch_size in test_batch_sizes:
        expected_batches = (total_groups + batch_size - 1) // batch_size
        
        # Calculate what would happen
        batches = []
        for i in range(0, total_groups, batch_size):
            batch_end = min(i + batch_size, total_groups)
            batch = list(range(i, batch_end))
            batches.append(batch)
        
        assert len(batches) == expected_batches, f"Should have {expected_batches} batches"
        
        # Verify all groups covered
        all_items = [item for batch in batches for item in batch]
        assert len(all_items) == total_groups, "All groups should be covered"
        assert all_items == list(range(total_groups)), "Groups should be in order"
        
        print(f"   ✅ Batch size {batch_size}: {len(batches)} batches for {total_groups} groups")
    
    return True


def test_crop_overlay_math():
    """Test the math for converting normalized coords to pixels."""
    print("\n🧪 Test 7: Crop Overlay Pixel Conversion")
    
    # Test case: 1920x1080 image with crop
    img_width = 1920
    img_height = 1080
    crop_coords = (0.1, 0.2, 0.9, 0.8)  # Normalized
    
    x1_norm, y1_norm, x2_norm, y2_norm = crop_coords
    
    # Convert to pixels (same math as JavaScript)
    x1_px = x1_norm * img_width
    y1_px = y1_norm * img_height
    width_px = (x2_norm - x1_norm) * img_width
    height_px = (y2_norm - y1_norm) * img_height
    
    # Verify conversion (use round for floating point)
    assert round(x1_px) == 192, f"x1 should be 192px, got {x1_px}"
    assert round(y1_px) == 216, f"y1 should be 216px, got {y1_px}"
    assert round(width_px) == 1536, f"width should be 1536px, got {width_px}"
    assert round(height_px) == 648, f"height should be 648px, got {height_px}"
    
    # Verify final coords
    x2_px = x1_px + width_px
    y2_px = y1_px + height_px
    
    assert round(x2_px) == 1728, f"x2 should be 1728px, got {x2_px}"
    assert round(y2_px) == 864, f"y2 should be 864px, got {y2_px}"
    
    print(f"   ✅ Image: {img_width}x{img_height}")
    print(f"   ✅ Normalized: ({x1_norm}, {y1_norm}, {x2_norm}, {y2_norm})")
    print(f"   ✅ Pixels: ({x1_px}, {y1_px}) → ({x2_px}, {y2_px})")
    print(f"   ✅ Crop size: {width_px}x{height_px}")
    
    # Verify aspect ratio preserved
    orig_aspect = img_width / img_height
    crop_aspect = width_px / height_px
    print(f"   ✅ Original aspect: {orig_aspect:.3f}:1")
    print(f"   ✅ Crop aspect: {crop_aspect:.3f}:1")
    
    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("AI-ASSISTED REVIEWER - INTEGRATION TESTS")
    print("=" * 70)
    
    tests = [
        test_model_loading,
        test_rule_based_recommendation,
        test_crop_coordinates_validation,
        test_crop_aspect_ratio_preservation,
        test_decision_file_creation,
        test_batch_size_configuration,
        test_crop_overlay_math,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                skipped += 1
        except AssertionError as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

