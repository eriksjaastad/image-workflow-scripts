#!/usr/bin/env python3
"""
Tests for centralized sorting determinism (timestamp then stage).
"""

import sys
import tempfile
from pathlib import Path

# Add project root so utils.* imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.companion_file_utils import sort_image_files_by_timestamp_and_stage


def test_sorting_timestamp_then_stage():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # Mixed order creation; names encode timestamp+stage
        names = [
            "20250706_171804_stage2_upscaled.png",
            "20250706_171632_stage1_generated.png",
            "20250706_171948_stage3_enhanced.png",
            "20250706_171652_stage1.5_face_swapped.png",
        ]
        for n in names:
            (d / n).write_text("x")

        files = list(d.glob("*.png"))
        sorted_files = sort_image_files_by_timestamp_and_stage(files)
        assert [p.name for p in sorted_files] == [
            "20250706_171632_stage1_generated.png",
            "20250706_171652_stage1.5_face_swapped.png",
            "20250706_171804_stage2_upscaled.png",
            "20250706_171948_stage3_enhanced.png",
        ]
