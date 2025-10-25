#!/usr/bin/env python3
"""Quick analysis of Erik's 14 fresh crops from today."""

import statistics

# Fresh crop data (normalized coordinates: x1, y1, x2, y2)
crops = [
    (0.0, 0.015625, 0.689453125, 0.705078125),
    (0.2958984375, 0.055013020833333336, 0.9996744791666666, 0.7587890625),
    (0.005208333333333333, 0.0771484375, 0.5244140625, 0.5963541666666666),
    (0.0, 0.047200520833333336, 0.716796875, 0.7639973958333334),
    (0.3229166666666667, 0.033203125, 0.9886067708333334, 0.6988932291666666),
    (0.1748046875, 0.0, 0.8421223958333334, 0.6673177083333334),
    (0.2864583333333333, 0.009440104166666666, 0.9983723958333334, 0.7213541666666666),
    (0.0, 0.12239583333333333, 0.6399739583333334, 0.7623697916666666),
    (0.0, 0.022786458333333332, 0.5764973958333334, 0.5992838541666666),
    (0.2177734375, 0.012369791666666666, 0.9150390625, 0.7096354166666666),
    (0.0035807291666666665, 0.064453125, 0.6627604166666666, 0.7236328125),
    (0.0556640625, 0.0, 0.9436848958333334, 0.8880208333333334),
    (0.005208333333333333, 0.1240234375, 0.5159505208333334, 0.634765625),
    (0.0, 0.048828125, 0.7356770833333334, 0.7845052083333334),
]

print("=" * 80)
print("ERIK'S 14 FRESH CROPS - DIVERSITY ANALYSIS")
print("=" * 80)
print()

# Calculate metrics
crop_widths = [(x2 - x1) * 100 for x1, y1, x2, y2 in crops]
crop_heights = [(y2 - y1) * 100 for x1, y1, x2, y2 in crops]
crop_areas = [(x2 - x1) * (y2 - y1) * 100 for x1, y1, x2, y2 in crops]

x1_positions = [x1 for x1, y1, x2, y2 in crops]
y1_positions = [y1 for x1, y1, x2, y2 in crops]

x_centers = [(x1 + x2) / 2 for x1, y1, x2, y2 in crops]
y_centers = [(y1 + y2) / 2 for x1, y1, x2, y2 in crops]

print("📊 CROP SIZE DISTRIBUTION")
print("-" * 80)
print(f"Crop Width (% of original):")
print(f"  Min:    {min(crop_widths):.1f}%")
print(f"  Max:    {max(crop_widths):.1f}%")
print(f"  Mean:   {statistics.mean(crop_widths):.1f}%")
print(f"  StdDev: {statistics.stdev(crop_widths):.1f}%")
print()
print(f"Crop Height (% of original):")
print(f"  Min:    {min(crop_heights):.1f}%")
print(f"  Max:    {max(crop_heights):.1f}%")
print(f"  Mean:   {statistics.mean(crop_heights):.1f}%")
print(f"  StdDev: {statistics.stdev(crop_heights):.1f}%")
print()
print(f"Crop Area (% of original):")
print(f"  Min:    {min(crop_areas):.1f}%")
print(f"  Max:    {max(crop_areas):.1f}%")
print(f"  Mean:   {statistics.mean(crop_areas):.1f}%")
print(f"  StdDev: {statistics.stdev(crop_areas):.1f}%")
print()

print("📍 CROP POSITION DISTRIBUTION")
print("-" * 80)
print(f"Top-Left X Position (normalized [0,1]):")
print(f"  Min:    {min(x1_positions):.3f}")
print(f"  Max:    {max(x1_positions):.3f}")
print(f"  Mean:   {statistics.mean(x1_positions):.3f}")
print(f"  StdDev: {statistics.stdev(x1_positions):.3f}")
print()
print(f"Top-Left Y Position (normalized [0,1]):")
print(f"  Min:    {min(y1_positions):.3f}")
print(f"  Max:    {max(y1_positions):.3f}")
print(f"  Mean:   {statistics.mean(y1_positions):.3f}")
print(f"  StdDev: {statistics.stdev(y1_positions):.3f}")
print()

print("🔍 DIVERSITY COMPARISON")
print("-" * 80)
width_cv = statistics.stdev(crop_widths) / statistics.mean(crop_widths)
height_cv = statistics.stdev(crop_heights) / statistics.mean(crop_heights)
x_std = statistics.stdev(x1_positions)
y_std = statistics.stdev(y1_positions)

print(f"Width Coefficient of Variation:  {width_cv:.3f}")
print(f"Height Coefficient of Variation: {height_cv:.3f}")
print(f"X Position Std Dev:              {x_std:.3f}")
print(f"Y Position Std Dev:              {y_std:.3f}")
print()

# Compare with earlier "uniform" data
print("📊 COMPARISON WITH EARLIER DATA (7,194 crops)")
print("-" * 80)
print("Metric                          | Earlier Data | Your 14 Crops")
print("--------------------------------|--------------|---------------")
print(f"Width CV (higher = more varied) | 0.176        | {width_cv:.3f}")
print(f"Height CV                       | 0.176        | {height_cv:.3f}")
print(f"X Position StdDev               | 0.101        | {x_std:.3f}")
print(f"Y Position StdDev               | 0.050        | {y_std:.3f}")
print()

if width_cv > 0.20 or height_cv > 0.20 or x_std > 0.12 or y_std > 0.04:
    print("✅ YOUR CROPS ARE MORE DIVERSE than the historical average!")
    print("   → Great training data for teaching content-aware cropping!")
else:
    print("⚠️  Your crops follow similar patterns to historical data")
    print("   → This is normal if you're cropping similar types of images")

print()
print("📋 INDIVIDUAL CROPS")
print("-" * 80)
for i, (x1, y1, x2, y2) in enumerate(crops, 1):
    w = (x2 - x1) * 100
    h = (y2 - y1) * 100
    print(f"{i:2d}. Size: {w:5.1f}%w × {h:5.1f}%h | Start: ({x1:.3f}, {y1:.3f})")

print()
print("=" * 80)

