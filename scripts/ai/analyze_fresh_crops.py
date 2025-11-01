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


# Calculate metrics
crop_widths = [(x2 - x1) * 100 for x1, y1, x2, y2 in crops]
crop_heights = [(y2 - y1) * 100 for x1, y1, x2, y2 in crops]
crop_areas = [(x2 - x1) * (y2 - y1) * 100 for x1, y1, x2, y2 in crops]

x1_positions = [x1 for x1, y1, x2, y2 in crops]
y1_positions = [y1 for x1, y1, x2, y2 in crops]

x_centers = [(x1 + x2) / 2 for x1, y1, x2, y2 in crops]
y_centers = [(y1 + y2) / 2 for x1, y1, x2, y2 in crops]



width_cv = statistics.stdev(crop_widths) / statistics.mean(crop_widths)
height_cv = statistics.stdev(crop_heights) / statistics.mean(crop_heights)
x_std = statistics.stdev(x1_positions)
y_std = statistics.stdev(y1_positions)


# Compare with earlier "uniform" data

if width_cv > 0.20 or height_cv > 0.20 or x_std > 0.12 or y_std > 0.04:
    pass
else:
    pass

for _i, (x1, y1, x2, y2) in enumerate(crops, 1):
    w = (x2 - x1) * 100
    h = (y2 - y1) * 100


