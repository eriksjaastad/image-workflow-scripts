#!/usr/bin/env python3
"""Calculate realistic project timeline based on historical cropping patterns."""

# Historical data from analysis
TOTAL_IMAGES = 19406
MEDIAN_CROPS_PER_HOUR = 259.3
AVG_SESSION_LENGTH_MIN = 44.0
AVG_CROPS_PER_SESSION = 205.5
MEDIAN_BREAK_MIN = 20.5

# Calculations
print(f"\n{'='*80}")
print(f"MOJO3 PROJECT TIMELINE ESTIMATE")
print(f"{'='*80}\n")

print(f"Total images to crop: {TOTAL_IMAGES:,}")
print(f"\nBased on your historical patterns:")
print(f"  Median speed: {MEDIAN_CROPS_PER_HOUR:.1f} crops/hour")
print(f"  Average session: {AVG_SESSION_LENGTH_MIN:.1f} minutes, {AVG_CROPS_PER_SESSION:.1f} crops")
print(f"  Median break: {MEDIAN_BREAK_MIN:.1f} minutes")

# Method 1: Simple calculation
pure_crop_hours = TOTAL_IMAGES / MEDIAN_CROPS_PER_HOUR
print(f"\n{'='*80}")
print(f"METHOD 1: Pure cropping time")
print(f"{'='*80}")
print(f"  Time needed: {pure_crop_hours:.1f} hours ({pure_crop_hours/24:.1f} days of 24/7 work)")

# Method 2: Session-based calculation (more realistic)
sessions_needed = TOTAL_IMAGES / AVG_CROPS_PER_SESSION
total_session_time_min = sessions_needed * AVG_SESSION_LENGTH_MIN
total_session_time_hr = total_session_time_min / 60
print(f"\n{'='*80}")
print(f"METHOD 2: Session-based estimate (more realistic)")
print(f"{'='*80}")
print(f"  Sessions needed: {sessions_needed:.1f}")
print(f"  Active cropping time: {total_session_time_hr:.1f} hours")

# Add breaks
num_breaks = sessions_needed - 1  # n-1 breaks for n sessions
total_break_time_min = num_breaks * MEDIAN_BREAK_MIN
total_break_time_hr = total_break_time_min / 60
total_time_with_breaks = total_session_time_hr + total_break_time_hr

print(f"  Break time: {total_break_time_hr:.1f} hours ({num_breaks:.0f} breaks)")
print(f"  TOTAL time: {total_time_with_breaks:.1f} hours")

# Realistic daily work estimates
print(f"\n{'='*80}")
print(f"REALISTIC TIMELINE (with breaks)")
print(f"{'='*80}")

for hours_per_day in [4, 6, 8]:
    days_needed = total_time_with_breaks / hours_per_day
    print(f"  At {hours_per_day} hours/day: {days_needed:.1f} days ({days_needed/7:.1f} weeks)")

# Speed improvement scenarios
print(f"\n{'='*80}")
print(f"QUEUE MODE SPEED BOOST")
print(f"{'='*80}")
print(f"\nCurrent median time between crops: 0.31 seconds (batch submit time)")
print(f"If you eliminate processing wait time, you could potentially:")

for boost_pct in [25, 50, 100]:
    boosted_rate = MEDIAN_CROPS_PER_HOUR * (1 + boost_pct/100)
    boosted_hours = TOTAL_IMAGES / boosted_rate
    improvement = pure_crop_hours - boosted_hours
    print(f"  {boost_pct:3d}% faster: {boosted_rate:6.1f} crops/hr → {boosted_hours:5.1f} hours (save {improvement:4.1f} hours)")

print(f"\n{'='*80}\n")
