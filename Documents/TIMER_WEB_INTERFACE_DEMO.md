---
title: Live Web Interface - Activity Timer Integration
status: Legacy
superseded_by: ACTIVITY_TIMER_SYSTEM_OVERVIEW.md
audience: USER, HISTORICAL
tags: [activity-timer, legacy, ui, analytics]
---

[LEGACY] This document describes the pre file-operation timing ActivityTimer UI. Kept for historical reference. See ACTIVITY_TIMER_SYSTEM_OVERVIEW.md for the current system.

# Live Web Interface - Activity Timer Integration

## 🖥️ What You'll See in Your Browser

### **Web Image Selector with Live Timer**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Image Version Selector - Batch 3/12                    ⏱️ Activity Timer    │
│                                                        Active: 47m 23s      │
│ ┌─ Group 201 ────────────────────────────────────────┐ Total: 52m 18s      │
│ │ [IMG1] [IMG2] [IMG3] ← Click to select            │ Efficiency: 90.6%    │
│ │  📸     📸     📸                                   │ 🟢 Active            │
│ └────────────────────────────────────────────────────┘                      │
│                                                                             │
│ ┌─ Group 202 ────────────────────────────────────────┐                      │
│ │ [IMG1] [IMG2] [IMG3] ← Selected: IMG2              │                      │
│ │  📸    [🔵]    📸                                   │                      │
│ └────────────────────────────────────────────────────┘                      │
│                                                                             │
│ ┌─ Group 203 ────────────────────────────────────────┐                      │
│ │ [IMG1] [IMG2] [IMG3]                               │                      │
│ │  📸     📸     📸                                   │                      │
│ └────────────────────────────────────────────────────┘                      │
│                                                                             │
│ Progress: 23/100 groups selected                                            │
│ [Process Batch] [Send to crop/] ☐                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **Timer Widget Behavior**

**🟢 Active State (User is working):**
```
┌─────────────────────────────┐
│ ⏱️ Activity Timer            │
│ Active: 47m 23s             │  ← Counting up while working
│ Total: 52m 18s              │  ← Total session time
│ Efficiency: 90.6%           │  ← Active/Total ratio
│ 🟢 Active                   │  ← Green when working
└─────────────────────────────┘
```

**🔴 Idle State (User stepped away):**
```
┌─────────────────────────────┐
│ ⏱️ Activity Timer            │
│ Active: 47m 23s             │  ← Stopped counting
│ Total: 55m 43s              │  ← Still counting total
│ Efficiency: 85.1%           │  ← Efficiency drops
│ 🔴 Idle (3m 25s)           │  ← Red with idle time
└─────────────────────────────┘
```

## 📊 Real-Time Data Collection

### **Every User Action Tracked:**
```
User clicks image → mark_activity() → Timer updates → Live display refreshes
     ↓                    ↓               ↓              ↓
Mouse movement    →  Activity detected → Active time++ → Efficiency recalc
Keyboard press    →  Reset idle timer  → Status: Active → UI updates
Scroll page       →  Track engagement  → Log interaction → Stats refresh
```

### **Background Monitoring:**
```
Every 30 seconds:
├─ Check if user has been idle > 10 minutes
├─ If idle: Stop active timer, mark as idle
├─ If active: Continue counting active time
└─ Update efficiency calculation

Every 5 seconds (in browser):
├─ Fetch /timer_stats from server
├─ Update timer display with latest data
├─ Show current active/total/efficiency
└─ Update status indicator (🟢/🔴)
```

## 🎯 What Each Metric Means

### **Active Time**
- **What it tracks:** Time when you're actually clicking, typing, scrolling
- **Why it matters:** Your real productive work time
- **Example:** "47m 23s" = You've been actively working for 47 minutes

### **Total Time** 
- **What it tracks:** Total session duration (including breaks)
- **Why it matters:** Shows how long you've had the tool open
- **Example:** "52m 18s" = Tool has been running for 52 minutes

### **Efficiency**
- **What it tracks:** Active time ÷ Total time × 100
- **Why it matters:** Shows how focused your work session is
- **Example:** "90.6%" = Very focused session, minimal breaks

### **Status Indicator**
- **🟢 Active:** You've interacted within last 10 minutes
- **🔴 Idle:** No interaction for 10+ minutes
- **Why it matters:** Instant feedback on your work state

## 📈 Live Analytics Dashboard

### **Command Line Reports (Available Anytime):**

```bash
# Quick daily summary
$ python scripts/util_timer_report.py
📊 Activity Timer Report
========================================
Total Active Time: 4h 23m
Total Session Time: 5h 12m  
Work Efficiency: 84.2%
Files Processed: 1,247
Sessions: 3

# Live monitoring (updates every 5 seconds)
$ python scripts/util_timer_report.py --live
🔴 Live Session Monitor
Updated: 14:23:45
========================================
📱 01_web_image_selector
   Duration: 47m 23s
   Active: 41m 12s (87.1% efficiency)
   Files Processed: 234
   Current Rate: 341 files/hour
   Status: 🟢 Active

# Weekly productivity analysis  
$ python scripts/util_timer_report.py --productivity
📈 Productivity Metrics - Last 7 Days
============================================================
Overall Performance:
  Total Active Hours: 28.2h
  Files per Hour: 156.3
  Average Efficiency: 84.7%

📋 Script Performance Ranking:
  1. 01_web_image_selector: 156.3 files/hour, 87.2% efficiency
  2. 04_batch_crop_tool: 142.7 files/hour, 83.8% efficiency
  3. 03_web_character_sorter: 198.4 files/hour, 79.1% efficiency
```

## 🔗 Integration with FileTracker

### **Dual Logging System:**

**When you process a batch:**
```
FileTracker logs:                ActivityTimer logs:
├─ Files moved: 127             ├─ Batch: "Processing batch 3"
├─ Files deleted: 43            ├─ Duration: 8m 27s  
├─ Source: XXX_CONTENT/         ├─ Active time: 7m 12s
├─ Dest: Reviewed/              ├─ Efficiency: 85.2%
└─ Timestamp: 14:23:15          └─ Operations: select(127), delete(43)

Combined Analysis:
┌─────────────────────────────────────────────────────┐
│ Batch Performance: 127 files in 7m 12s             │
│ Rate: 17.6 files/minute (above average)            │
│ Efficiency: 85.2% (good focus)                     │
│ Keep ratio: 74.7% (127/170 files)                  │
│ Quality: High performance session                   │
└─────────────────────────────────────────────────────┘
```

## 🎮 Interactive Features

### **Automatic Activity Detection:**
- **Mouse clicks** → Instant activity marker
- **Keyboard input** → Reset idle timer  
- **Page scrolling** → Track engagement
- **Mouse movement** → Throttled activity (every 2 seconds)

### **Smart Idle Detection:**
- **< 10 minutes idle** → Still considered active
- **> 10 minutes idle** → Automatically pause active timer
- **Return to work** → Resume message: "Resumed after 12m 34s idle"

### **Batch Tracking:**
- **Automatic batch detection** → When you click "Process Batch"
- **Manual batch markers** → For custom workflow segments
- **Batch performance** → Individual batch timing and efficiency

## 🚀 Actionable Insights You'll Get

### **Immediate Feedback (While Working):**
1. **"Am I in the zone?"** → Check efficiency % in real-time
2. **"How long have I been working?"** → See active time counter
3. **"Should I take a break?"** → Monitor efficiency drops

### **Session Analysis (After Working):**
1. **"How productive was I today?"** → Daily efficiency report
2. **"Which tool works best for me?"** → Cross-script comparison
3. **"When am I most productive?"** → Time-of-day analysis

### **Long-term Optimization (Weekly/Monthly):**
1. **"What's my peak performance time?"** → Hour-by-hour efficiency
2. **"How can I improve my workflow?"** → Bottleneck identification  
3. **"What batch size works best?"** → Optimal batch size analysis

### **Data-Driven Decisions:**
```
Example Insights:
├─ "You're 23% more efficient in morning sessions (9-11am)"
├─ "Batch size 75-100 gives you best files/hour rate"  
├─ "04_batch_crop_tool efficiency drops after 90 minutes"
├─ "Your keep rate is 8% higher when efficiency > 85%"
└─ "Peak productivity: Tuesday 2-4pm (avg 187 files/hour)"
```

This system transforms your workflow from guesswork to data-driven optimization! 📊✨

## 🎯 Ready to Use Right Now!

The timer is **already integrated** into `01_web_image_selector.py`. Just run it normally:

```bash
python scripts/01_web_image_selector.py XXX_CONTENT/
```

You'll immediately see:
- ✅ Live timer widget in top-right corner
- ✅ Real-time efficiency tracking  
- ✅ Automatic activity detection
- ✅ Session data being collected

Then check your analytics:
```bash
python scripts/util_timer_report.py --daily
```

**No setup required - it just works!** 🚀
