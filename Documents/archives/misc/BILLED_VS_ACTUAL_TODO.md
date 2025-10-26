# Billed vs Actual Chart Improvements - TODO List
**Audience:** Developers

**Last Updated:** 2025-10-26


**Date:** October 18, 2025  
**Status:** 📋 Planning Phase

---

## Requirements Summary

Improve the "Billed vs Actual" chart with:
- Time range selector buttons (15min, 1hour, daily, project)
- "Hide empty projects" checkbox
- Project count dropdown (1, 2, 3, 4, 5, All)
- Default to current + previous project only
- Data point capping to keep graph readable

---

## TODO List

### 1. Add Time Range Buttons (Top Right)
- [ ] Add button group: `1hour | daily | project`
- [ ] Position in top-right of "Billed vs Actual" section
- [ ] Default selection: **"project"** (totals)
- [ ] Wire up to filter data by time range
- [ ] Store selection in state (survives page refresh if needed)

**Notes:**
- ~~"15min"~~ = REMOVED (we don't have billed hours at 15-min granularity)
- "1hour" = Hourly breakdown
- "daily" = Daily breakdown
- "project" = Total per project (current behavior, DEFAULT)

---

### 2. Add "Hide Empty Projects" Checkbox
- [ ] Add checkbox below or near time range buttons
- [ ] Label: "Hide empty projects"
- [ ] When checked: Hide projects with 0 actual hours
- [ ] Default: Unchecked (show all)
- [ ] Save state across page loads (optional)

**Logic:**
- Empty = `actual_hours == 0` (no work tracked)
- Keep showing projects with billed hours even if actual = 0

---

### 3. Default to Current + Previous Project
- [ ] Identify "current project" (most recent by date or status=active)
- [ ] Identify "previous project" (second most recent)
- [ ] Default display: Show only these 2 projects
- [ ] Apply on initial page load

**Sorting logic:**
- Sort projects by `startedAt` date (newest first)
- Active projects first, then by date

---

### 4. Add Project Count Dropdown
- [ ] Add dropdown selector
- [ ] Options: `1 | 2 | 3 | 4 | 5 | All projects`
- [ ] Default: `2` (current + previous)
- [ ] Position near time range buttons
- [ ] Updates chart dynamically when changed

**Behavior:**
- "1" = Show current project only
- "2" = Show current + previous (default)
- "3" = Show 3 most recent projects
- etc.
- "All" = Show all projects

---

### 5. Cap Data Points Per Time Range
- [ ] Define max data points per time range to keep graph readable
- [ ] Implement logic to limit data based on selection
- [ ] Test different caps to find optimal values

**Caps to implement:**
- **1hour view:**
  - Cap at ~48 points (2 days worth)
  - Or: Last 24 hours = 24 points
  - Test to find readable limit
  
- **daily view:**
  - Cap at ~30 points (1 month)
  - Or: Last 14 days = 14 points
  
- **project view:**
  - No cap needed (already limited by project count dropdown)

**Implementation:**
- When time range < "project", filter to recent time window
- Show "Showing last N hours/days" indicator if capped

---

### 6. Backend Data Preparation
- [ ] Add endpoint or modify existing to support time range filtering
- [ ] Calculate work hours broken down by:
  - 15-minute bins (from our new bins system!)
  - Hourly aggregates
  - Daily aggregates
- [ ] Match with billed hours from timesheet
- [ ] Return data in format chart can consume

**Data structure needed:**
```json
{
  "time_range": "15min",
  "projects": [
    {
      "project_id": "mojo2",
      "project_name": "mojo-2",
      "data_points": [
        {
          "timestamp": "2025-10-17T14:00:00Z",
          "billed_hours": 0.25,
          "actual_hours": 0.4
        },
        ...
      ]
    }
  ]
}
```

---

### 7. Frontend Chart Updates
- [ ] Update chart rendering to handle time-series data (not just totals)
- [ ] X-axis: Time labels (15min, hourly, daily) or project names
- [ ] Y-axis: Hours
- [ ] Two series per project: Billed (blue?) and Actual (green?)
- [ ] Tooltip showing exact values
- [ ] Legend explaining colors

**Chart type: BAR GRAPH (all views)**
- Bar chart for all views (1hour, daily, project)
- Grouped bars (billed vs actual side-by-side)
- Standard for all dashboard graphs

---

### 8. UI/UX Polish
- [ ] Add loading indicator while data fetches
- [ ] Show "No data" message if empty
- [ ] Add help text/tooltip explaining the chart
- [ ] Ensure responsive layout (buttons don't overflow on mobile)
- [ ] Smooth transitions when changing views

---

## Implementation Order (Recommended)

**Phase 1: Project filtering (easiest)**
1. TODO #3: Default to current + previous
2. TODO #4: Project count dropdown
3. TODO #2: Hide empty projects checkbox

**Phase 2: Time range breakdown (medium)**
4. TODO #1: Time range buttons (UI only)
5. TODO #6: Backend data for time ranges
6. TODO #7: Frontend chart updates

**Phase 3: Polish (last)**
7. TODO #5: Data point capping
8. TODO #8: UI/UX polish

---

## ✅ Questions RESOLVED

1. **Default time range:** ✅ "project" (totals)
2. **Exact data point caps:** Test during Phase 3
3. **Chart type:** ✅ Bar graph (all views)
4. **Billed hours breakdown:** Even distribution across project work dates (to be implemented)

---

## Files to Modify

**Backend:**
- `scripts/dashboard/analytics.py` - Add time range filtering to `_build_billed_vs_actual`
- `scripts/dashboard/data_engine.py` - May need time-series work data

**Frontend:**
- `scripts/dashboard/dashboard_template.html` - Add UI controls
- `scripts/dashboard/productivity_dashboard.py` - Wire up new parameters

**Docs:**
- Update dashboard guide with new features

---

## Ready to Start?

Once you approve this TODO list, I'll start with **Phase 1** (project filtering) since it's the easiest and will give you immediate value!

Let me know if you want to adjust any of these items or change the order! 🚀

