# Productivity Overview Table
**Status:** Active
**Audience:** Developers

**Last Updated:** 2025-10-26


**Status**: ✅ Implemented  
**Date**: October 16, 2025  
**Location**: Dashboard (top of page, above detailed table)

## Overview

The **Productivity Overview** table provides a high-level summary of productivity metrics across all projects, focusing on images processed per hour (img/h) by tool and overall.

## Table Structure

```
⚡ Productivity Overview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Project          | Web Sel | Char Sort | Multi Crop | Overall 
                 | img/h   | img/h     | img/h      | img/h
────────────────────────────────────────────────────────────────────────
Mojo2            | 1448    | -         | -          | 1448
Mojo1            | 1048    | 930       | 183        | 886
jmlimages-random | 12178   | 3573      | 1544       | 5765
tattersail-0918  | 12031   | 8214      | -          | 10122
mixed-0919       | 15500   | -         | -          | 15500
```

## Columns

1. **Project**: Project name (matches detailed table order)
2. **Web Sel** (img/h): AI-Assisted Reviewer productivity
3. **Char Sort** (img/h): Character Sorter productivity  
4. **Multi Crop** (img/h): Multi Crop Tool productivity
5. **Overall** (img/h): **Combined productivity across all tools** (highlighted)

## Calculation Logic

### Per-Tool img/h
```python
tool_img_h = total_images_processed_by_tool / total_hours_worked_on_tool
```

### Overall img/h
```python
overall_img_h = sum(all_tool_images) / sum(all_tool_hours)
```

The **Overall** column gives you a single metric to compare projects:
- Takes ALL images processed across ALL tools
- Divides by ALL hours worked across ALL tools
- Shows combined productivity regardless of tool mix

## Use Cases

### 1. Quick Performance Comparison
**Question**: "Which project was I most productive on?"  
**Answer**: Look at the **Overall** column → highest img/h = best overall productivity

### 2. Tool-Specific Analysis
**Question**: "Am I faster at cropping or selecting?"  
**Answer**: Compare **Web Sel** vs **Multi Crop** columns across projects

### 3. Workflow Efficiency
**Question**: "Did adding character sorting slow me down?"  
**Answer**: Compare projects with vs without **Char Sort** data

### 4. Productivity Trends
**Question**: "Am I getting faster over time?"  
**Answer**: Look at **Overall** from oldest to newest projects

## Filtering

### Hide Empty Projects
The table includes a **"Hide empty projects"** checkbox (checked by default) that filters out:
- Projects with no tool data (all tools show "-")
- Projects with 0 overall img/h

This keeps the table clean and focused on projects with actual productivity data.

**To toggle:**
- ✅ Checked: Only show projects with data (default)
- ⬜ Unchecked: Show all projects (including empty ones)

The filter applies dynamically without reloading the page.

## Design Decisions

### Why "Overall" Instead of Average?
We **sum** all images and divide by **sum** of all hours (not average of per-tool img/h) because:
- More accurate representation of total project productivity
- Accounts for time distribution across tools
- Matches how you think: "I processed X images in Y hours"

### Example:
```
Web Selector:  10,000 images in 10 hours = 1000 img/h
Crop Tool:     5,000 images in 5 hours = 1000 img/h

❌ Average of img/h: (1000 + 1000) / 2 = 1000 img/h
✅ Overall img/h: 15,000 images / 15 hours = 1000 img/h

In this case they're the same, but if time distribution varies:

Web Selector:  10,000 images in 5 hours = 2000 img/h
Crop Tool:     5,000 images in 10 hours = 500 img/h

❌ Average: (2000 + 500) / 2 = 1250 img/h (misleading!)
✅ Overall: 15,000 / 15 = 1000 img/h (accurate)
```

## Implementation Details

### Backend (analytics.py)
- New method: `_build_productivity_overview_table()`
- Takes detailed table data as input
- Extracts img/h metrics for each tool
- Calculates overall img/h by summing images and hours
- Uses `STANDARD_TOOL_ORDER` for consistency

### Frontend (dashboard_template.html)
- New table: `<table id="productivity-overview-table">`
- New function: `updateProductivityOverview(data)`
- Renders in same project order as detailed table
- Uses alternating row colors for readability
- Highlights "Overall" column with accent color

### API Response
```json
{
  "productivity_overview": [
    {
      "projectId": "mojo2",
      "title": "Mojo2",
      "tool_metrics": {
        "AI-Assisted Reviewer": 1448,
        "Web Character Sorter": null,
        "Multi Crop Tool": null
      },
      "overall_img_h": 1448
    }
  ]
}
```

## Future Enhancements

Possible additions:
1. **Sort by column** - click header to sort by that tool's img/h
2. **Color coding** - green for high productivity, red for low
3. **Sparklines** - mini graphs showing productivity trend over project days
4. **Percentile indicators** - show if project is in top 25%, median, etc.
5. **Hide empty tools** - checkbox to hide columns with no data across all projects

## Related Documentation

- `IMAGE_PROCESSING_WORKFLOW.md` - Workflow context
- `../../reference/WEB_STYLE_GUIDE.md` - UI design patterns
- `CENTRALIZED_TOOL_ORDER.md` - Tool ordering consistency
- `HOUR_BLOCKING_SYSTEM.md` - How hours are calculated

