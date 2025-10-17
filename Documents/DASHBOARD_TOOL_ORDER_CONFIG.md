# Centralized Tool Order

**Date:** October 15, 2025

## Overview

All dashboard charts, tables, and toggles now use a single centralized tool order defined in the backend. This ensures consistency across the entire dashboard interface.

## Standard Tool Order

```python
STANDARD_TOOL_ORDER = [
    'Desktop Image Selector Crop',
    'Web Image Selector',
    'Web Character Sorter',
    'Multi Crop Tool'
]
```

This order is defined in `scripts/dashboard/analytics.py` and exported via the API metadata.

## Where This Order is Used

### Backend (`analytics.py`)
- ✅ `STANDARD_TOOL_ORDER` constant defined at module level
- ✅ Used in `_build_tools_breakdown_for_project()`
- ✅ Exported in API response metadata: `metadata.standard_tool_order`

### Frontend (`dashboard_template.html`)
- ✅ **Productivity Table**: Tool columns rendered in standard order
- ✅ **Tool Column Toggles**: Checkboxes rendered in standard order
- ✅ **Files Processed by Tool Chart**: Tool toggles in standard order
- ✅ **Chart Series**: All timeseries charts use standard order

## Default Visibility

Tools shown by default (checkboxes checked):
- ✅ Web Image Selector
- ✅ Web Character Sorter (newly added to table)
- ✅ Multi Crop Tool

Tools hidden by default (checkboxes unchecked):
- ⬜ Desktop Image Selector Crop (less commonly used)

## Benefits

1. **Consistency**: Same order everywhere (tables, charts, toggles)
2. **Maintainability**: Single source of truth - change once, updates everywhere
3. **Extensibility**: Easy to add new tools - just update `STANDARD_TOOL_ORDER`
4. **User Experience**: Predictable layout across all dashboard views

## Adding New Tools

To add a new tool to the dashboard:

1. Update `STANDARD_TOOL_ORDER` in `scripts/dashboard/analytics.py`
2. Ensure tool name matches FileTracker script names
3. Add default visibility to `defaultVisibility` object in template
4. No other changes needed - all views will automatically include it!

## Technical Details

**API Contract:**
```json
{
  "metadata": {
    "standard_tool_order": [
      "Desktop Image Selector Crop",
      "Web Image Selector",
      "Web Character Sorter",
      "Multi Crop Tool"
    ]
  }
}
```

**Frontend Usage:**
```javascript
// All views now use:
const toolOrder = data.metadata?.standard_tool_order || [...fallback...];
```

---

**Related Files:**
- `scripts/dashboard/analytics.py`
- `scripts/dashboard/dashboard_template.html`
- `Documents/TECHNICAL_KNOWLEDGE_BASE.md`

