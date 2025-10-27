#!/usr/bin/env python3
"""
Tests for flexible panel behavior in BaseDesktopImageTool.

These tests avoid calling any blocking UI (no plt.show()). They verify that
the base tool can render exactly 1 panel when requested, and that the
hide_unused_subplots logic respects the configured min/max bounds.
"""


import matplotlib


def test_base_tool_allows_single_panel(tmp_path):
    # Force a non-interactive backend regardless of environment
    try:
        matplotlib.use('Agg', force=True)
    except Exception:
        pass

    # Import after backend set
    from utils.base_desktop_image_tool import BaseDesktopImageTool  # type: ignore

    class _Probe(BaseDesktopImageTool):
        def submit_batch(self):
            pass

        def go_back(self):
            pass

        def run(self):
            pass

    tool = _Probe(directory=tmp_path, aspect_ratio=None, tool_name="probe")
    # Configure bounds: allow exactly 1 panel
    tool.min_panels = 1
    tool.max_panels = 3

    # Reconfigure display down to a single panel
    tool.setup_display(1)

    assert getattr(tool, 'current_num_images', None) == 1
    assert len(tool.axes) == 1
    assert tool.axes[0].get_visible() is True


def test_hide_unused_respects_bounds(tmp_path):
    try:
        matplotlib.use('Agg', force=True)
    except Exception:
        pass

    from utils.base_desktop_image_tool import BaseDesktopImageTool  # type: ignore

    class _Probe(BaseDesktopImageTool):
        def submit_batch(self):
            pass

        def go_back(self):
            pass

        def run(self):
            pass

    tool = _Probe(directory=tmp_path, aspect_ratio=None, tool_name="probe")
    tool.min_panels = 1
    tool.max_panels = 3

    # Start with 3 panels
    tool.setup_display(3)
    assert len(tool.axes) == 3

    # Reduce to 1 via hide_unused_subplots
    tool.hide_unused_subplots(1)
    assert getattr(tool, 'current_num_images', None) == 1
    assert len(tool.axes) == 1
    assert tool.axes[0].get_visible() is True


