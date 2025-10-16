Productivity Dashboard Specification
Charting Approach

Primary visualization type: Bar charts for all time-based breakdowns.

Overlay: A “cloud” (faded background shape) representing historical averages for the same time slice. This provides context without obscuring the current bar values.

Update markers: Visual cues (vertical line or icon) plotted on the chart timeline whenever a script is updated.

Time Scales

Intraday view

Bars represent fixed time slices (configurable: 15 minutes or 1 hour).

X-axis: clock time across the day.

Y-axis: either active minutes or files processed.

Overlay: average historical productivity for each time slice (the “cloud”).

Daily view

Bars represent total daily productivity.

X-axis: calendar days.

Y-axis: totals (active minutes, files processed, or both).

Overlay: average daily totals.

Update markers: script update events aligned to the day they occurred.

Weekly view

Bars represent one week of work.

X-axis: week number or start date of the week.

Y-axis: total active minutes/files processed for the week.

Overlay: average weekly totals.

Update markers: vertical lines when updates happen inside that week.

Monthly view

Bars represent one month of work.

X-axis: calendar months.

Y-axis: monthly totals.

Overlay: historical monthly average.

Update markers: script update events aligned to the month.

Pie chart

Displays share of time per tool/script.

Typically shown for monthly or cumulative periods.

Useful to see which tool consumes the largest share of overall time.

Script Update Tracking

Maintain a simple log (CSV or JSONL) of updates:

date, tool, description
2025-09-12, web_selector, "Added batch skipping"
2025-09-20, clustering, "Optimized embeddings"


Each update should be shown on relevant charts as a vertical marker (line or icon).

When hovering or clicking, the description should display.

This makes it possible to visually correlate productivity shifts with tool changes.

This spec ensures visual consistency (bar charts everywhere), contextual overlays (average clouds), and traceability (update markers) across all views.