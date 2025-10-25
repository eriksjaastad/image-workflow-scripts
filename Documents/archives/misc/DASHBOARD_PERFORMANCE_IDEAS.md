# Dashboard Performance Ideas & Issues
**Date:** October 17, 2025

## Current Problem
- Dashboard still running **insanely slowly** even after bins implementation
- Demo bin system ran **crazy fast**
- Dashboard load time still very long
- Concern: Adding another month of projects will make this **completely unmanageable**

## Performance Investigation Needed
- **Where is the bottleneck?** Need detailed timing logs
- Archive data might be too dense
- Maybe 15-minute chunks need to be combined differently
- Need to understand what's taking so long

## UI/UX Improvement Ideas

### Chart Data Density
- **Problem:** Can't fit all projects on graphs in current display
- **Possible solution:** Default chart layout showing only:
  - Current project
  - Last project
  - Some UI to expand/show other projects
  
### Time Frames
- Remove global time selector at top of page?
- Add time frames to **each individual chart**
- Every chart defaults to hourly or daily view
- Most charts only show: Last project vs Current project

### Billed vs Actual Chart
- Want **hourly or daily breakdown** of work vs billed hours
- **Hide projects with no actual hours** (don't show if we don't have data)
- Add checkbox: "Hide empty projects" (like we do on other charts)

### Last 24 Hours View
- Currently don't have any charts that show just last 24 hours
- Need this as a quick view option

## Questions to Answer
1. Where exactly is the bottleneck? (Need timing breakdown)
2. Is bins system actually enabled/working?
3. Is the issue with loading all projects at once?
4. Are we still reading archive data inefficiently?
5. Should we implement chart-level lazy loading?

## Next Steps
1. Add detailed timing logs to dashboard
2. Run dashboard and capture timing breakdown
3. Identify actual bottleneck
4. Determine if bins are actually being used
5. Plan performance fixes based on real data

