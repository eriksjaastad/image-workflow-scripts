# Crop Progress Tracking System Design

## Overview
A systematic progress tracking system for the multi-crop tool that allows users to resume cropping sessions exactly where they left off, with independent tracking per directory.

## Problem Statement
When processing large image directories (100-1000+ images), users need to:
- Resume cropping sessions after breaks (hours/days later)
- Track progress across multiple directories independently
- Avoid re-processing already completed images
- Maintain clean deliverable directories (no tracking files mixed with client data)

## Solution Architecture

### Progress File Storage
```
scripts/crop_progress/
├── face_groups_person_0001_progress.json
├── character_group_1_progress.json
├── _asian_progress.json
├── _black_progress.json
├── _sort_again_progress.json
└── [directory_name]_progress.json
```

### Progress File Format
```json
{
  "directory": "_asian/",
  "directory_hash": "abc123...",
  "total_images": 150,
  "completed_count": 73,
  "current_index": 73,
  "last_processed_file": "20250803_065656_stage2_upscaled.png",
  "last_session_start": "2025-09-25T14:30:00Z",
  "last_session_end": "2025-09-25T16:45:00Z",
  "session_count": 3,
  "completed_files": [
    "20250803_065656_stage2_upscaled.png",
    "20250803_070037_stage2_upscaled.png",
    "..."
  ],
  "skipped_files": [
    "corrupted_image.png"
  ],
  "output_directory": "cropped/",
  "created": "2025-09-25T10:00:00Z",
  "modified": "2025-09-25T16:45:00Z"
}
```

## User Experience Flow

### Starting New Directory
```bash
$ python scripts/04_multi_crop_tool.py _asian/
🎯 Starting crop session for '_asian/'
📊 Found 150 images to process
🚀 Beginning at image 1/150...
```

### Resuming Existing Directory
```bash
$ python scripts/04_multi_crop_tool.py _asian/
🎯 Resuming crop session for '_asian/'
📊 Progress: 73/150 images completed (48.7%)
⏰ Last session: 2 hours ago
🚀 Resuming at image 74/150...
```

### Progress Display
```
Processing: _asian/ [████████░░] 80% (120/150)
Current: 20250803_172343_stage2_upscaled.png
Session: 47 images processed, 2h 15m work time
```

## Implementation Details

### Progress File Management
- **File naming**: `{sanitized_directory_name}_progress.json`
- **Directory sanitization**: Replace `/`, `\`, spaces with underscores
- **Atomic writes**: Use temp file + rename to prevent corruption
- **Backup**: Keep `.bak` file of previous session

### Resume Logic
1. **Scan target directory** for all image files
2. **Load progress file** if exists
3. **Validate consistency** (file count, directory hash)
4. **Filter completed files** from processing queue
5. **Start from next unprocessed** image

### Progress Updates
- **Real-time updates**: Save progress every 10 images processed
- **Session tracking**: Record start/end times, image counts
- **Error handling**: Track skipped/failed images separately
- **Clean exit**: Save final progress on Ctrl+C or completion

### Integration Points
- **FileTracker**: Log all crop operations for audit trail
- **File-Operation Timing**: Track work time based on file operations
- **Multi-Crop Tool**: Modify to use progress system
- **Base Desktop Tool**: Inherit progress tracking from base class

## File System Safety

### Client Directory Protection
- **No tracking files** in work directories (`_asian/`, `character_group_1/`, etc.)
- **All progress data** stored in `scripts/crop_progress/`
- **Clean deliverables** - only images and companion metadata (YAML and/or caption files) sent to client

### Data Integrity
- **Directory hashing**: Detect if source directory changes
- **File validation**: Verify completed files still exist
- **Corruption recovery**: Rebuild progress from output directory if needed
- **Version compatibility**: Handle progress file format changes

## Command Line Interface

### New Flags
```bash
--resume          # Resume from last position (default: true)
--reset-progress  # Start fresh, ignore existing progress
--show-progress   # Display current progress and exit
--progress-dir    # Custom progress file location
```

### Examples
```bash
# Normal usage (auto-resume)
python scripts/04_multi_crop_tool.py _asian/

# Start fresh
python scripts/04_multi_crop_tool.py _asian/ --reset-progress

# Check status
python scripts/04_multi_crop_tool.py _asian/ --show-progress
```

## Benefits

### Productivity
- **No lost work** - always resume exactly where you left off
- **Flexible scheduling** - work in chunks across multiple days
- **Multi-directory workflow** - switch between directories without confusion
- **Visual progress** - always know how much work remains

### Data Safety
- **Clean deliverables** - no tracking files mixed with client data
- **Audit trail** - complete record of all processing activities
- **Error recovery** - handle interruptions gracefully
- **Backup protection** - progress files backed up automatically

### Workflow Integration
- **Systematic processing** - supports methodical directory-by-directory approach
- **File-operation timing** - integrates with new intelligent timing system
- **File logging** - works with FileTracker for complete audit trail
- **Scalable** - handles directories from 10 to 10,000+ images

## Future Enhancements

### Advanced Features
- **Batch statistics** - processing speed, time estimates
- **Progress synchronization** - share progress across multiple machines
- **Smart scheduling** - suggest optimal break points
- **Quality metrics** - track crop quality scores per session

### Web Interface
- **Progress dashboard** - visual overview of all directories
- **Remote monitoring** - check progress from mobile device
- **Batch planning** - estimate time requirements for large jobs
- **Historical analysis** - productivity trends over time

## Implementation Priority
1. **Core progress tracking** - basic save/resume functionality
2. **CLI integration** - modify multi-crop tool to use progress system
3. **Error handling** - robust recovery from interruptions
4. **Progress display** - visual feedback during processing
5. **Advanced features** - statistics, web interface, etc.

---

*Last Updated: October 3, 2025*
*This document reflects the current multi-crop tool architecture and file-operation-based timing system.*

This system transforms the crop tool from a "start over every time" process into a professional, resumable workflow that scales with large image processing jobs.
