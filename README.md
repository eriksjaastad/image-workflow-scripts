# Image Processing Workflow Scripts

A comprehensive set of Python scripts for processing and organizing large collections of AI-generated images through a structured workflow.

## Overview

This toolkit provides a complete workflow for processing thousands of images through quality filtering, face grouping, character sorting, pair comparison, and final cropping/organization.

## Scripts

### Core Workflow Scripts

1. **`01_image_version_selector.py`** - Select best versions from image triplets
2. **`02_face_grouper.py`** - Group similar faces using AI face recognition  
3. **`03_character_sorter.py`** - Sort images by character types or body groups
4. **`04_pair_compare.py`** - Side-by-side comparison for duplicate elimination
5. **`05_crop_tool.py`** - Interactive cropping with aspect ratio control

### Utility Scripts

- **`file_tracker.py`** - Comprehensive file operation logging and tracking
- Additional utility scripts for specialized processing tasks

## Setup

### Prerequisites

- Python 3.8+
- Virtual environment support

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd "Eros Mate"
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Workflow

1. **Start with image selection:**
```bash
python scripts/01_image_version_selector.py <source_directory>
```

2. **Optional face grouping:**
```bash
python scripts/02_face_grouper.py "Reviewed"
```

3. **Optional character sorting:**
```bash
python scripts/03_character_sorter.py <directory>
```

4. **Pair comparison for final selection:**
```bash
python scripts/04_pair_compare.py <directory>
```

5. **Final cropping and organization:**
```bash
python scripts/05_crop_tool.py <directory>
```

### File Tracking

All scripts include comprehensive file operation tracking via `FileTracker`. Logs are automatically managed with daily cleanup and backup.

## Features

- **Scalable processing** - Handle thousands of images efficiently
- **AI-powered face recognition** - Group similar faces automatically
- **Interactive tools** - User-friendly interfaces for manual review
- **Complete file tracking** - Audit trail of all file operations
- **Flexible workflow** - Skip steps as needed for different image batches
- **Aspect ratio control** - Maintain or customize image proportions

## Directory Structure

The workflow creates and manages several working directories:
- `Reviewed/` - Selected best versions
- `face_group_*/` - Grouped similar faces
- `character_group_*/` - Sorted character types
- `crop/` - Images needing cropping review
- `cropped/` - Final processed images

## Requirements

See `requirements.txt` for complete dependency list. Key packages include:
- OpenCV for image processing
- DeepFace for face recognition
- scikit-learn for clustering
- matplotlib for interactive interfaces
- Pillow for image manipulation

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
