# YOLO Dataset Generation Tool

A comprehensive tool for generating YOLO segmentation datasets using physics simulation with the Galbot robot. 

## Overview

This tool simulates random placement of objects in a physical environment and uses the Galbot robot head camera to obtain RGB images and labels. It supports both single-object and multi-object scenarios.

## Key Features

- **Physics-based Simulation**: Realistic object placement using MuJoCo physics engine
- **Multi-object Support**: Generate datasets with 2-6 objects simultaneously
- **Collision Avoidance**: Intelligent placement to prevent object overlapping
- **Flexible Configuration**: Customizable object types, positions, and orientations
- **YOLO Format Output**: Direct compatibility with YOLO training pipelines

## Architecture

### Supported Objects
- `power_drill`
- `cube`
- `mug`
- `bin`
- `extrusion`
- `toy`

### Dataset Structure

Each object (including multi-mode) collects 30 images total, distributed across train, test, and validation sets in a 7:2:1 ratio:

- **Training Set (Train)**: 21 images
- **Test Set (Test)**: 6 images 
- **Validation Set (Val)**: 3 images 

```
dataset/
├── images/
│   ├── train/
│   │   ├── bin_01.jpg
│   │   ├── bin_02.jpg
│   │   ├── ...
│   │   ├── power_drill_01.jpg
│   │   ├── power_drill_02.jpg
│   │   ├── ...
│   │   ├── multi_01.jpg
│   │   └── multi_02.jpg
│   ├── test/
│   │   ├── bin_22.jpg
│   │   ├── bin_23.jpg
│   │   ├── ...
│   │   ├── power_drill_22.jpg
│   │   ├── power_drill_23.jpg
│   │   └── ...
│   └── val/
│       ├── bin_28.jpg
│       ├── bin_29.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── bin_01.txt
│   │   ├── bin_02.txt
│   │   ├── ...
│   │   ├── power_drill_01.txt
│   │   ├── power_drill_02.txt
│   │   ├── ...
│   │   ├── multi_01.txt
│   │   └── multi_02.txt
│   ├── test/
│   │   ├── bin_22.txt
│   │   ├── bin_23.txt
│   │   ├── ...
│   │   ├── power_drill_22.txt
│   │   ├── power_drill_23.txt
│   │   └── ...
│   └── val/
│       ├── bin_28.txt
│       ├── bin_29.txt
│       └── ...
└── classes.txt
```

## Usage

### Complete Command Example
```bash
# Generate complete dataset for all objects (recommended for first-time use)
python generate_YOLO_dataset.py
```

### Custom Dataset Generation
```bash
# Generate dataset for specific object
python generate_YOLO_dataset.py --object power_drill

# Generate multi-object dataset (2-6 objects)
python generate_YOLO_dataset.py --object multi

# Generate multi-object dataset with custom object count
python generate_YOLO_dataset.py --object multi --min_objects 3 --max_objects 5

# Generate specific dataset split only
python generate_YOLO_dataset.py --object power_drill --split train

# Generate dataset with custom index range
python generate_YOLO_dataset.py --object cube --start_idx 10 --end_idx 20
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--object` | str | None | Object name or 'multi' for multi-object |
| `--start_idx` | int | 1 | Starting index for data collection |
| `--end_idx` | int | 31 | Ending index for data collection |
| `--split` | str | None | Dataset split (train/test/val) |
| `--min_objects` | int | 2 | Minimum objects for multi-object mode |
| `--max_objects` | int | 6 | Maximum objects for multi-object mode |

**Index Range Information**:
- Default range: 1-31 (30 images total)
- Training set: indices 1-21
- Test set: indices 22-27
- Validation set: indices 28-30


### ⚠️ Important Notes

**Data Overwrite Warning**: 
- If you regenerate a specific object dataset, it will **automatically delete** all existing data for that object in the dataset folder
- If you regenerate the complete dataset, it will **delete all existing data** in the dataset folder
- Always backup your dataset before regeneration if needed

**Examples of data deletion**:
- `python generate_YOLO_dataset.py --object power_drill` → Deletes all existing power_drill data
- `python generate_YOLO_dataset.py --object multi` → Deletes all existing multi-object data  
- `python generate_YOLO_dataset.py` → Deletes all existing data

**Regeneration scenarios**:
- If you're unsatisfied with a specific object's dataset quality, you can regenerate it
- If you want to collect multi-object datasets, you can generate them separately
- Always check the generated data quality before proceeding with training

**Auto-Retry Mechanism**: 
The tool automatically detects when objects fall off the table or intersect with each other (in multi-object mode). When such issues are detected, the system automatically re-collects the current frame data to ensure high-quality dataset generation.


### Performance Tips
- Use headless mode for faster generation
- Adjust physics simulation parameters for stability
- Monitor system resources during generation



