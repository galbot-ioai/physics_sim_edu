# YOLO-Seg

## Installation

```bash
pip install -r requirements.txt
```

## Usage
### Training
To train the model, run the following command:
```bash
python -m src.train --dataset=sim
```
### Inference
To perform inference on images, run the following command:
```bash
python -m src.predict
```
### Example
To run inference on a sample image, you can use the following command:
```bash
python seg.py
```

## Dataset
If you want to use your own dataset, please follow the instructions below to prepare it.
### Preparation
1. Install the `labelme` tool:
   ```bash
   pip install pyqt5
   pip install labelme==5.0.1
   ```
2. Use `labelme` to annotate your images and save the annotations in JSON format.
3. Convert the JSON annotations to YOLO format using the provided script:
   ```bash
   python -m src.convert_labelme_to_yolo
   ```
### Directory Structure
Your dataset should be organized as follows, with images and labels in separate directories for training, validation, and testing, and labels in YOLO format (text files with the same name as the images): (train_files:test_files:val_files = 7:2:1)
```
dataset/
    ├── images/
    │   ├── train/
    │   │   ├── 1_color_image.jpg
    │   │   ├── 2_color_image.jpg
    │   │   └── ...
    │   ├── test/
    │   │   ├── 3_color_image.jpg
    │   │   └── ...
    │   └── val/
    │       ├── 4_color_image.jpg
    │       └── ...
    ├── labels/
    │   ├── train/
    │   │   ├── 1_color_image.txt
    │   │   ├── 2_color_image.txt
    │   │   └── ...
    │   ├── test/
    │   │   ├── 3_color_image.txt
    │   │   └── ...
    │   └── val/
    │       ├── 4_color_image.txt
    │       └── ...