# Military Vehicles Computer Vision

A computer vision pipeline for **military vehicle detection** (YOLO) and **classification** (MobileNetV2) built on a COCO-annotated dataset.

## Overview

The project covers two parallel tasks:

- **Detection** — fine-tune YOLOv11-nano to localize vehicles in full images
- **Classification** — fine-tune MobileNetV2 to identify the class of a pre-cropped vehicle patch

Both pipelines share the same source dataset in COCO JSON format and use Weights & Biases for experiment tracking.

## Setup

Requires **Python 3.12**.

```bash
# Install uv (if not already installed)
pip install uv

# Create venv and install dependencies
uv sync
```

Key dependencies: `tensorflow>=2.16`, `keras>=3.14`, `ultralytics>=8.3`, `torch==2.2.0+cu121`, `wandb`.

## Data Preparation

### 1. COCO -> YOLO labels

```bash
python coco_to_yolo.py \
  --file data/_annotations.coco.json \
  --output data/labels/train \
  --classes data/
```

Generates per-image `.txt` files with normalized YOLO coordinates and `classes.txt`.

### 2. COCO -> Square crops

```bash
python coco_to_crops.py \
  --json-path data/_annotations.coco.json \
  --images-dir data/images/train \
  --output-dir data/cropped/train \
  --margin 0.2
```

Each object is cropped as a square with 20% context margin and saved under `output-dir/<class_name>/`.

### 3. Verify the data

```bash
python verify_data.py \
  --yolo-dir data/ \
  --crops-dir data/cropped \
  --n 5
```

Saves `verify_yolo.png` and `verify_crops.png` with annotated samples.

## Training

### YOLO (Detection)

```bash
python training/train_yolo.py \
  --model yolo11n.pt \
  --data data/dataset.yml \
  --epochs 100 \
  --img-size 640 \
  --batch-size 16 \
  --device auto \
  --patience 10 \
  --wandb \
  --wandb-project yolo11-finetune \
  --tag yolo11n_run1
```

Saved weights: `runs/<tag>/weights/best.pt`

### MobileNetV2 (Classification)

```bash
python training/train_mobile_net_v2.py \
  --dataset data/cropped \
  --epochs 100 \
  --img-size 224 \
  --batch-size 64 \
  --lr 1e-5 \
  --num-classes 6 \
  --patience 10 \
  --wandb-project mobilenetv2-finetune \
  --tag mobilenetv2_run1
```

Saved weights: `runs/classification/MobileNetV2_e100_<timestamp>.keras`

Training monitors `val_f1` (macro) with early stopping. After training, the model is evaluated on the test split and logs a confusion matrix and classification report to W&B.