# Yolov11m-TrashCan

# Marine Debris Detection (TrashCan-Instance) — YOLO training scripts

This repository contains the scripts and commands used to reproduce our YOLOv11 training experiments on the TrashCan-Instance dataset (underwater marine debris).

The repo is intended for:
- dataset preparation (COCO → YOLO format),
- generating dataset variants (crop-based augmentation, 3-class mapping),
- running training/ablation experiments with Ultralytics YOLO.

---

## Contents

- `yolo_scripts/`  
  Python scripts for dataset preparation and dataset variants.

- `commands.txt`  
  Copy-paste commands for training runs (baseline + ablations).  
  **Important:** the commands use placeholders (e.g., `<DATA_YAML>`). Replace them with paths on your machine.

- `.gitignore`  
  Prevents committing datasets, runs, and weights.

---

## Not included

To keep the repository lightweight and compliant, the following are NOT included:
- dataset images/labels,
- `runs/` training outputs,
- model weights (`*.pt`).

---

## Requirements

- Python 3.10+ recommended
- Ultralytics YOLO

## How to run

### 1) Clone this repository

```bash
git clone <REPO_URL>
cd <PROJECT_ROOT>
```
###** 2) Create and activate a virtual environment (Windows)**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U ultralytics pillow pyyaml tqdm

