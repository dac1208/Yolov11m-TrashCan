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

Install:
```bash
pip install -U ultralytics pillow pyyaml tqdm 

HOW TO RUN

1) CLONE THIS REPOSITORY
   - git clone <REPO_URL>
   - cd <PROJECT_ROOT>


2) CREATE AND ACTIVATE A VIRTUAL ENVIRONMENT (WINDOWS)

   PowerShell:
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1
   - pip install -U ultralytics pillow pyyaml tqdm

   CMD:
   - python -m venv .venv
   - .\.venv\Scripts\activate.bat
   - pip install -U ultralytics pillow pyyaml tqdm


3) PREPARE DATASETS (OPTIONAL)
   - Run these only if you need to generate the YOLO folders/YAMLs.

   - python yolo_scripts\make_yolo_dataset.py
   - python yolo_scripts\make_cropmix_standard.py
   - python yolo_scripts\make_cropmix_singleobj.py
   - python yolo_scripts\make_3cls_dataset.py


4) RUN TRAINING COMMANDS
   - Open Commands.txt
   - Replace placeholders like <DATA_YAML> or <RUN_NAME>
   - Copy-paste the selected command into the terminal


OUTPUTS
- Ultralytics saves training artifacts into:
  - runs/<run_name>/
    - results.csv
    - results.png
    - confusion_matrix.png
    - confusion_matrix_normalized.png
    - labels.jpg
    - weights/best.pt
    - weights/last.pt

