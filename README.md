```md
# DAM_Classifier — Draw-A-Man (DAM) Checklist Multi-Label Classifier

This repository trains an image model to automatically score children’s Draw-A-Man (DAM) drawings against a **48-item checklist** (multi-label classification: each checklist item is a binary label).

The 48 target features correspond to the “Appendix 2 – Checklist for Draw-a-Man test” included in this repo (see `Appendix 2 DAM Checklist .docx`).

---

## What this repo does

- Trains a **multi-label** image classifier (default: 48 labels) using a `timm` backbone.
- Supports **standard train/val split** (folders) and optional **k-fold cross-validation**.
- Logs **Micro F1**, **Macro F1**, and **element-wise accuracy**, and saves a “best” checkpoint.
- Handles **class imbalance** via optional `pos_weight` calculation and clamping.
- Provides configurable augmentations and image preprocessing (including optional crop-to-ink).

---

## Checklist labels (48)

Each drawing is scored for the presence/quality of the 48 DAM features (gross detail, attachments, head detail, joints, fine head detail, clothing, hand detail, proportion, motor coordination, etc.).  
See: `Appendix 2 DAM Checklist .docx`.

---

## Repository layout (expected)

Typical layout:
```

.
├── config.toml
├── train.py
├── data.py
├── model.py
├── loss.py
├── engine.py
├── utils.py
├── labels/
│   └── Score_j.xlsx
└── img_dataset/
├── train/
│   ├── ... images ...
└── val/
├── ... images ...

````

---

## Data format

### 1) Images
- Place images under:
  - `img_dataset/train/`
  - `img_dataset/val/`
- Supported extensions: `.jpg .jpeg .png .bmp .webp`
- **Important:** Each image must contain a **3-digit ID** somewhere in its filename (e.g., `drawing_023.png`). The loader extracts the first `\d{3}` it finds and uses it as the key.

### 2) Labels (Excel)
- Labels are read from an Excel file (default: `labels/Score_j.xlsx`).
- The loader searches for columns whose header contains `"image"` (case-insensitive).
- For each such column:
  - It extracts the same **3-digit ID** from the column name.
  - It reads the **first 48 rows** (one per checklist criterion) as the label vector.
  - NaNs are treated as `0.0`.

Result: each drawing ID maps to a 48-dimensional multi-hot label vector.

---

## Installation

Create an environment and install dependencies.

Example (pip):
```bash
pip install torch torchvision timm pandas openpyxl pillow numpy
````

Notes:

* Python needs `tomllib` (Py3.11+) or `tomli` (older Python). If you are on Python < 3.11:

```bash
pip install tomli
```

---

## Quick start (single run)

1. Edit `config.toml` if needed (paths, backbone, batch size, etc.).
2. Run training:

```bash
python train.py --config config.toml
```

Training will:

* Build the model from `cfg.model` (default: `convnextv2_tiny`).
* Load data from `cfg.data`.
* Train for `cfg.train.epochs`.
* Save logs/checkpoints under `cfg.system.runs_dir / cfg.system.run_name` (auto-uniqued).

---

## Cross-validation mode (optional)

Enable CV in `config.toml`:

```toml
[cv]
enabled = true
num_runs = 5
seed = 999
```

Then run the same command:

```bash
python train.py --config config.toml
```

Outputs will be written under a directory like:

```
runs/<run_name>_CV/
  fold_1/
  fold_2/
  ...
```

---

## Configuration reference (high-impact settings)

Open `config.toml` and adjust:

### System

* `system.seed`: global seed for reproducibility
* `system.device`: `"cuda"` or `"cpu"`
* `system.run_name`: run folder name
* `system.runs_dir`: root runs directory

### Data

* `data.csv_path`: path to Excel labels
* `data.img_root_dir`: image root containing `train/` and `val/`
* `data.img_size`: input size (train uses RandomResizedCrop; val uses Resize)
* `data.num_workers`: DataLoader workers

### Model

* `model.backbone`: any `timm` backbone string
* `model.pretrained`: use ImageNet pretrained weights
* `model.num_classes`: number of checklist labels (default 48)
* `model.use_pose_pretrain`: optionally load backbone weights from a checkpoint path

### Train

* `train.epochs`, `train.batch_size`, `train.learning_rate`, `train.weight_decay`
* `train.loss`: `"bce"` or `"asl"`
* `train.threshold`: sigmoid threshold for metrics
* `train.metric_for_best`: `"val_f1_micro"` (default) or `"val_f1_macro"`
* `train.use_weighted_loss`: auto-compute `pos_weight` from training set for BCE
* `train.pos_weight_clamp`: cap for rare-label weights

---

## Outputs

For each run directory (e.g., `runs/convnextv2_tiny_ms1/`), you should see:

* `best.pth`
  Best checkpoint (by selected metric). Includes model state, best metric value, epoch metrics, LR, threshold, and full config.

* `history.json`
  Per-epoch log including train loss, val loss, Micro F1, Macro F1, and accuracy.

* `best_model_metadata.json`
  Human-readable summary of best epoch and where it was saved.

---

## Metrics

Validation computes:

* **Micro F1** (global TP/FP/FN aggregated across all labels)
* **Macro F1** (mean per-label F1)
* **Element-wise accuracy** (mean of correct label predictions across all labels and samples)

All metrics use `sigmoid(logits) >= threshold`.

---

## Notes on modeling choices (practical tips)

* **Class imbalance is expected** in checklist-style labels:

  * Start with `train.use_weighted_loss = true` (BCE) and tune `pos_weight_clamp`.
  * Consider switching to `train.loss = "asl"` if rare labels are consistently missed.

* **Threshold tuning matters**:

  * `train.threshold = 0.5` is a baseline; Macro F1 may improve with per-label thresholds or a tuned global threshold.

* **Augmentations**:

  * Training uses RandomResizedCrop, ColorJitter, and RandomAffine by default.
  * Validation uses deterministic Resize.
  * All images are converted to grayscale (3 channels) and normalized with ImageNet stats.

---

## Troubleshooting

* **No data found / 0 samples**

  * Confirm your images have a 3-digit ID in the filename.
  * Confirm the Excel column headers contain “image” and also include the same 3-digit ID.
  * Confirm `img_dataset/train` and `img_dataset/val` exist under `data.img_root_dir`.

* **CUDA not used**

  * If CUDA is unavailable, the code will fall back to CPU automatically.

---

## Acknowledgements / Reference

Checklist adapted from University of Washington (Goodenough Draw-A-Person / Draw-A-Man scoring reference). See `Appendix 2 DAM Checklist .docx`.

```

Sources used to derive the README’s behavior/claims: model construction and optional pose-weight loading :contentReference[oaicite:0]{index=0}, utilities (ID extraction, run directory handling, atomic JSON logging) :contentReference[oaicite:1]{index=1}, default configuration keys/values :contentReference[oaicite:2]{index=2}, training loop + metrics + artifacts saved :contentReference[oaicite:3]{index=3}, dataset/label loading and transforms :contentReference[oaicite:4]{index=4}, CLI entrypoint and CV logic :contentReference[oaicite:5]{index=5}, loss options (BCE / ASL and pos_weight) :contentReference[oaicite:6]{index=6}, and the 48-item DAM checklist definition :contentReference[oaicite:7]{index=7}.
```
