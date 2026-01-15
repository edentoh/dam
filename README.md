````markdown
# DAM_Classifier (Draw-a-Man Checklist Multi-Label Classifier)

Train a deep learning model to score children’s “Draw-a-Man” drawings against a **48-item DAM checklist** (multi-label classification: each drawing can have multiple checklist items present).

## What this repo does
- **Multi-label image classification** with a configurable `timm` backbone (default: ConvNeXtV2 Tiny).
- Supports **BCEWithLogitsLoss** (optionally class-weighted) or **Asymmetric Loss (ASL)**.
- Training supports either:
  - **Fixed split** using `img_dataset/train` and `img_dataset/val`, or
  - **K-fold cross-validation** (CV) by shuffling and splitting all images from both folders.
- Logs training history and saves a “best model” checkpoint with metadata.

---

## Repository structure
- `train.py` — main entrypoint for training (fixed split or CV).
- `data.py` — Excel label loading, image discovery, transforms, dataloaders.
- `model.py` — backbone creation via `timm`, optional backbone weight init (“pose pretrain”).
- `loss.py` — BCE / weighted BCE / ASL.
- `engine.py` — training loop, metrics, checkpointing, JSON history.
- `config.toml` — configuration (data/model/train/CV).

---

## Setup

### 1) Install dependencies
Recommended: Python 3.10+.

```bash
pip install torch torchvision timm pillow pandas openpyxl numpy
````

If you are on Python < 3.11:

```bash
pip install tomli
```

---

## Data and labels

### Image folder layout

Place images under:

```
img_dataset/
  train/
    ... your images ...
  val/
    ... your images ...
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

**Important:** Your image filename must include a **3-digit ID** (e.g., `001`, `128`, `905`). The training code extracts the first `\d{3}` occurrence and uses it to match labels.

### Label file format (Excel)

By default, labels are read from:

```
labels/Score_j.xlsx
```

Expected format:

* The code looks for **columns whose header includes “image”** (case-insensitive).
* For each such column, it extracts the **3-digit ID** from the column name.
* It then reads the **first 48 rows** from that column as the 48 checklist targets:

  * Values should be numeric (0/1 recommended).
  * Non-numeric will be coerced to NaN and then converted to 0.

In short: one Excel column per image, with 48 rows representing the 48 DAM checklist items.

---

## Configure your run (`config.toml`)

Key settings you will commonly edit:

### System

* `system.seed`: reproducibility seed.
* `system.device`: `"cuda"` or `"cpu"` (auto-falls back to CPU if CUDA not available).
* `system.run_name`: name for the run folder under `runs/`.

### Data

* `data.csv_path`: path to label Excel (e.g., `labels/Score_j.xlsx`).
* `data.img_root_dir`: dataset root (e.g., `img_dataset`).
* `data.img_size`: resize/crop target size.
* `data.num_workers`: dataloader workers.
* Optional pre-processing: `data.use_crop_to_ink` (see note below).

### Model

* `model.backbone`: any `timm` image model name (e.g., `convnextv2_tiny`).
* `model.num_classes`: should be **48** for the DAM checklist.
* Optional: `model.use_pose_pretrain` + `model.pose_pretrain_backbone` to load backbone weights (non-strict load).

### Training

* `train.epochs`, `train.batch_size`, `train.learning_rate`, `train.weight_decay`.
* `train.threshold`: probability threshold used for metrics.
* `train.loss`: `"bce"` or `"asl"`.
* `train.use_weighted_loss`: if `true`, auto-computes `pos_weight` from the training set (clamped by `train.pos_weight_clamp`).
* `train.metric_for_best`: choose best model by `val_f1_micro` (default) or `val_f1_macro`.

### Cross-validation (optional)

Set:

```toml
[cv]
enabled = true
num_runs = 5
seed = 999
```

This will create a `..._CV/` run folder with `fold_1/`, `fold_2/`, etc.

---

## Train

### Fixed train/val folders

```bash
python train.py --config config.toml
```

### Cross-validation

Enable `[cv].enabled = true` in `config.toml`, then:

```bash
python train.py --config config.toml
```

---

## Outputs

Runs are saved under:

```
runs/<run_name>/
```

Key files:

* `best.pth` — best checkpoint (based on `train.metric_for_best`).
* `history.json` — epoch-by-epoch training/validation metrics.
* `best_model_metadata.json` — human-readable summary of the best checkpoint.

In CV mode:

```
runs/<run_name>_CV/
  cv_config.json
  fold_1/
  fold_2/
  ...
```

---

## Metrics

Validation computes:

* **Micro F1** (global TP/FP/FN across all labels)
* **Macro F1** (mean over labels)
* **Element-wise accuracy** (mean over all label decisions)

All metrics use:

* `sigmoid(logits)` to obtain probabilities
* `train.threshold` to convert probabilities into 0/1 predictions

---

## Notes / common pitfalls

### 1) Crop-to-ink configuration key

The crop-to-ink transform is controlled by:

```toml
[data]
use_crop_to_ink = true
```

(If your `config.toml` currently has `train.use_crop_to_ink`, that key will not affect preprocessing unless you move/duplicate it under `[data]`.)

### 2) ID matching must be consistent

If the image filename contains `123` but the Excel column header does not include the same `123` (and the word “image”), the sample will be dropped.

### 3) Class imbalance

For sparse checklist items, enable:

```toml
[train]
use_weighted_loss = true
pos_weight_clamp = 9.0
```

This computes per-label positive weights from the training split and clamps extremes.

---

## Checklist definition

This project assumes **48 checklist targets** aligned to the DAM checklist you are using. Ensure the row order in the Excel file matches your checklist order exactly (row 1 ↔ item 1, …, row 48 ↔ item 48).

---

## License / attribution

If you redistribute the checklist or derived scoring rules, ensure you comply with the original checklist’s terms and provide appropriate attribution in your academic/clinical context.

```

Implementation references (from your repo files): model construction and optional backbone init :contentReference[oaicite:0]{index=0}, utilities (seeding, ID extraction, run folder handling) :contentReference[oaicite:1]{index=1}, default configuration example :contentReference[oaicite:2]{index=2}, training loop/metrics/checkpoint outputs :contentReference[oaicite:3]{index=3}, dataset + Excel label parsing + transforms :contentReference[oaicite:4]{index=4}, training entrypoint + CV behavior :contentReference[oaicite:5]{index=5}, loss functions and auto pos-weight computation :contentReference[oaicite:6]{index=6}, and the 48-item DAM checklist definition :contentReference[oaicite:7]{index=7}.
```
