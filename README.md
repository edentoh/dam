# DAM Classifier (Draw-a-Man Checklist)

A deep learning framework for multi-label classification of children's "Draw-a-Man" drawings. This project scores drawings against a **48-item checklist** using PyTorch and `timm` backbones.

## 📌 Overview

This repository implements a complete pipeline for training, validating, and deploying a computer vision model to automate the scoring of the DAM checklist.

**Key Features:**
* **Multi-Label Classification:** Predicts 48 binary attributes simultaneously.
* **Flexible Backbones:** Supports any model from the `timm` library (e.g., ConvNeXt V2, ResNet, EfficientNet).
* **Advanced Training:**
  * **Loss Functions:** Supports standard `BCEWithLogitsLoss` (with optional class weighting) and **Asymmetric Loss (ASL)** for handling class imbalance.
  * **Discriminative Learning Rates:** Allows different learning rates for the backbone vs. the classifier head.
  * **Hybrid Freeze:** Option to freeze the backbone for $N$ epochs before fine-tuning.
* **Preprocessing:** Includes a custom "Crop to Ink" transform that centers and crops the drawing content based on pixel intensity.
* **Cross-Validation:** Built-in support for K-Fold Cross-Validation.
* **Inference:** dedicated scripts to predict on new images and export results to Excel.

## 🛠️ Installation

1. **Clone the repository.**
2. **Install dependencies:**
   Python 3.10+ is recommended.

   ```bash
   pip install -r requirements.txt

```

*Key libraries: `torch`, `timm`, `pandas`, `openpyxl`, `pillow`, `numpy`.*

## 📂 Data Preparation

### 1. Image Directory Structure

Images should be placed in a root folder (e.g., `img_dataset`) with subfolders for splits if using fixed-split training.

```text
img_dataset/
├── train/
│   ├── drawing_001.jpg
│   └── ...
└── val/
    ├── drawing_105.jpg
    └── ...

```

**Important:** Image filenames **must** contain a 3-digit ID (e.g., `abc_001.jpg`, `123.png`) to link them to the labels.

### 2. Labels (Excel Format)

Labels are loaded from an Excel file (e.g., `labels/Score_j.xlsx`).

* **Columns:** Must contain "image" in the header (e.g., "Image 001"). The code extracts the ID from this header.
* **Rows:** The code specifically reads the **first 48 rows** as the checklist criteria.
* **Values:** `1` (present), `0` (absent), or empty (coerced to 0).

## ⚙️ Configuration (`config.toml`)

All project settings are managed in `config.toml`. Key sections include:

* **[system]:** Device selection (`cuda`/`cpu`), seeds, and run names.
* **[model]:**
* `backbone`: The `timm` model name (e.g., `convnextv2_tiny`).
* `num_classes`: Set to **48**.


* **[train]:**
* `loss`: Choose `"bce"` or `"asl"`.
* `use_weighted_loss`: Auto-calculates positive weights based on training data balance.
* `use_discriminative_lr`: Enable distinct learning rates for the head and backbone.
* `cv`: Enable Cross-Validation settings.


* **[train.data]**: Paths to images and labels for training.
* **[predict]**: Settings for the inference scripts.

## 🚀 Usage

### 1. Training

To train the model using the settings in `config.toml`:

```bash
# Ensure you are in the root directory
python -m scripts.train --config config.toml

```

**Cross-Validation:**
To run K-Fold CV, set `enabled = true` under the `[train.cv]` section in `config.toml`. The script will generate a specific folder for each fold and a summary JSON.

### 2. Computing Thresholds (Optional)

After training, you can calculate the optimal probability threshold for *each* of the 48 items to maximize accuracy on the validation set:

```bash
python -m scripts.compute_threshold_vector

```

This saves a `threshold_vector.json` alongside your model, which allows for per-item sensitivity adjustments during inference.

### 3. Inference (Predict to Excel)

To run the model on a folder of images and generate an Excel report:

1. Update `[predict]` in `config.toml` with your `model_path` and `input_image_dir`.
2. Run:

```bash
python -m scripts.predict_to_excel

```

This generates `DAM_Predictions.xlsx` containing:

* **Predictions_0_1:** Binary outputs (using thresholds).
* **Probabilities_0_1:** Raw confidence scores.
* **Metrics:** Accuracy/F1 scores (if labels are available for the input images).

## 🏗️ Project Structure

```text
.
├── config.toml           # Main configuration file
├── requirements.txt      # Dependencies
├── dam/                  # Core Python package
│   ├── config.py         # Config loader
│   ├── data.py           # Dataset & Dataloader logic
│   ├── engine.py         # Training loop & evaluation
│   ├── loss.py           # Custom loss functions (ASL, Weighted BCE)
│   ├── model.py          # Model builder (timm wrapper)
│   ├── transforms.py     # Custom transforms (CropToInk)
│   └── ...
└── scripts/              # Entrypoint scripts
    ├── train.py                   # Training entrypoint
    ├── predict_to_excel.py        # Batch inference
    └── compute_threshold_vector.py # Threshold optimization

```

## 📜 License & Credits

This project utilizes:

* [PyTorch](https://pytorch.org/)
* [timm](https://github.com/huggingface/pytorch-image-models)
* [OpenPyXL](https://openpyxl.readthedocs.io/)

```

```
