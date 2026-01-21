# DAM (Draw-A-Man) Assessment Project

A modular deep learning framework for automating the scoring of children's "Draw-a-Man" drawings. This project includes a multi-label classification model for the 48-item checklist, a binary "gating" model to validate inputs, and a full inference pipeline (CLI & Web API).

## 📌 Features

* **Score Model**: Multi-label classification (48 binary attributes) using `timm` backbones (e.g., ConvNeXt V2).
* **Quality Gating**:
    * **Heuristic Gate**: Filters blank, dark, or non-drawing images based on pixel stats.
    * **ML Gate**: Binary classifier ("Is this a valid DAM drawing?") to reject out-of-distribution inputs.
* **Advanced Inference**: Test-Time Augmentation (TTA) checks and optimized probability thresholds per criteria.
* **Web API**: FastAPI server for real-time scoring.

## 🛠️ Installation

1.  **Clone the repository:**

2.  **Install dependencies:**
    It is recommended to use a virtual environment (Python 3.9+).
    ```bash
    # Install libraries
    pip install -r requirements.txt

    # Install the 'dam' package in editable mode
    pip install -e .
    ```

## ⚙️ Configuration

The project uses a split configuration system:
* **`.env`**: System settings (API keys, device, web server limits, gate thresholds).
* **TOML files** (in `configs/`): Experiment hyperparameters (epochs, learning rates, data paths).

**Standard Configs:**
* `configs/config_score.toml`: For the main 48-item scoring model.
* `configs/config_gating.toml`: For the binary validity model.

**Environment Setup:**
* Create a `.env` file from the example to configure API keys and default thresholds.
 ```bash
 cp .env.example .env
 # Edit .env to set DAM_API_KEY, device preferences, etc.
 ```

---
## 🚀 Usage

It is recommended to run commands from the project root.

### 1. Data Preparation (for gate training)
Utilities to organize raw datasets before training.

* **Flatten a raw download folder:**
    Moves images to a single folder and quarantines non-image files.
    ```bash
    python -m scripts.flatten_images --root raw_downloads --out img_dataset/flattened
    ```

* **Filter "Hard Negatives" for Gating:**
    Finds non-DAM images that pass basic heuristics (to train the ML gate against).
    ```bash
    python -m scripts.filter_by_gate --input false_images --output candidates_passed
    ```

* **Generate Binary Labels:**
    Creates the CSV required to train the Gate Model.
    ```bash
    python -m scripts.make_binary_labels --pos image_cropped --neg candidates_passed --out labels_gate
    ```

### 2. Training

* **Train the Main Scoring Model (48 classes):**
    ```bash
    python -m scripts.train_score --config configs/config_score.toml
    ```

* **Train the Binary Gate Model (Is-DAM?):**
    ```bash
    python -m scripts.train_gate --config configs/config_gating.toml
    ```

### 3. Optimization

* **Compute Optimal Thresholds:**
    After training the score model, calculate the best probability threshold for each of the 48 criteria using the validation set.
    ```bash
    python -m scripts.compute_threshold_vector --config configs/config_score.toml
    ```
    *This saves `threshold_vector.json` next to your model file.*

### 4. Inference

* **Batch Prediction (CLI):**
    Score a folder of images and export results to Excel (`DAM_Predictions.xlsx`).
    ```bash
    python -m scripts.predict_cli --config configs/config_score.toml
    ```

* **Web API Server:**
    Start the FastAPI server (auto-reloads on code changes).
    ```bash
    python -m scripts.app --host 0.0.0.0 --port 8000 --reload
    ```
    *Access the web UI at `http://localhost:8000`*

## 📂 Project Structure

```text
.
├── configs/                 # TOML configuration files
├── scripts/                 # Entrypoint scripts (run with python -m)
│   ├── train_score.py       # Main model training
│   ├── train_gate.py        # Binary gate training
│   ├── predict_cli.py       # Excel export inference
│   └── app.py               # Web server
├── src/
│   └── dam/                 # Core package
│       ├── api/             # FastAPI routes & dependencies
│       ├── core/            # Config & constants
│       ├── data/            # Datasets & Transforms (CropToInk)
│       ├── gating/          # Heuristic & ML gating logic
│       ├── inference/       # Predictor & threshold logic
│       ├── modeling/        # Model builders (timm/HF)
│       └── training/        # Training engine & losses
└── requirements.txt         # Python dependencies
