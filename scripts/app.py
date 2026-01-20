import io
import os
import time
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from dam.gating import gate_image, gate_predictions

try:
    import tomllib as toml
except ImportError:
    import tomli as toml

import torch
import timm
from timm.data import resolve_data_config, create_transform
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv
load_dotenv()


# =========================
# Upload gating (non-ML filters)
# =========================
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v)

# If you want to tune gates without editing code, set these env vars (optional):
#   DAM_GATE_ENABLED=1|0
#   DAM_GATE_RETURN_METRICS=1|0  (include gate metrics in error detail + response)
#   DAM_GATE_USE_TTA_FLIP=1|0
#   DAM_GATE_MIN_INK_FRAC, DAM_GATE_MAX_INK_FRAC, DAM_GATE_MIN_MARGIN, DAM_GATE_MAX_FLIP_L1
GATE_ENABLED = _env_bool("DAM_GATE_ENABLED", True)
GATE_RETURN_METRICS = _env_bool("DAM_GATE_RETURN_METRICS", False)
GATE_USE_TTA_FLIP = _env_bool("DAM_GATE_USE_TTA_FLIP", True)

# Partial overrides (the rest uses gating.py defaults)
GATE_OVERRIDES = {
    "enabled": GATE_ENABLED,
    "use_tta_flip": GATE_USE_TTA_FLIP,
    "min_ink_frac": _env_float("DAM_GATE_MIN_INK_FRAC", 0.002),
    "max_ink_frac": _env_float("DAM_GATE_MAX_INK_FRAC", 0.45),
    "min_margin": _env_float("DAM_GATE_MIN_MARGIN", 0.06),
    "max_flip_l1": _env_float("DAM_GATE_MAX_FLIP_L1", 0.15),
}


# =========================
# Security configuration
# =========================
# Set these as environment variables (recommended).
API_KEY = os.environ.get("DAM_API_KEY", "").strip()
if not API_KEY:
    # You can run without it, but NOT recommended for a public demo.
    # We hard-fail so you don't accidentally expose an open endpoint.
    raise RuntimeError("Missing DAM_API_KEY environment variable. Set it before starting the server.")

# Limit upload size (bytes). Typical phone photo under 5MB.
MAX_UPLOAD_BYTES = int(os.environ.get("DAM_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5MB

# Simple per-IP rate limit: N requests per window seconds
RATE_LIMIT_N = int(os.environ.get("DAM_RATE_LIMIT_N", "20"))         # 20 requests
RATE_LIMIT_WINDOW = int(os.environ.get("DAM_RATE_LIMIT_WINDOW", "60"))  # per 60 seconds

# Allow only your demo site origin(s) to call API from browser JavaScript
# (This doesn't block curl/postman, but it blocks random websites)
ALLOWED_ORIGINS = [
    "https://aden-battlesome-begrudgingly.ngrok-free.app",
    # You can add your static-host domain here if you host the frontend elsewhere:
    # "https://your-frontend.example.com",
]


# In-memory store: ip -> (window_start_ts, count)
_rate_state: Dict[str, Tuple[float, int]] = {}


def check_rate_limit(ip: str):
    now = time.time()
    ws, cnt = _rate_state.get(ip, (now, 0))
    if now - ws >= RATE_LIMIT_WINDOW:
        ws, cnt = now, 0
    cnt += 1
    _rate_state[ip] = (ws, cnt)
    if cnt > RATE_LIMIT_N:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again shortly.")


def require_api_key(request: Request):
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# =========================
# Model + preprocessing
# =========================
class CropToInk:
    def __init__(self, threshold: int = 245, pad: int = 12, min_size: int = 50):
        self.threshold = int(threshold)
        self.pad = int(pad)
        self.min_size = int(min_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        g = img.convert("L")
        arr = np.array(g)
        mask = arr < self.threshold
        if int(mask.sum()) < self.min_size:
            return img
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        y0 = max(0, y0 - self.pad)
        x0 = max(0, x0 - self.pad)
        y1 = min(arr.shape[0] - 1, y1 + self.pad)
        x1 = min(arr.shape[1] - 1, x1 + self.pad)
        return img.crop((x0, y0, x1 + 1, y1 + 1))


def load_config(path: str = "config.toml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config.toml at: {p.resolve()}")
    with open(p, "rb") as f:
        return toml.load(f)


def _resolve_under_model_dir(model_path: Path, maybe_rel: Path) -> Path:
    """If maybe_rel is relative, resolve it next to model_path."""
    return maybe_rel if maybe_rel.is_absolute() else (model_path.parent / maybe_rel)


def load_threshold_vector(path: Path, num_classes: int, fallback_thr: float):
    """Load per-item thresholds from JSON.

    Supports:
      - JSON dict containing {"thresholds": [...]} (preferred)
      - raw JSON list [...]

    Returns: (thr_vec, info_dict)
    """
    info = {
        "threshold_mode": "scalar_fallback",
        "threshold_vector_path": str(path),
    }

    if not path.exists():
        thr_vec = np.full((num_classes,), float(fallback_thr), dtype=np.float32)
        info["threshold_mode"] = "scalar_fallback_missing_json"
        return thr_vec, info

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        arr = obj
        info["threshold_mode"] = "vector_from_json_list"
    elif isinstance(obj, dict) and "thresholds" in obj:
        arr = obj["thresholds"]
        info["threshold_mode"] = "vector_from_json.thresholds"
    else:
        raise ValueError(f"Unsupported threshold JSON format in {path}. Expect list or dict with 'thresholds'.")

    if not isinstance(arr, list) or len(arr) != num_classes:
        raise ValueError(
            f"Threshold vector must be a list of length {num_classes}. Got {type(arr)} len={len(arr) if isinstance(arr, list) else 'NA'}."
        )

    thr_vec = np.array(arr, dtype=np.float32)
    thr_vec = np.clip(thr_vec, 0.0, 1.0)
    return thr_vec, info


cfg = load_config("config.toml")

device_pref = cfg.get("system", {}).get("device", "cuda")
device = torch.device(device_pref if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")

backbone = cfg["model"]["backbone"]
num_classes = int(cfg["model"].get("num_classes", 48))

predict_cfg = cfg.get("predict", {})

# Gating config: config.toml can optionally define [predict.gating].
# Environment variables in GATE_OVERRIDES will override selected keys.
gate_cfg = {}
try:
    if isinstance(predict_cfg.get('gating', {}), dict):
        gate_cfg.update(predict_cfg.get('gating', {}))
except Exception:
    pass
gate_cfg.update(GATE_OVERRIDES)

data_cfg = predict_cfg.get("data", cfg.get("data", {}))  # back-compat if needed

# =========================
# Optional: is_dam (binary) model
# =========================
# Supports either:
#   [predict.is_dam] model_path, threshold, enabled
# or flat keys under [predict]: is_dam_model_path, is_dam_threshold, is_dam_enabled
is_dam_section = {}
try:
    if isinstance(predict_cfg.get("is_dam", {}), dict):
        is_dam_section.update(predict_cfg.get("is_dam", {}))
except Exception:
    pass

if "is_dam_model_path" in predict_cfg:
    is_dam_section.setdefault("model_path", predict_cfg.get("is_dam_model_path"))
if "is_dam_threshold" in predict_cfg:
    is_dam_section.setdefault("threshold", predict_cfg.get("is_dam_threshold"))
if "is_dam_enabled" in predict_cfg:
    is_dam_section.setdefault("enabled", predict_cfg.get("is_dam_enabled"))

IS_DAM_ENABLED = _env_bool("DAM_IS_DAM_ENABLED", bool(is_dam_section.get("enabled", True)))
IS_DAM_RETURN_METRICS = _env_bool("DAM_IS_DAM_RETURN_METRICS", False)
IS_DAM_THRESHOLD = _env_float("DAM_IS_DAM_THRESHOLD", float(is_dam_section.get("threshold", 0.5)))
IS_DAM_MODEL_PATH_STR = _env_str(
    "DAM_IS_DAM_MODEL_PATH",
    str(is_dam_section.get("model_path", "runs/is_dam_v1/best.pt")),
).strip()

is_dam_model_path = Path(IS_DAM_MODEL_PATH_STR) if IS_DAM_MODEL_PATH_STR else None

model_path = Path(predict_cfg["model_path"])
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

# Load checkpoint
ckpt = torch.load(model_path, map_location="cpu")
img_size = int(ckpt.get("img_size", data_cfg.get("img_size", 384)))

# Build model
model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.to(device)
model.eval()

# Load optional is_dam model (binary classifier)
is_dam_model = None
is_dam_tfm = None
is_dam_backbone = None
is_dam_img_size = None

# LOGGING FOR STARTUP
if IS_DAM_ENABLED:
    print(f"[System] DAM ML Gate ENABLED. Loading from {is_dam_model_path}...")
    if is_dam_model_path is None:
        raise RuntimeError("DAM_IS_DAM_ENABLED is true but no is_dam_model_path is configured.")
    if not is_dam_model_path.exists():
        raise FileNotFoundError(f"is_dam model not found: {is_dam_model_path.resolve()}")

    is_dam_ckpt = torch.load(is_dam_model_path, map_location="cpu")
    is_dam_ckpt_cfg = is_dam_ckpt.get("cfg", {}) if isinstance(is_dam_ckpt, dict) else {}

    is_dam_backbone = str(is_dam_ckpt_cfg.get("backbone", is_dam_section.get("backbone", backbone)))
    is_dam_img_size = int(is_dam_ckpt_cfg.get("img_size", is_dam_section.get("img_size", 224)))

    if isinstance(is_dam_ckpt, dict) and isinstance(is_dam_ckpt.get("model"), dict):
        is_dam_state = is_dam_ckpt["model"]
    elif isinstance(is_dam_ckpt, dict) and isinstance(is_dam_ckpt.get("model_state"), dict):
        is_dam_state = is_dam_ckpt["model_state"]
    else:
        # Fallback: assume the checkpoint itself is a state_dict
        is_dam_state = is_dam_ckpt

    is_dam_model = timm.create_model(is_dam_backbone, pretrained=False, num_classes=1)
    is_dam_model.load_state_dict(is_dam_state, strict=True)
    is_dam_model.to(device)
    is_dam_model.eval()

    # Use timm's inference transform that matches the trained backbone.
    is_dam_data_cfg = resolve_data_config({}, model=is_dam_model)
    is_dam_data_cfg["input_size"] = (3, is_dam_img_size, is_dam_img_size)
    is_dam_tfm = create_transform(**is_dam_data_cfg, is_training=False)
else:
    print("[System] DAM ML Gate DISABLED (IS_DAM_ENABLED=False).")

# Load threshold vector (preferred) with scalar fallback
thr_scalar = float(predict_cfg.get("threshold_scalar_fallback", predict_cfg.get("threshold", 0.5)))
thr_vec_path_cfg = Path(predict_cfg.get("threshold_vector_path", "threshold_vector.json"))
thr_vec_path = _resolve_under_model_dir(model_path, thr_vec_path_cfg)
thr_vec, thr_info = load_threshold_vector(thr_vec_path, num_classes=num_classes, fallback_thr=thr_scalar)

require_vec = bool(predict_cfg.get("require_threshold_vector", False))
if require_vec and thr_info["threshold_mode"].startswith("scalar_fallback"):
    raise RuntimeError(
        f"require_threshold_vector=true but vector thresholds not available. Tried: {thr_vec_path.resolve()}"
    )

# Transforms
ops = []
if bool(data_cfg.get("use_crop_to_ink", False)):
    ops.append(
        CropToInk(
            threshold=int(data_cfg.get("crop_threshold", 245)),
            pad=int(data_cfg.get("crop_pad", 12)),
            min_size=int(data_cfg.get("crop_min_size", 50)),
        )
    )

ops.extend(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

tfm = transforms.Compose(ops)


# =========================
# FastAPI app
# =========================
app = FastAPI(title="DAM Predictor (Demo, secured)")

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../DAM
WEB_DIR = PROJECT_ROOT / "web"
INDEX_HTML = WEB_DIR / "index.html"

@app.get("/")
def home():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return {"status": "ok", "message": "DAM Predictor API"}

# CORS: allow only your demo origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/predict")
def options_predict():
    # CORS middleware will add the correct headers.
    return Response(status_code=204)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "backbone": backbone,
        "img_size": img_size,
        "num_classes": num_classes,
        "threshold_mode": thr_info["threshold_mode"],
        "threshold_vector_path": str(thr_vec_path),
        "threshold_scalar_fallback": thr_scalar,
        "require_threshold_vector": require_vec,
        "rate_limit": {"n": RATE_LIMIT_N, "window_sec": RATE_LIMIT_WINDOW},
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "gating": {
            "enabled": bool(gate_cfg.get("enabled", True)),
            "use_tta_flip": bool(gate_cfg.get("use_tta_flip", True)),
            "return_metrics": GATE_RETURN_METRICS,
        },
        "is_dam": {
            "enabled": bool(IS_DAM_ENABLED and is_dam_model is not None),
            "model_path": str(is_dam_model_path) if is_dam_model_path else None,
            "threshold": float(IS_DAM_THRESHOLD),
            "backbone": is_dam_backbone,
            "img_size": is_dam_img_size,
            "return_metrics": IS_DAM_RETURN_METRICS,
        },
    }


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Security checks
    require_api_key(request)

    # Rate limit (best-effort)
    ip = request.client.host if request.client else "unknown"
    check_rate_limit(ip)

    # Read upload
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_BYTES} bytes.")

    # Decode image safely
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Gate 1: cheap image-level filters (paper/blank/tiny)
    g1 = gate_image(img, cfg=gate_cfg)
    if not g1.ok:
        print(f"[{ip}] REJECTED: Non-ML gate failed: {g1.message}")
        detail = {"error": g1.code, "message": g1.message}
        if GATE_RETURN_METRICS:
            detail["metrics"] = g1.metrics
        raise HTTPException(status_code=422, detail=detail)

    # Gate 1.5: ML validity filter (is_dam)
    is_dam_prob = None
    if IS_DAM_ENABLED and is_dam_model is not None and is_dam_tfm is not None:
        xg = is_dam_tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit_g = is_dam_model(xg)
            is_dam_prob = float(torch.sigmoid(logit_g).squeeze().item())

        if is_dam_prob < float(IS_DAM_THRESHOLD):
            print(f"[{ip}] REJECTED: Image rejected by is_dam model (prob={is_dam_prob:.3f} < threshold={IS_DAM_THRESHOLD})")
            detail = {
                "error": "not_dam",
                "message": "Image rejected: it does not appear to be a Draw-a-Man drawing.",
            }
            if IS_DAM_RETURN_METRICS or GATE_RETURN_METRICS:
                detail["is_dam_prob"] = is_dam_prob
            raise HTTPException(status_code=422, detail=detail)

    # Inference
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)  # (48,)

    # Optional flip TTA for stability gating only (does not affect scoring)
    prob_flip = None
    if bool(gate_cfg.get("enabled", True)) and bool(gate_cfg.get("use_tta_flip", True)):
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        x2 = tfm(img_flip).unsqueeze(0).to(device)
        with torch.no_grad():
            logits2 = model(x2)
            prob_flip = torch.sigmoid(logits2).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # Gate 2: prediction-level filters (uncertainty + instability)
    g2 = gate_predictions(prob, prob_flip=prob_flip, cfg=gate_cfg)
    if not g2.ok:
        print(f"[{ip}] REJECTED: Prediction stability gate failed: {g2.message}")
        detail = {"error": g2.code, "message": g2.message}
        if GATE_RETURN_METRICS:
            detail["metrics"] = g2.metrics
        raise HTTPException(status_code=422, detail=detail)

    if prob.shape[0] != thr_vec.shape[0]:
        raise HTTPException(
            status_code=500,
            detail=f"Threshold vector length mismatch: prob={prob.shape[0]} thr_vec={thr_vec.shape[0]}",
        )

    passed = (prob >= thr_vec).astype(int)
    total_score = int(passed.sum())

    items = [
        {
            "item": i + 1,
            "prob": float(prob[i]),
            "threshold": float(thr_vec[i]),
            "pass": int(passed[i]),
        }
        for i in range(int(prob.shape[0]))
    ]

    resp = {
        "filename": file.filename,
        "threshold_mode": thr_info["threshold_mode"],
        "threshold_vector_path": str(thr_vec_path),
        "threshold_scalar_fallback": thr_scalar,
        "total_score": total_score,
        "items": items,
    }

    # === ADDED: Put prob at ROOT LEVEL on success ===
    if is_dam_prob is not None:
        print(f"[{ip}] SUCCESS: returning result with is_dam_prob={is_dam_prob:.3f}")
        # Add to root for easier access
        resp["is_dam_prob"] = float(is_dam_prob)
        # Keep nested object if needed for consistency, or remove if clutter. Keeping for now.
        resp["is_dam"] = {
            "prob": float(is_dam_prob),
            "threshold": float(IS_DAM_THRESHOLD),
            "pass": int(is_dam_prob >= float(IS_DAM_THRESHOLD)),
        }

    if GATE_RETURN_METRICS:
        resp["gating"] = {
            "image": g1.metrics,
            "prediction": g2.metrics,
        }

    return resp