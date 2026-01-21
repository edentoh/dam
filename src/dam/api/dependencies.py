import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import timm
from timm.data import resolve_data_config, create_transform
from torchvision import transforms
from fastapi import Request, HTTPException
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Internal Imports (No more config.py) ---
from dam.data.transforms import CropToInk
from dam.inference.thresholds import load_threshold_vector

# --- Helper Functions ---
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return v.strip().lower() in {"1", "true", "yes", "y", "on"} if v else default

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v and v.strip() else default

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v and v.strip() else default

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return str(v) if v else default

# --- 1. Security & System Constants ---
API_KEY = _env_str("DAM_API_KEY", "")
MAX_UPLOAD_BYTES = _env_int("DAM_MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
RATE_LIMIT_N = _env_int("DAM_RATE_LIMIT_N", 20)
RATE_LIMIT_WINDOW = _env_int("DAM_RATE_LIMIT_WINDOW", 60)
DEVICE_NAME = _env_str("DAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Main Model Config (New Envs) ---
MODEL_BACKBONE = _env_str("DAM_MODEL_BACKBONE", "convnextv2_tiny")
MODEL_NUM_CLASSES = _env_int("DAM_MODEL_NUM_CLASSES", 48)
MODEL_PATH = _env_str("DAM_MODEL_PATH", "runs/default_run/best.pth")
MODEL_IMG_SIZE = _env_int("DAM_MODEL_IMG_SIZE", 384)

# --- 3. Threshold Config ---
THRESHOLD_SCALAR = _env_float("DAM_THRESHOLD_SCALAR", 0.5)
THRESHOLD_VECTOR_PATH = _env_str("DAM_THRESHOLD_VECTOR_PATH", "threshold_vector.json")
REQUIRE_THRESHOLD_VECTOR = _env_bool("DAM_REQUIRE_THRESHOLD_VECTOR", False)

# --- 4. Preprocessing Config ---
USE_CROP_TO_INK = _env_bool("DAM_USE_CROP_TO_INK", False)
CROP_PAD = _env_int("DAM_CROP_PAD", 12)
CROP_MIN_SIZE = _env_int("DAM_CROP_MIN_SIZE", 50)

# --- 5. Gating Config ---
GATE_ENABLED = _env_bool("DAM_GATE_ENABLED", True)
GATE_RETURN_METRICS = _env_bool("DAM_GATE_RETURN_METRICS", False)
GATE_USE_TTA_FLIP = _env_bool("DAM_GATE_USE_TTA_FLIP", True)

GATE_CFG = {
    "enabled": GATE_ENABLED,
    "use_tta_flip": GATE_USE_TTA_FLIP,
    "min_ink_frac": _env_float("DAM_GATE_MIN_INK_FRAC", 0.002),
    "max_ink_frac": _env_float("DAM_GATE_MAX_INK_FRAC", 0.45),
    "min_margin": _env_float("DAM_GATE_MIN_MARGIN", 0.06),
    "max_flip_l1": _env_float("DAM_GATE_MAX_FLIP_L1", 0.15),
    # Add heuristic defaults directly here or read more envs if needed
    "min_side_px": _env_int("DAM_GATE_MIN_SIDE_PX", 256),
    "min_mean_gray": _env_float("DAM_GATE_MIN_MEAN_GRAY", 60.0),
}

# --- 6. Binary Gate (Is-DAM) Config ---
IS_DAM_ENABLED = _env_bool("DAM_IS_DAM_ENABLED", True)
IS_DAM_RETURN_METRICS = _env_bool("DAM_IS_DAM_RETURN_METRICS", False)
IS_DAM_THRESHOLD = _env_float("DAM_IS_DAM_THRESHOLD", 0.5)
IS_DAM_MODEL_PATH = _env_str("DAM_IS_DAM_MODEL_PATH", "runs/is_dam_v1/best.pt")


# --- Rate Limiting Logic ---
_rate_state: Dict[str, Tuple[float, int]] = {}

def check_rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    ws, cnt = _rate_state.get(ip, (now, 0))
    if now - ws >= RATE_LIMIT_WINDOW:
        ws, cnt = now, 0
    cnt += 1
    _rate_state[ip] = (ws, cnt)
    if cnt > RATE_LIMIT_N:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

def verify_api_key(request: Request):
    if not API_KEY:
         raise RuntimeError("Missing DAM_API_KEY env var.")
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# --- Inference Context ---
class InferenceContext:
    def __init__(self):
        print("Loading Inference Context from ENV...")
        self.device = torch.device(DEVICE_NAME)
        
        # 1. Main Model
        self.backbone = MODEL_BACKBONE
        self.num_classes = MODEL_NUM_CLASSES
        self.img_size = MODEL_IMG_SIZE
        
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path.resolve()}")
        
        print(f"Loading Main Model: {self.backbone} from {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")
        
        self.model = timm.create_model(self.backbone, pretrained=False, num_classes=self.num_classes)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Thresholds
        self.thr_scalar = THRESHOLD_SCALAR
        thr_vec_path = Path(THRESHOLD_VECTOR_PATH)
        
        # If path is relative, try resolving it next to the model file first
        if not thr_vec_path.is_absolute() and (model_path.parent / thr_vec_path).exists():
            thr_vec_path = model_path.parent / thr_vec_path
        
        self.thr_vec, self.thr_info = load_threshold_vector(
            thr_vec_path, 
            num_classes=self.num_classes, 
            fallback_thr=self.thr_scalar
        )
        self.thr_vec_path = thr_vec_path

        if REQUIRE_THRESHOLD_VECTOR and "scalar_fallback" in self.thr_info["threshold_mode"]:
             print("[WARNING] DAM_REQUIRE_THRESHOLD_VECTOR=True but vector file not found.")

        # 3. Binary Gate Model
        self.is_dam_model = None
        self.is_dam_tfm = None
        self.is_dam_backbone = None
        self.is_dam_img_size = None
        
        is_dam_path = Path(IS_DAM_MODEL_PATH)
        if IS_DAM_ENABLED:
            if not is_dam_path.exists():
                 print(f"[Warning] IS_DAM enabled but model not found at {is_dam_path}")
            else:
                print(f"Loading IS_DAM Gate from {is_dam_path}...")
                gate_ckpt = torch.load(is_dam_path, map_location="cpu")
                gate_cfg = gate_ckpt.get("cfg", {}) if isinstance(gate_ckpt, dict) else {}
                
                # Try to infer config from checkpoint, fallback to main backbone if missing
                self.is_dam_backbone = str(gate_cfg.get("backbone", self.backbone))
                self.is_dam_img_size = int(gate_cfg.get("img_size", 224))
                
                gate_state = gate_ckpt.get("model", gate_ckpt.get("model_state", gate_ckpt))
                
                self.is_dam_model = timm.create_model(self.is_dam_backbone, pretrained=False, num_classes=1)
                self.is_dam_model.load_state_dict(gate_state, strict=True)
                self.is_dam_model.to(self.device)
                self.is_dam_model.eval()
                
                gate_data_cfg = resolve_data_config({}, model=self.is_dam_model)
                gate_data_cfg["input_size"] = (3, self.is_dam_img_size, self.is_dam_img_size)
                self.is_dam_tfm = create_transform(**gate_data_cfg, is_training=False)

        # 4. Transforms
        ops = []
        if USE_CROP_TO_INK:
            ops.append(CropToInk(pad=CROP_PAD, min_size=CROP_MIN_SIZE))

        ops.extend([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.tfm = transforms.Compose(ops)
        
        # Expose config dictionaries for the /health endpoint
        self.gate_cfg = GATE_CFG

_context: Optional[InferenceContext] = None

def get_inference_context() -> InferenceContext:
    global _context
    if _context is None:
        _context = InferenceContext()
    return _context