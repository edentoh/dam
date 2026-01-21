import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# --- Internal Imports ---
# FIXED: Updated imports to match the names in src/dam/gating/
from dam.gating import gate_image_heuristics, check_prediction_stability
from .dependencies import (
    verify_api_key, 
    check_rate_limit, 
    get_inference_context, 
    InferenceContext,
    MAX_UPLOAD_BYTES,
    GATE_RETURN_METRICS,
    IS_DAM_THRESHOLD,
    IS_DAM_RETURN_METRICS,
    IS_DAM_ENABLED,
    API_KEY # Imported just to check if exists
)
from .schemas import PredictResponse, HealthResponse

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = PROJECT_ROOT / "web"
INDEX_HTML = WEB_DIR / "index.html"

_env_origins = os.environ.get("DAM_ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in _env_origins.split(",") if origin.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

# --- App Init ---
app = FastAPI(title="DAM Predictor (Demo, secured)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---

@app.on_event("startup")
async def startup_event():
    # Pre-load models so the first request isn't slow
    # Note: verify_api_key dependency handles the runtime check, 
    # but checking here ensures immediate failure on startup if config is wrong.
    if not API_KEY:
         raise RuntimeError("Missing DAM_API_KEY environment variable. Check your .env file.")
    get_inference_context()

@app.get("/")
def home():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return {"status": "ok", "message": "DAM Predictor API"}

@app.options("/predict")
def options_predict():
    return Response(status_code=204)

@app.get("/health", response_model=HealthResponse)
def health(ctx: InferenceContext = Depends(get_inference_context)):
    return {
        "status": "ok",
        "device": str(ctx.device),
        "backbone": ctx.backbone,
        "img_size": ctx.img_size,
        "num_classes": ctx.num_classes,
        "threshold_mode": ctx.thr_info["threshold_mode"],
        "threshold_vector_path": str(ctx.thr_vec_path),
        "threshold_scalar_fallback": ctx.thr_scalar,
        # Updated: Read from dependencies constant instead of ctx.cfg
        "require_threshold_vector": bool(os.environ.get("DAM_REQUIRE_THRESHOLD_VECTOR", "False") == "True"),
        "rate_limit": {"n": 20, "window_sec": 60},
        "max_upload_bytes": 5 * 1024 * 1024, # You can import MAX_UPLOAD_BYTES if needed
        "gating": {
            "enabled": bool(ctx.gate_cfg.get("enabled", True)),
            "use_tta_flip": bool(ctx.gate_cfg.get("use_tta_flip", True)),
            "return_metrics": False, # You can import GATE_RETURN_METRICS if needed
        },
        "is_dam": {
            "enabled": bool(ctx.is_dam_model is not None),
            # Updated: use ctx.is_dam_backbone directly
            "backbone": str(ctx.is_dam_backbone) if ctx.is_dam_backbone else None, 
            "img_size": ctx.is_dam_img_size if ctx.is_dam_img_size else 0,
            "threshold": 0.5, # Or import IS_DAM_THRESHOLD
        },
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    ctx: InferenceContext = Depends(get_inference_context),
    _auth: None = Depends(verify_api_key),
    _rate: None = Depends(check_rate_limit)
):
    ip = request.client.host if request.client else "unknown"

    # 1. Read Upload
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_BYTES} bytes.")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Heuristic Gate (Image Quality)
    # FIXED: Function name updated to match src/dam/gating/heuristics.py
    g1 = gate_image_heuristics(img, cfg=ctx.gate_cfg)
    if not g1.ok:
        print(f"[{ip}] REJECTED: Non-ML gate failed: {g1.message}")
        detail = {"error": g1.code, "message": g1.message}
        if GATE_RETURN_METRICS:
            detail["metrics"] = g1.metrics
        raise HTTPException(status_code=422, detail=detail)

    # 3. Binary ML Gate (Is DAM?)
    is_dam_prob = None
    if IS_DAM_ENABLED and ctx.is_dam_model is not None and ctx.is_dam_tfm is not None:
        xg = ctx.is_dam_tfm(img).unsqueeze(0).to(ctx.device)
        with torch.no_grad():
            logit_g = ctx.is_dam_model(xg)
            is_dam_prob = float(torch.sigmoid(logit_g).squeeze().item())

        if is_dam_prob < float(IS_DAM_THRESHOLD):
            print(f"[{ip}] REJECTED: Image rejected by is_dam model (prob={is_dam_prob:.3f})")
            detail = {
                "error": "not_dam",
                "message": "Image rejected: it does not appear to be a Draw-a-Man drawing.",
            }
            if IS_DAM_RETURN_METRICS or GATE_RETURN_METRICS:
                detail["is_dam_prob"] = is_dam_prob
            raise HTTPException(status_code=422, detail=detail)

    # 4. Main Inference
    x = ctx.tfm(img).unsqueeze(0).to(ctx.device)

    with torch.no_grad():
        logits = ctx.model(x)
        prob = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # 5. TTA Flip (Optional, for Stability Gate)
    prob_flip = None
    if bool(ctx.gate_cfg.get("enabled", True)) and bool(ctx.gate_cfg.get("use_tta_flip", True)):
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        x2 = ctx.tfm(img_flip).unsqueeze(0).to(ctx.device)
        with torch.no_grad():
            logits2 = ctx.model(x2)
            prob_flip = torch.sigmoid(logits2).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # 6. Stability Gate
    # FIXED: Function name updated to match src/dam/gating/uncertainty.py
    g2 = check_prediction_stability(prob, prob_flip=prob_flip, cfg=ctx.gate_cfg)
    if not g2.ok:
        print(f"[{ip}] REJECTED: Prediction stability gate failed: {g2.message}")
        detail = {"error": g2.code, "message": g2.message}
        if GATE_RETURN_METRICS:
            detail["metrics"] = g2.metrics
        raise HTTPException(status_code=422, detail=detail)

    if prob.shape[0] != ctx.thr_vec.shape[0]:
        raise HTTPException(
            status_code=500,
            detail=f"Threshold vector length mismatch: prob={prob.shape[0]} thr_vec={ctx.thr_vec.shape[0]}",
        )

    # 7. Formulate Response
    passed = (prob >= ctx.thr_vec).astype(int)
    total_score = int(passed.sum())

    items_list = [
        {
            "item": i + 1,
            "prob": float(prob[i]),
            "threshold": float(ctx.thr_vec[i]),
            "pass": int(passed[i]),
        }
        for i in range(int(prob.shape[0]))
    ]

    resp = {
        "filename": file.filename,
        "threshold_mode": ctx.thr_info["threshold_mode"],
        "threshold_vector_path": str(ctx.thr_vec_path),
        "threshold_scalar_fallback": ctx.thr_scalar,
        "total_score": total_score,
        "items": items_list,
    }

    if is_dam_prob is not None:
        print(f"[{ip}] SUCCESS: is_dam_prob={is_dam_prob:.3f}")
        resp["is_dam_prob"] = float(is_dam_prob)
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