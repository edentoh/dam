import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Optional, Union

# Optional: Import transformers only if needed to keep dependencies light
try:
    from transformers import AutoModelForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HFWrapper(nn.Module):
    """Wraps a HuggingFace model to behave like a standard PyTorch model (accepting just x)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).logits


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    in_chans: int = 3
) -> nn.Module:
    """
    Universal model builder.
    - If backbone starts with 'hf:', loads via Transformers (AutoModelForImageClassification).
    - Otherwise, loads via Timm.
    """
    backbone = str(backbone).strip()

    # --- HuggingFace Path ---
    if backbone.lower().startswith("hf:"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for 'hf:' models. pip install transformers")
        
        repo = backbone.split(":", 1)[1].strip()
        hf_model = AutoModelForImageClassification.from_pretrained(repo, use_safetensors=True)

        # Replace Head Logic
        if hasattr(hf_model, "classifier") and isinstance(hf_model.classifier, nn.Module):
            in_features = getattr(hf_model.classifier, "in_features", None)
            if in_features is None and hasattr(hf_model.classifier, "weight"):
                in_features = hf_model.classifier.weight.shape[1]
            hf_model.classifier = nn.Linear(int(in_features), num_classes)
        
        elif hasattr(hf_model, "head") and isinstance(hf_model.head, nn.Module):
            in_features = getattr(hf_model.head, "in_features", None)
            if in_features is None and hasattr(hf_model.head, "weight"):
                in_features = hf_model.head.weight.shape[1]
            hf_model.head = nn.Linear(int(in_features), num_classes)
        else:
            raise RuntimeError("HF model head not found (expected .classifier or .head).")

        return HFWrapper(hf_model)

    # --- Timm Path ---
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans
    )
    return model


class ModelBuilder:
    """
    Config-driven factory for the Training loop.
    Handles 'pose_pretrain' logic specific to the DAM project.
    """
    @staticmethod
    def build(cfg: dict, device: torch.device):
        model_cfg = cfg.get("model", {})
        backbone = model_cfg.get("backbone", "convnextv2_tiny")
        num_classes = int(model_cfg.get("num_classes", 48))
        pretrained = bool(model_cfg.get("pretrained", True))

        print(f"[Model] Building: {backbone} (pretrained={pretrained})")
        
        model = build_model(backbone, num_classes, pretrained)
        model.to(device)

        # Custom Pose Pretraining Loading
        if model_cfg.get("use_pose_pretrain", False):
            path = model_cfg.get("pose_pretrain_backbone", "")
            ModelBuilder._load_pose_weights(model, path)

        return model

    @staticmethod
    def _load_pose_weights(model, path):
        p = Path(path)
        if not p.exists():
            print(f"[PoseInit] Warning: Path not found {p}, skipping.")
            return

        print(f"[PoseInit] Loading backbone from {p}")
        ckpt = torch.load(p, map_location="cpu")
        
        # Handle various checkpoint formats
        state = ckpt
        if isinstance(ckpt, dict):
            state = ckpt.get("backbone_state", ckpt.get("model", ckpt.get("state_dict", ckpt)))

        # Loose loading (strict=False) to allow head mismatch
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[PoseInit] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")