import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Optional, Union

from .heads import BackboneWrapper, LinearHead, MLDecoderHead, MultiTaskModel, TotalScoreHead

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
        head_type = str(model_cfg.get("head_type", "linear")).lower()
        drop_path_rate = float(model_cfg.get("drop_path_rate", 0.0))

        aux_cfg = cfg.get("aux_head", {})
        aux_enabled = bool(aux_cfg.get("enabled", False))

        print(f"[Model] Building: {backbone} (pretrained={pretrained}, drop_path_rate={drop_path_rate})")
        
        # --- HuggingFace Path ---
        if str(backbone).strip().lower().startswith("hf:"):
            if head_type != "linear" or aux_enabled:
                raise NotImplementedError(
                    "HF backbones currently support only head_type='linear' and aux_head disabled."
                )
            model = build_model(backbone, num_classes, pretrained)
            model.to(device)

            # Custom Pose Pretraining Loading
            if model_cfg.get("use_pose_pretrain", False):
                path = model_cfg.get("pose_pretrain_backbone", "")
                ModelBuilder._load_pose_weights(model, path)

            return model

        # --- DINOv2 Backbone + Custom Heads ---
        if str(backbone).strip().lower().startswith("dinov2:"):
            if pretrained is False:
                print("[Model] DINOv2 ignores pretrained=False (only pretrained weights are available).")

            variant = str(backbone).split(":", 1)[1].strip()
            hub_name = variant if variant.startswith("dinov2_") else f"dinov2_{variant}"

            try:
                dino = torch.hub.load("facebookresearch/dinov2", hub_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load DINOv2 model '{hub_name}' via torch.hub. "
                    f"Ensure the weights are cached or internet access is available. Error: {e}"
                )

            if not hasattr(dino, "num_features"):
                embed_dim = getattr(dino, "embed_dim", None)
                if embed_dim is not None:
                    dino.num_features = int(embed_dim)

            backbone_wrap = BackboneWrapper(dino)
            in_dim = getattr(backbone_wrap, "num_features", None)
            if in_dim is None:
                raise RuntimeError("DINOv2 backbone does not expose num_features; cannot build heads.")

            head_dropout = float(model_cfg.get("head_dropout", 0.0))

            if head_type == "linear":
                head = LinearHead(in_dim, num_classes, dropout=head_dropout)
            elif head_type == "ml_decoder":
                ml_dim = model_cfg.get("ml_decoder_dim", None)
                ml_dim = int(ml_dim) if ml_dim is not None else int(in_dim)
                ml_heads = int(model_cfg.get("ml_decoder_num_heads", 8))
                ml_dropout = float(model_cfg.get("ml_decoder_dropout", 0.1))
                head = MLDecoderHead(
                    in_dim=in_dim,
                    num_classes=num_classes,
                    d_model=ml_dim,
                    num_heads=ml_heads,
                    dropout=ml_dropout,
                )
            else:
                raise ValueError(f"Unknown head_type: {head_type}")

            aux_head = None
            if aux_enabled:
                aux_dropout = float(aux_cfg.get("dropout", head_dropout))
                aux_head = TotalScoreHead(in_dim, dropout=aux_dropout)

            model = MultiTaskModel(backbone_wrap, head, aux_head=aux_head)
            model.to(device)
            return model

        # --- Timm Backbone + Custom Heads ---
        backbone_model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        use_grad_checkpointing = bool(model_cfg.get("grad_checkpointing", False))
        if use_grad_checkpointing:
            if hasattr(backbone_model, "set_grad_checkpointing"):
                backbone_model.set_grad_checkpointing(True)
                print("[Model] Gradient checkpointing enabled for backbone.")
            else:
                print("[Model] Warning: gradient checkpointing requested but unsupported by this backbone.")
        backbone_wrap = BackboneWrapper(backbone_model)
        in_dim = getattr(backbone_wrap, "num_features", None)
        if in_dim is None:
            raise RuntimeError("Backbone does not expose num_features; cannot build heads.")

        head_dropout = float(model_cfg.get("head_dropout", 0.0))

        if head_type == "linear":
            head = LinearHead(in_dim, num_classes, dropout=head_dropout)
        elif head_type == "ml_decoder":
            ml_dim = model_cfg.get("ml_decoder_dim", None)
            ml_dim = int(ml_dim) if ml_dim is not None else int(in_dim)
            ml_heads = int(model_cfg.get("ml_decoder_num_heads", 8))
            ml_dropout = float(model_cfg.get("ml_decoder_dropout", 0.1))
            head = MLDecoderHead(
                in_dim=in_dim,
                num_classes=num_classes,
                d_model=ml_dim,
                num_heads=ml_heads,
                dropout=ml_dropout,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        aux_head = None
        if aux_enabled:
            aux_dropout = float(aux_cfg.get("dropout", head_dropout))
            aux_head = TotalScoreHead(in_dim, dropout=aux_dropout)

        model = MultiTaskModel(backbone_wrap, head, aux_head=aux_head)
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
        if len(missing) > 0 and hasattr(model, "backbone"):
            # Try loading into backbone only (useful when wrapper prefixes keys)
            target = model.backbone
            if hasattr(target, "backbone"):
                target = target.backbone
            missing, unexpected = target.load_state_dict(state, strict=False)
        print(f"[PoseInit] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
