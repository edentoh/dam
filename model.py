<<<<<<< HEAD
import torch
import torch.nn as nn
import timm
from pathlib import Path

class ModelBuilder:
    @staticmethod
    def build(cfg: dict, device: torch.device):
        model_cfg = cfg['model']
        backbone = model_cfg['backbone']
        num_classes = model_cfg.get('num_classes', 48)
        pretrained = model_cfg.get('pretrained', True)

        print(f"Building model: {backbone} (pretrained={pretrained})")
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes
        ).to(device)

        if model_cfg.get('use_pose_pretrain', False):
            path = model_cfg.get('pose_pretrain_backbone', "")
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
        state = ckpt.get("backbone_state", ckpt) # Fallback if key missing
        
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[PoseInit] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
=======
import torch
import torch.nn as nn
import timm
from pathlib import Path

class ModelBuilder:
    @staticmethod
    def build(cfg: dict, device: torch.device):
        model_cfg = cfg['model']
        backbone = model_cfg['backbone']
        num_classes = model_cfg.get('num_classes', 48)
        pretrained = model_cfg.get('pretrained', True)

        print(f"Building model: {backbone} (pretrained={pretrained})")
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes
        ).to(device)

        if model_cfg.get('use_pose_pretrain', False):
            path = model_cfg.get('pose_pretrain_backbone', "")
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
        state = ckpt.get("backbone_state", ckpt) # Fallback if key missing
        
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[PoseInit] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
>>>>>>> d01af25 (update files)
