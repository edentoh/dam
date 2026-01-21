import torch
import timm
from pathlib import Path
from timm.data import resolve_data_config, create_transform

class BinaryGate:
    """
    Wraps the 'Is-DAM' binary classifier.
    """
    def __init__(self, model_path: str, device: torch.device, backbone: str = "convnextv2_tiny"):
        self.device = device
        self.path = Path(model_path)
        
        print(f"[BinaryGate] Loading from {self.path}...")
        if not self.path.exists():
            raise FileNotFoundError(f"Binary gate model not found: {self.path}")

        ckpt = torch.load(self.path, map_location="cpu")
        
        # Resolve config from checkpoint if available
        ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        self.backbone = str(ckpt_cfg.get("backbone", backbone))
        self.img_size = int(ckpt_cfg.get("img_size", 224))

        # Extract weights
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt

        # Build Model
        self.model = timm.create_model(self.backbone, pretrained=False, num_classes=1)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Build Transforms
        data_cfg = resolve_data_config({}, model=self.model)
        data_cfg["input_size"] = (3, self.img_size, self.img_size)
        self.transform = create_transform(**data_cfg, is_training=False)

    def predict(self, img) -> float:
        """Returns probability (0.0 to 1.0) that image is valid."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).squeeze().item()
        return float(prob)