import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader

# Internal
from dam.modeling.builder import ModelBuilder
from dam.data.datasets import InferenceDataset
from dam.data.transforms import build_transforms
from .utils import safe_num_workers
from .calibration import CorrelationCalibrator

class DAMPredictor:
    """
    High-level interface for running inference.
    Handles model instantiation, checkpoint loading, and batch processing.
    """
    def __init__(self, model_path: Path, cfg: dict, device: torch.device):
        self.device = device
        self.cfg = cfg
        self.model_path = model_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"[Predictor] Loading model from {self.model_path}")
        ckpt = torch.load(self.model_path, map_location="cpu")
        
        # Determine image size from checkpoint or config
        data_cfg = cfg.get("predict", {}).get("data", cfg.get("data", {}))
        self.img_size = int(ckpt.get("img_size", data_cfg.get("img_size", 384)))

        # Build Model
        num_classes = int(cfg["model"].get("num_classes", 48))
        cfg_build = deepcopy(cfg)
        cfg_build.setdefault("model", {})
        cfg_build["model"]["pretrained"] = False
        cfg_build["model"]["use_pose_pretrain"] = False
        self.model = ModelBuilder.build(cfg_build, self.device)
        
        # Load State
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Build Transforms
        # We pass the config. build_transforms looks for [predict.data] automatically if is_train=False
        self.transform = build_transforms(cfg, is_train=False)
        self.calibrator = CorrelationCalibrator(cfg, self.model_path, num_classes)

    def predict_batch(self, items: list) -> dict:
        """
        Runs inference on a list of (img_id, path) tuples.
        Returns: dict {img_id: prob_numpy_array}
        """
        if not items:
            return {}

        batch_size = int(self.cfg.get("predict", {}).get("batch_size", 16))
        num_workers = safe_num_workers(int(self.cfg.get("predict", {}).get("data", {}).get("num_workers", 0)))
        
        dataset = InferenceDataset(items, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda")
        )

        probs_by_id = {}
        
        with torch.no_grad():
            for img_ids, x in loader:
                x = x.to(self.device, non_blocking=True)
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs)
                elif isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                else:
                    logits = outputs
                # Sigmoid -> CPU -> Numpy
                probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                probs = self.calibrator.apply(probs)
                
                for i, img_id in enumerate(img_ids):
                    probs_by_id[str(img_id)] = probs[i]
        
        return probs_by_id
