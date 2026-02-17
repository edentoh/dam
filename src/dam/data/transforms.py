import numpy as np
from PIL import Image
from torchvision import transforms

class CropToInk:
    """
    Pre-processing: Crops image to ink bounding box.
    Uses adaptive thresholding to handle dark/shadowed images.
    """
    def __init__(self, pad: int = 12, min_size: int = 50, fixed_threshold: int = None):
        self.fixed_threshold = fixed_threshold
        self.pad = int(pad)
        self.min_size = int(min_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert to grayscale
        g = img.convert("L")
        arr = np.array(g)

        # --- Adaptive Logic ---
        if self.fixed_threshold is not None:
            thresh = self.fixed_threshold
        else:
            # Estimate paper brightness (95th percentile ignores the dark ink)
            bg_est = np.percentile(arr, 95)
            # Set threshold 45 units below the paper brightness
            thresh = max(0, int(bg_est - 45))
        # ----------------------

        # Create mask: True where pixels are DARKER than threshold (ink)
        mask = arr < thresh

        # Safety check: if image is blank or noise, return original
        if int(mask.sum()) < self.min_size:
            return img

        # Find coordinates of the ink pixels
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        # Add padding
        y0 = max(0, y0 - self.pad)
        x0 = max(0, x0 - self.pad)
        y1 = min(arr.shape[0] - 1, y1 + self.pad)
        x1 = min(arr.shape[1] - 1, x1 + self.pad)

        return img.crop((x0, y0, x1 + 1, y1 + 1))


def build_transforms(cfg: dict, is_train: bool = False):
    """
    Factory function to build the appropriate transform pipeline.
    Reads settings from the 'data' or 'predict.data' config section.
    """
    # Try to find relevant data config section
    if "predict" in cfg and not is_train:
        data_cfg = cfg.get("predict", {}).get("data", cfg.get("data", {}))
    elif "train" in cfg and is_train:
         data_cfg = cfg.get("train", {}).get("data", cfg.get("data", {}))
    else:
        data_cfg = cfg.get("data", {})

    ops = []

    # 1. Optional Custom Crop
    if data_cfg.get("use_crop_to_ink", False):
        ops.append(
            CropToInk(
                pad=int(data_cfg.get("crop_pad", 12)),
                min_size=int(data_cfg.get("crop_min_size", 50)),
                fixed_threshold=data_cfg.get("crop_threshold", None) # Optional fixed override
            )
        )

    # 2. Standard Preprocessing
    ops.append(transforms.Grayscale(num_output_channels=3))
    size = int(data_cfg.get("img_size", 384))

    if is_train:
        # Training Augmentations
        ops.extend([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomApply([transforms.ColorJitter(0.15, 0.15)], p=0.5),
            transforms.RandomAffine(12, (0.03, 0.03), (0.95, 1.05), 3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        # Validation/Inference Deterministic Transforms
        ops.extend([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    return transforms.Compose(ops)