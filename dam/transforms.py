import numpy as np
from PIL import Image

class CropToInk:
    """
    Pre-processing: Crops image to ink bounding box.
    Uses adaptive thresholding to handle dark/shadowed images.
    """

    def __init__(self, pad: int = 12, min_size: int = 50, fixed_threshold: int = None):
        # If fixed_threshold is set (e.g. 245), it behaves like the old version.
        # If None, it uses adaptive logic.
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