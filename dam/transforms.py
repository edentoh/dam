import numpy as np
from PIL import Image


class CropToInk:
    """Pre-processing: Crops image to ink bounding box."""

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
