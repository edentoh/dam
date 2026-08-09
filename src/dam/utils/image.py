from pathlib import Path
from typing import BinaryIO, Union

from PIL import Image, ImageOps

ImageSource = Union[str, Path, BinaryIO]


def load_rgb_image(source: ImageSource) -> Image.Image:
    """
    Open an image, apply EXIF orientation if present, and return an RGB PIL image.
    """
    with Image.open(source) as img:
        return ImageOps.exif_transpose(img).convert("RGB")
