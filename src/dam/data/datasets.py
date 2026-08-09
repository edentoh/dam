import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Callable

from dam.utils.image import load_rgb_image

class DAMDataset(Dataset):
    """
    Standard dataset for training/validation.
    Returns: (image_tensor, label_tensor, image_id)
    """
    def __init__(self, items: List[Tuple[str, float, str]], transform: Optional[Callable] = None):
        # items structure: [(path_str, label_array, img_id), ...]
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y, img_id = self.items[idx]
        img = load_rgb_image(path)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(y, dtype=torch.float32), img_id


class InferenceDataset(Dataset):
    """
    Dataset for inference (no labels required).
    Returns: (image_id, image_tensor)
    """
    def __init__(self, items: List[Tuple[str, Path]], transform: Optional[Callable] = None):
        # items structure: [(img_id, path_object), ...]
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_id, path = self.items[idx]
        img = load_rgb_image(path)
        
        if self.transform:
            x = self.transform(img)
        else:
            x = img
            
        return img_id, x
