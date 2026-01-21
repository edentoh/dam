import random
import os
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    Seeds standard Python random, NumPy, and PyTorch (CPU & CUDA).
    Sets determinism flags for CuDNN.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    