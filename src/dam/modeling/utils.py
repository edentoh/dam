import torch.nn as nn

def get_module_by_path(root: nn.Module, path: str):
    """Resolves dotted attribute paths (e.g., 'head.fc') against a module."""
    cur = root
    for part in str(path).split("."):
        if not part:
            continue
        cur = getattr(cur, part)
    return cur

def resolve_classifier_modules(model: nn.Module) -> list[nn.Module]:
    """
    Best-effort resolution of a model's classifier/head modules.
    Useful for applying discriminative learning rates.
    """
    if not hasattr(model, "get_classifier"):
        return []

    cls = model.get_classifier()
    if cls is None:
        return []

    modules = []
    if isinstance(cls, str):
        try:
            modules.append(get_module_by_path(model, cls))
        except Exception:
            return []
    elif isinstance(cls, (list, tuple)):
        for item in cls:
            if item is None:
                continue
            if isinstance(item, str):
                try:
                    modules.append(get_module_by_path(model, item))
                except Exception:
                    continue
            else:
                modules.append(item)
    else:
        modules.append(cls)

    out = []
    for m in modules:
        if hasattr(m, "parameters"):
            out.append(m)
    return out

def infer_in_channels(model: nn.Module) -> int:
    """
    Inspects the first Conv2d layer to determine input channels.
    Useful for building matching transforms.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.in_channels)
    return 3