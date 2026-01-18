from __future__ import annotations


def get_predict_data_cfg(cfg: dict) -> dict:
    """New layout: [predict.data], fallback to [data]."""
    pdcfg = cfg.get("predict", {}).get("data", {})
    if pdcfg:
        return pdcfg
    return cfg.get("data", {})


def get_labels_path_for_predict(cfg: dict) -> str | None:
    """Resolve labels_path with priority:

    1) [predict.labels].labels_path
    2) [train.data].labels_path
    3) [data].csv_path (back-compat)
    """
    pl = cfg.get("predict", {}).get("labels", {})
    if isinstance(pl, dict) and pl.get("labels_path"):
        return pl.get("labels_path")

    tl = cfg.get("train", {}).get("data", {})
    if isinstance(tl, dict) and tl.get("labels_path"):
        return tl.get("labels_path")

    d = cfg.get("data", {})
    if isinstance(d, dict) and d.get("csv_path"):
        return d.get("csv_path")

    return None
