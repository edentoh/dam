#!/usr/bin/env python3
"""filter_by_base_gate.py

Filter a folder of images through the *base* (non-ML) DAM gate.

Goal
----
You provide a large pool of images in `false_images_base/`.
This script runs `gating.gate_image()` on each image and copies/moves the
images that *PASS* the base gate into a new folder (hard negatives).

It also writes:
  - a JSONL report with per-file gate results
  - a summary JSON with counts per gate code

Examples
--------
# Copy passed images into ./false_images_passed_base_gate
python filter_by_base_gate.py --input false_images_base

# Specify output and also save failed images grouped by failure code
python filter_by_base_gate.py --input false_images_base \
  --output false_images_passed_base_gate \
  --failed-out false_images_failed_base_gate

# Move passed images (instead of copying)
python filter_by_base_gate.py --input false_images_base --move

Notes
-----
- Thresholds are read from .env (python-dotenv) if present, using the same
  variable names you configured for the server gate.
- This script uses ONLY the image-level gate (no model inference).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from pathlib import Path
import sys

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Local module from your repo
from dam.gating import gate_image


IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def _as_float(v: Optional[str], default: float) -> float:
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _as_int(v: Optional[str], default: int) -> int:
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def load_gate_cfg_from_env() -> Dict[str, Any]:
    """Load the image-gate config from environment variables (.env supported)."""
    cfg: Dict[str, Any] = {
        # Primary knobs (you already listed these in .env)
        "min_ink_frac": _as_float(os.getenv("DAM_GATE_MIN_INK_FRAC"), 0.002),
        "max_ink_frac": _as_float(os.getenv("DAM_GATE_MAX_INK_FRAC"), 0.45),
    }

    # Optional extra knobs (only used if you add them to .env)
    cfg["min_side_px"] = _as_int(os.getenv("DAM_GATE_MIN_SIDE_PX"), 256)
    cfg["min_mean_gray"] = _as_float(os.getenv("DAM_GATE_MIN_MEAN_GRAY"), 90.0)
    cfg["min_white_frac"] = _as_float(os.getenv("DAM_GATE_MIN_WHITE_FRAC"), 0.25)
    cfg["min_edge_density"] = _as_float(os.getenv("DAM_GATE_MIN_EDGE_DENSITY"), 0.004)
    cfg["min_soft_passes"] = _as_int(os.getenv("DAM_GATE_MIN_SOFT_PASSES"), 1)

    return cfg


def iter_image_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def safe_open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter images by DAM base gate (gating.gate_image).")
    parser.add_argument("--input", "-i", required=True, help="Input folder (e.g., false_images_base)")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Folder to place images that PASS the base gate. Default: <input>_passed_base_gate",
    )
    parser.add_argument(
        "--failed-out",
        default=None,
        help="Optional folder to copy/move images that FAIL the base gate, grouped by failure code.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="JSONL report path. Default: <output>/base_gate_report.jsonl",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Summary JSON path. Default: <output>/base_gate_summary.json",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subfolders")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Do not preserve relative subfolder structure in output folders",
    )
    args = parser.parse_args()

    load_dotenv()  # loads .env if present

    in_dir = Path(args.input).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input folder not found: {in_dir}")

    out_dir = Path(args.output).expanduser().resolve() if args.output else in_dir.with_name(in_dir.name + "_passed_base_gate")
    out_dir.mkdir(parents=True, exist_ok=True)

    failed_dir = Path(args.failed_out).expanduser().resolve() if args.failed_out else None
    if failed_dir:
        failed_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report).expanduser().resolve() if args.report else out_dir / "base_gate_report.jsonl"
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else out_dir / "base_gate_summary.json"

    cfg = load_gate_cfg_from_env()
    copier = shutil.move if args.move else shutil.copy2

    files = list(iter_image_files(in_dir, recursive=not args.no_recursive))
    if not files:
        print(f"No images found in {in_dir} (extensions: {sorted(IMG_EXTS)})")
        return 0

    counts = Counter()
    passed = 0
    failed = 0
    errors = 0

    ensure_parent(report_path)
    with report_path.open("w", encoding="utf-8") as f_report:
        for src in tqdm(files, desc="Gating images", unit="img"):
            rel = src.name if args.flat else str(src.relative_to(in_dir))

            try:
                img = safe_open_rgb(src)
                res = gate_image(img, cfg=cfg).to_dict()
                res_record = {"path": rel, "abs_path": str(src), **res}

                if res["ok"]:
                    dst = out_dir / rel
                    ensure_parent(dst)
                    copier(str(src), str(dst))
                    passed += 1
                    counts["ok"] += 1
                else:
                    failed += 1
                    counts[res.get("code", "fail")] += 1
                    if failed_dir is not None:
                        code = res.get("code", "fail")
                        dst = failed_dir / code / rel
                        ensure_parent(dst)
                        copier(str(src), str(dst))

                f_report.write(json.dumps(res_record, ensure_ascii=False) + "\n")

            except Exception as e:
                errors += 1
                counts["error"] += 1
                err_record = {
                    "path": rel,
                    "abs_path": str(src),
                    "ok": False,
                    "code": "error",
                    "message": f"Exception: {type(e).__name__}: {e}",
                    "metrics": {},
                }
                f_report.write(json.dumps(err_record, ensure_ascii=False) + "\n")

    summary = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "failed_out_dir": str(failed_dir) if failed_dir else None,
        "report": str(report_path),
        "summary": str(summary_path),
        "recursive": not args.no_recursive,
        "moved": bool(args.move),
        "flat": bool(args.flat),
        "cfg": cfg,
        "num_total": len(files),
        "num_passed": passed,
        "num_failed": failed,
        "num_errors": errors,
        "counts_by_code": dict(counts),
    }

    write_json(summary_path, summary)

    print("\nDone")
    print(json.dumps({k: summary[k] for k in ["num_total", "num_passed", "num_failed", "num_errors", "counts_by_code"]}, indent=2))
    print(f"Report:  {report_path}")
    print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
