"""
filter_by_gate.py

Filter a folder of images through the *base* (non-ML) DAM gate.
Use this to create "Hard Negatives" (images that pass basic checks but aren't DAMs)
for training the Gate Model.

Usage:
  python -m scripts.filter_by_gate --input raw_downloads --output clean_candidates
"""
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv

# --- Shared Imports ---
from dam.core.constants import IMG_EXTS
from dam.gating.heuristics import gate_image_heuristics, GateResult

def load_gate_cfg_from_env() -> dict:
    """Load config matching the API logic."""
    def _env(key, default, type_):
        v = os.getenv(key)
        return type_(v) if v else default

    return {
        "min_ink_frac": _env("DAM_GATE_MIN_INK_FRAC", 0.002, float),
        "max_ink_frac": _env("DAM_GATE_MAX_INK_FRAC", 0.45, float),
        "min_side_px": _env("DAM_GATE_MIN_SIDE_PX", 256, int),
        "min_mean_gray": _env("DAM_GATE_MIN_MEAN_GRAY", 60.0, float),
        "min_white_frac": _env("DAM_GATE_MIN_WHITE_FRAC", 0.25, float),
        "min_edge_density": _env("DAM_GATE_MIN_EDGE_DENSITY", 0.004, float),
        "min_soft_passes": _env("DAM_GATE_MIN_SOFT_PASSES", 1, int),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input folder")
    parser.add_argument("--output", "-o", default=None, help="Output folder for PASSED images")
    parser.add_argument("--failed-out", default=None, help="Output folder for FAILED images")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    parser.add_argument("--no-recursive", action="store_true", help="Flat search")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_gate_cfg_from_env()

    in_dir = Path(args.input).expanduser().resolve()
    if not in_dir.exists():
        raise SystemExit(f"Input not found: {in_dir}")

    out_dir = Path(args.output).expanduser().resolve() if args.output else in_dir.with_name(in_dir.name + "_passed")
    out_dir.mkdir(parents=True, exist_ok=True)

    failed_dir = Path(args.failed_out).expanduser().resolve() if args.failed_out else None
    
    files = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    
    print(f"Filtering {len(files)} images using DAM heuristics...")
    counts = Counter()
    copier = shutil.move if args.move else shutil.copy2

    report_path = out_dir / "gate_report.jsonl"
    
    with report_path.open("w", encoding="utf-8") as f_report:
        for src in tqdm(files):
            try:
                # Use the Shared Gating Logic
                with Image.open(src) as img:
                    img = img.convert("RGB")
                    # We create a copy because gate_image might read pixel data lazily
                    # and we want to keep the file handle clean or just rely on PIL's load.
                    img.load() 
                    res = gate_image_heuristics(img, cfg=cfg)

                rec = {"path": str(src), **res.to_dict()}
                f_report.write(json.dumps(rec) + "\n")

                if res.ok:
                    counts["pass"] += 1
                    rel = src.relative_to(in_dir)
                    dst = out_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    copier(src, dst)
                else:
                    counts[res.code] += 1
                    if failed_dir:
                        rel = src.relative_to(in_dir)
                        # Organize failures by code (e.g. failed/too_dark/img.jpg)
                        dst = failed_dir / res.code / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        copier(src, dst)

            except Exception as e:
                counts["error"] += 1
                print(f"Error processing {src.name}: {e}")

    print("\nResults:")
    print(f"  Passed: {counts['pass']}")
    print(f"  Failed details: {dict(counts)}")
    print(f"  Report saved to: {report_path}")

if __name__ == "__main__":
    main()