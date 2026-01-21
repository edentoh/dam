"""
make_binary_labels.py

Create labels for the binary Gate Model.
- Positives (1): Real DAM drawings (e.g. from your labeled dataset)
- Negatives (0): Non-DAM images that passed the heuristic gate (Hard Negatives)

Usage:
  python -m scripts.make_binary_labels --pos image_cropped --neg false_images_passed --out labels_gate
"""
import argparse
import hashlib
import json
import shutil
import random
from pathlib import Path

# --- Shared Imports ---
from dam.core.constants import IMG_EXTS

def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", required=True, help="Folder of Real DAMs (Label 1)")
    parser.add_argument("--neg", required=True, help="Folder of Non-DAMs (Label 0)")
    parser.add_argument("--out", default="labels_gate", help="Output folder")
    parser.add_argument("--mode", choices=["keep", "copy"], default="keep", help="Copy images or just reference paths")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    project_root = Path.cwd()
    pos_dir = Path(args.pos).resolve()
    neg_dir = Path(args.neg).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_files = list(iter_images(pos_dir))
    neg_files = list(iter_images(neg_dir))
    
    if not pos_files or not neg_files:
        raise SystemExit(f"Found {len(pos_files)} positives and {len(neg_files)} negatives. Need both.")

    print(f"Found {len(pos_files)} positives, {len(neg_files)} negatives.")

    rows = []
    
    # Process Positives
    for p in pos_files:
        if args.mode == "copy":
            # Copy to out_dir/images with unique name
            ext = p.suffix.lower()
            name = f"dam_{sha1_short(p.stem)}{ext}"
            dst = out_dir / "images" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            # CSV path relative to project root
            p_rel = dst.relative_to(project_root)
        else:
            try:
                p_rel = p.relative_to(project_root)
            except ValueError:
                p_rel = p # Absolute if outside root
        
        rows.append((str(p_rel), 1))

    # Process Negatives
    for p in neg_files:
        if args.mode == "copy":
            ext = p.suffix.lower()
            name = f"neg_{sha1_short(p.stem)}{ext}"
            dst = out_dir / "images" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            p_rel = dst.relative_to(project_root)
        else:
            try:
                p_rel = p.relative_to(project_root)
            except ValueError:
                p_rel = p
        
        rows.append((str(p_rel), 0))

    # Shuffle
    random.seed(args.seed)
    random.shuffle(rows)

    # Write CSV
    out_csv = out_dir / "binary_labels.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("path,label\n")
        for path, label in rows:
            f.write(f"{path},{label}\n")

    # Write Summary
    summary = {
        "pos_source": str(pos_dir),
        "neg_source": str(neg_dir),
        "num_pos": len(pos_files),
        "num_neg": len(neg_files),
        "total": len(rows),
        "mode": args.mode
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Created labels at: {out_csv}")
    print(f"Total: {len(rows)} images.")

if __name__ == "__main__":
    main()