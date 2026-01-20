#!/usr/bin/env python3
"""
Create binary labels for a DAM-valid gate classifier.

Defaults:
  - positives: image_cropped (label=1)
  - negatives: false_images (label=0)

Outputs:
  - <out>/binary_labels.csv   (path,label)
  - <out>/binary_labels.jsonl (optional)
  - <out>/binary_labels_summary.json

Modes:
  --mode keep (default): keep original files; CSV paths are relative to project root
  --mode copy: copy images into <out>/images/ and write labels against copied paths

Example:
  python scripts/make_binary_labels.py --pos image_cropped --neg false_images --out labels_gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(root: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_to_project_root(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def copy_unique(src: Path, dst_dir: Path, prefix: str, project_root: Path) -> str:
    """
    Copy src into dst_dir with a stable unique name to avoid collisions.
    Returns relative path (posix) to the copied file from project root.
    """
    rel = rel_to_project_root(src, project_root)
    h = sha1_short(rel)
    ext = src.suffix.lower()
    name = f"{prefix}_{h}{ext}"
    dst = dst_dir / name
    shutil.copy2(str(src), str(dst))
    return rel_to_project_root(dst, project_root)


def write_csv(rows: List[Tuple[str, int]], out_csv: Path) -> None:
    out_csv.write_text(
        "path,label\n" + "\n".join([f"{p},{y}" for p, y in rows]) + "\n",
        encoding="utf-8",
    )


def write_jsonl(records: List[Dict], out_jsonl: Path) -> None:
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="image_cropped", help="Positive folder (DAM drawings), label=1")
    ap.add_argument("--neg", default="false_images", help="Negative folder (non-DAM), label=0")
    ap.add_argument("--out", default="labels_gate", help="Output folder for labels")
    ap.add_argument("--mode", choices=["keep", "copy"], default="keep", help="keep=reference originals, copy=copy into one folder")
    ap.add_argument("--no-recursive", action="store_true", help="Do not recurse into subfolders")
    ap.add_argument("--write-jsonl", action="store_true", help="Also write binary_labels.jsonl")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle rows (deterministic)")
    ap.add_argument("--seed", type=int, default=1337, help="Seed for deterministic shuffle")
    args = ap.parse_args()

    project_root = Path.cwd().resolve()
    pos_dir = Path(args.pos).expanduser().resolve()
    neg_dir = Path(args.neg).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not pos_dir.exists():
        raise SystemExit(f"Positive folder not found: {pos_dir}")
    if not neg_dir.exists():
        raise SystemExit(f"Negative folder not found: {neg_dir}")

    ensure_dir(out_dir)

    recursive = not args.no_recursive

    pos_files = sorted(iter_images(pos_dir, recursive=recursive))
    neg_files = sorted(iter_images(neg_dir, recursive=recursive))

    if len(pos_files) == 0:
        raise SystemExit(f"No images found under positives: {pos_dir}")
    if len(neg_files) == 0:
        raise SystemExit(f"No images found under negatives: {neg_dir}")

    rows: List[Tuple[str, int]] = []
    records: List[Dict] = []

    if args.mode == "copy":
        img_out = out_dir / "images"
        ensure_dir(img_out)

        for p in pos_files:
            new_rel = copy_unique(p, img_out, prefix="dam", project_root=project_root)
            rows.append((new_rel, 1))
        for p in neg_files:
            new_rel = copy_unique(p, img_out, prefix="false", project_root=project_root)
            rows.append((new_rel, 0))
    else:
        # keep originals, write relative-to-project-root paths where possible
        for p in pos_files:
            rows.append((rel_to_project_root(p, project_root), 1))
        for p in neg_files:
            rows.append((rel_to_project_root(p, project_root), 0))

    # Optional deterministic shuffle
    if args.shuffle:
        import random
        rnd = random.Random(args.seed)
        rnd.shuffle(rows)

    # Build JSONL records if requested
    if args.write_jsonl:
        for path, label in rows:
            records.append({"path": path, "label": label})

    out_csv = out_dir / "binary_labels.csv"
    write_csv(rows, out_csv)

    if args.write_jsonl:
        out_jsonl = out_dir / "binary_labels.jsonl"
        write_jsonl(records, out_jsonl)

    summary = {
        "project_root": str(project_root).replace("\\", "/"),
        "pos_dir": str(pos_dir).replace("\\", "/"),
        "neg_dir": str(neg_dir).replace("\\", "/"),
        "out_dir": str(out_dir).replace("\\", "/"),
        "mode": args.mode,
        "recursive": recursive,
        "num_pos": len(pos_files),
        "num_neg": len(neg_files),
        "num_total": len(rows),
        "label_map": {"dam": 1, "false": 0},
        "csv": str(out_csv).replace("\\", "/"),
    }

    (out_dir / "binary_labels_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Done")
    print(json.dumps({k: summary[k] for k in ["num_pos", "num_neg", "num_total", "mode", "csv"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
