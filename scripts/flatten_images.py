#!/usr/bin/env python3
"""
flatten_images.py

Flatten a directory tree:
- Move (or copy) all images into a single output folder.
- Remove non-image files (default: move to quarantine; optional: delete).
- Optionally delete empty directories afterwards.

Usage examples:
  # Dry-run first (recommended)
  python flatten_images.py --root false_images_base --out false_images_base_flat --dry-run

  # Move images, quarantine non-images, delete empty dirs
  python flatten_images.py --root false_images_base --out false_images_base_flat --cleanup-empty-dirs

  # Move images, DELETE non-images (irreversible)
  python flatten_images.py --root false_images_base --out false_images_base_flat --delete-non-images --cleanup-empty-dirs
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Set

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


IMG_EXTS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

OTHER_COMMON_EXTS: Set[str] = {
    ".json", ".csv", ".xlsx", ".xls", ".txt", ".yaml", ".yml", ".toml",
    ".pdf", ".doc", ".docx", ".ppt", ".pptx"
}


def is_image_file(path: Path, verify: bool = True) -> bool:
    if path.suffix.lower() not in IMG_EXTS:
        return False
    if verify and PIL_AVAILABLE:
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception:
            return False
    return True


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def unique_dest(dest_dir: Path, filename: str) -> Path:
    """Avoid overwriting by adding _001, _002..."""
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dest_dir / (base + ext)
    if not candidate.exists():
        return candidate

    i = 1
    while True:
        candidate = dest_dir / f"{base}_{i:03d}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing subfolders/files to flatten")
    ap.add_argument("--out", default=None, help="Output folder for all images (single folder). Default: <root>_flat_images")
    ap.add_argument(
        "--non-image-out",
        default=None,
        help="Folder to move non-image files into (quarantine). Default: <root>_removed_non_images",
    )
    ap.add_argument("--copy", action="store_true", help="Copy instead of move (default moves)")
    ap.add_argument("--delete-non-images", action="store_true", help="Delete non-image files instead of quarantining")
    ap.add_argument("--no-verify", action="store_true", help="Do not verify images with PIL (extension-only)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions but do not modify files")
    ap.add_argument("--cleanup-empty-dirs", action="store_true", help="Remove empty directories after operations")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root folder not found: {root}")

    out_dir = Path(args.out).expanduser().resolve() if args.out else root.with_name(root.name + "_flat_images")
    non_img_dir = (
        Path(args.non_image_out).expanduser().resolve()
        if args.non_image_out
        else root.with_name(root.name + "_removed_non_images")
    )

    verify = not args.no_verify

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.delete_non_images:
            non_img_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    non_img_count = 0
    skipped_out_count = 0

    for p in iter_files(root):
        # Skip anything already inside output/quarantine directories if they live under root
        try:
            if out_dir in p.parents:
                skipped_out_count += 1
                continue
            if (not args.delete_non_images) and (non_img_dir in p.parents):
                skipped_out_count += 1
                continue
        except Exception:
            pass

        if is_image_file(p, verify=verify):
            dest = unique_dest(out_dir, p.name)
            img_count += 1
            action = "COPY" if args.copy else "MOVE"
            print(f"{action} IMAGE: {p} -> {dest}")
            if not args.dry_run:
                ensure_parent(dest)
                if args.copy:
                    shutil.copy2(str(p), str(dest))
                else:
                    shutil.move(str(p), str(dest))
        else:
            non_img_count += 1
            if args.delete_non_images:
                print(f"DELETE NON-IMAGE: {p}")
                if not args.dry_run:
                    try:
                        p.unlink()
                    except Exception as e:
                        print(f"  !! Failed delete: {p} ({type(e).__name__}: {e})")
            else:
                # quarantine while preserving relative path
                rel = p.relative_to(root)
                dest = non_img_dir / rel
                print(f"MOVE NON-IMAGE: {p} -> {dest}")
                if not args.dry_run:
                    ensure_parent(dest)
                    shutil.move(str(p), str(dest))

    if args.cleanup_empty_dirs:
        # Remove empty dirs bottom-up under root (excluding out_dir / non_img_dir if under root)
        for d in sorted([x for x in root.rglob("*") if x.is_dir()], key=lambda x: len(str(x)), reverse=True):
            # don’t delete output/quarantine dirs if they ended up under root
            if d == out_dir or d == non_img_dir:
                continue
            if out_dir in d.parents or non_img_dir in d.parents:
                continue
            try:
                if not any(d.iterdir()):
                    print(f"RMDIR EMPTY: {d}")
                    if not args.dry_run:
                        d.rmdir()
            except Exception:
                pass

    print("\nSummary")
    print(f"  Root:            {root}")
    print(f"  Images ->        {out_dir}")
    if args.delete_non_images:
        print(f"  Non-images:      deleted")
    else:
        print(f"  Non-images ->    {non_img_dir}")
    print(f"  Images moved/copied: {img_count}")
    print(f"  Non-images handled:  {non_img_count}")
    print(f"  Skipped (already in out dirs): {skipped_out_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
