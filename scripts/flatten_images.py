"""
flatten_images.py

Flatten a directory tree:
- Move (or copy) all images into a single output folder.
- Remove non-image files (quarantine or delete).
- Useful for raw datasets downloaded from the web.

Usage:
  python -m scripts.flatten_images --root raw_downloads --out dataset_flat --dry-run
"""
import argparse
import shutil
from pathlib import Path
from typing import Iterable, Set

# --- Shared Imports ---
from dam.core.constants import IMG_EXTS

# Optional: Verify images with PIL if available
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder to flatten")
    ap.add_argument("--out", default=None, help="Output folder")
    ap.add_argument("--non-image-out", default=None, help="Quarantine folder for non-images")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move")
    ap.add_argument("--delete-non-images", action="store_true", help="Delete non-images")
    ap.add_argument("--no-verify", action="store_true", help="Skip PIL verification")
    ap.add_argument("--dry-run", action="store_true", help="Simulate only")
    ap.add_argument("--cleanup-empty-dirs", action="store_true", help="Remove empty folders")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    out_dir = Path(args.out).expanduser().resolve() if args.out else root.with_name(root.name + "_flat")
    non_img_dir = (
        Path(args.non_image_out).expanduser().resolve()
        if args.non_image_out
        else root.with_name(root.name + "_quarantine")
    )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.delete_non_images:
            non_img_dir.mkdir(parents=True, exist_ok=True)

    counts = {"img": 0, "non_img": 0, "skip": 0}
    verify = not args.no_verify

    for p in iter_files(root):
        # Skip if already in output/quarantine
        if out_dir in p.parents or (not args.delete_non_images and non_img_dir in p.parents):
            counts["skip"] += 1
            continue

        if is_image_file(p, verify=verify):
            dest = unique_dest(out_dir, p.name)
            counts["img"] += 1
            print(f"[IMG] {p.name} -> {dest.name}")
            if not args.dry_run:
                shutil.copy2(p, dest) if args.copy else shutil.move(p, dest)
        else:
            counts["non_img"] += 1
            if args.delete_non_images:
                print(f"[DEL] {p.name}")
                if not args.dry_run:
                    try: p.unlink()
                    except: pass
            else:
                rel = p.relative_to(root)
                dest = non_img_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"[OTHER] {p.name} -> {dest}")
                if not args.dry_run:
                    shutil.move(p, dest)

    if args.cleanup_empty_dirs and not args.dry_run:
        for d in sorted([x for x in root.rglob("*") if x.is_dir()], key=lambda x: len(str(x)), reverse=True):
            if d == out_dir or d == non_img_dir: continue
            try: 
                if not any(d.iterdir()): d.rmdir()
            except: pass

    print(f"\nDone. Images: {counts['img']}, Non-images: {counts['non_img']}, Skipped: {counts['skip']}")

if __name__ == "__main__":
    main()