"""
Create a frozen train/val split with score-balance + multi-label stratification.

Core behavior:
- Target split: 160 train / 40 val (for ~200 images)
- Total score per image = sum of first 48 checklist labels
- Quantile-bin total score (default 5 bins)
- Multi-label iterative stratification on [48 criteria + score-bin one-hot]
- Hard constraints:
  - Criteria with exactly 1 positive globally are forced to train
  - Criteria with >=3 positives should have >=1 positive in val when feasible
  - Constraint repair swaps are performed within the same score bin
- Split is frozen via manifests: train_ids.txt + val_ids.txt
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Easy-to-edit parameters
# =========================
LABELS_XLSX = Path("labels/Score_j.xlsx")
SOURCE_IMAGES_DIR = Path("image_cropped")
OUTPUT_DATASET_DIR = Path("image_dataset3")

NUM_CRITERIA_ROWS = 48
NUM_SCORE_BINS = 10

TARGET_TRAIN = 160
TARGET_VAL = 40
STRICT_TARGET_COUNTS = True  # if True, requires exactly TARGET_TRAIN+TARGET_VAL available images

SPLIT_SEED = 42
FREEZE_IF_MANIFEST_EXISTS = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OVERWRITE_EXISTING = False

COPY_STATS_FILENAME = "split_summary.json"
MANIFEST_FILENAME = "split_manifest.csv"
TRAIN_IDS_FILENAME = "train_ids.txt"
VAL_IDS_FILENAME = "val_ids.txt"


def extract_numeric_id(text: str) -> int | None:
    m = re.search(r"(\d+)", str(text))
    if not m:
        return None
    return int(m.group(1))


def load_label_rows(labels_xlsx: Path, num_rows: int) -> pd.DataFrame:
    df = pd.read_excel(labels_xlsx, engine="openpyxl")
    image_cols = [c for c in df.columns if isinstance(c, str) and "image" in c.lower()]
    df_criteria = df.iloc[:num_rows].copy()

    rows: list[dict] = []
    for col in image_cols:
        img_id = extract_numeric_id(col)
        if img_id is None:
            continue

        y = pd.to_numeric(df_criteria[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if y.shape[0] != num_rows:
            continue

        row = {
            "image_id": int(img_id),
            "image_col": str(col),
            "total_score": float(y.sum()),
        }
        for j in range(num_rows):
            row[f"c{j}"] = int(y[j] > 0.0)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["image_id", "image_col"]).drop_duplicates("image_id", keep="first")
    out = out.reset_index(drop=True)
    return out


def index_source_images(source_dir: Path) -> dict[int, list[Path]]:
    id_to_paths: dict[int, list[Path]] = {}
    for p in source_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        img_id = extract_numeric_id(p.stem)
        if img_id is None:
            continue
        id_to_paths.setdefault(img_id, []).append(p)
    for paths in id_to_paths.values():
        paths.sort()
    return id_to_paths


def determine_split_sizes(n_images: int) -> tuple[int, int]:
    total_target = TARGET_TRAIN + TARGET_VAL
    if STRICT_TARGET_COUNTS:
        if n_images != total_target:
            raise ValueError(
                f"STRICT_TARGET_COUNTS=True requires exactly {total_target} available labeled images; found {n_images}."
            )
        return TARGET_TRAIN, TARGET_VAL

    val_ratio = TARGET_VAL / float(total_target)
    n_val = int(round(n_images * val_ratio))
    n_val = max(1, min(n_images - 1, n_val))
    n_train = n_images - n_val
    return n_train, n_val


def assign_score_bins(total_scores: np.ndarray, num_bins: int) -> np.ndarray:
    # Rank-based qcut avoids collapsing bins when scores have many ties.
    ranks = pd.Series(total_scores).rank(method="first")
    q = min(max(1, int(num_bins)), int(len(total_scores)))
    bins = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
    return bins.to_numpy(dtype=np.int32)


def iterative_multilabel_val_split(
    y_aug: np.ndarray,
    n_val: int,
    locked_train: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Iterative greedy approximation of multi-label stratification.
    Chooses val indices to match per-label target prevalence.
    """
    n, n_labels = y_aug.shape
    if n_val <= 0 or n_val >= n:
        raise ValueError(f"n_val must be in [1, n-1], got {n_val} for n={n}.")
    if int((~locked_train).sum()) < n_val:
        raise ValueError("Not enough non-locked samples to fill validation split.")

    rng = np.random.default_rng(seed)
    target = y_aug.sum(axis=0).astype(np.float64) * (n_val / float(n))
    label_totals = y_aug.sum(axis=0).astype(np.float64)

    val_mask = np.zeros(n, dtype=bool)
    current = np.zeros(n_labels, dtype=np.float64)

    while int(val_mask.sum()) < n_val:
        remaining = np.where((~val_mask) & (~locked_train))[0]
        if remaining.size == 0:
            break

        deficits = target - current
        rem_pos = y_aug[remaining].sum(axis=0)
        feasible_labels = np.where(rem_pos > 0)[0]

        if feasible_labels.size > 0:
            deficit_labels = feasible_labels[deficits[feasible_labels] > 1e-9]
            pool_labels = deficit_labels if deficit_labels.size > 0 else feasible_labels
            rarest = np.argmin(label_totals[pool_labels])
            focus_label = int(pool_labels[rarest])
            candidates = remaining[y_aug[remaining, focus_label] > 0]
            if candidates.size == 0:
                candidates = remaining
        else:
            candidates = remaining

        gain = np.maximum(deficits, 0.0)
        sub = y_aug[candidates]
        cover = sub @ gain
        overshoot = np.maximum((current + sub) - target, 0.0).sum(axis=1)
        rarity = (sub * (1.0 / np.clip(label_totals, 1.0, None))).sum(axis=1)

        noise = rng.uniform(0.0, 1e-6, size=candidates.shape[0])
        score = cover + 0.05 * rarity - 0.2 * overshoot + noise
        pick = int(candidates[int(np.argmax(score))])

        val_mask[pick] = True
        current += y_aug[pick]

    while int(val_mask.sum()) < n_val:
        remaining = np.where((~val_mask) & (~locked_train))[0]
        if remaining.size == 0:
            break
        sub = y_aug[remaining]
        penalty = np.maximum((current + sub) - target, 0.0).sum(axis=1) + 0.01 * np.abs((current + sub) - target).sum(axis=1)
        pick = int(remaining[int(np.argmin(penalty))])
        val_mask[pick] = True
        current += y_aug[pick]

    if int(val_mask.sum()) != n_val:
        raise RuntimeError(f"Failed to build exact val split size. Expected {n_val}, got {int(val_mask.sum())}.")
    return val_mask


def enforce_hard_constraints(
    y_criteria: np.ndarray,
    score_bins: np.ndarray,
    val_mask: np.ndarray,
    locked_train: np.ndarray,
    n_val: int,
) -> tuple[np.ndarray, dict]:
    """
    Repair pass:
    - singleton positives are kept in train
    - for criteria with >=3 positives, try to ensure val has >=1 positive
      via same-score-bin swaps.
    """
    val_mask = val_mask.copy()
    n, n_labels = y_criteria.shape
    pos_totals = y_criteria.sum(axis=0).astype(int)
    target_val = pos_totals.astype(np.float64) * (n_val / float(n))

    singleton_criteria = np.where(pos_totals == 1)[0].tolist()
    need_val_positive = np.where(pos_totals >= 3)[0].tolist()
    must_keep_nonzero = np.asarray(need_val_positive, dtype=np.int32)

    swaps_done = 0
    unresolved: list[int] = []

    # Ensure locked samples are never in val.
    val_mask[locked_train] = False
    while int(val_mask.sum()) < n_val:
        candidates = np.where((~val_mask) & (~locked_train))[0]
        if candidates.size == 0:
            raise RuntimeError("Could not refill val after enforcing locked_train.")
        val_mask[int(candidates[0])] = True
    while int(val_mask.sum()) > n_val:
        candidates = np.where(val_mask & (~locked_train))[0]
        if candidates.size == 0:
            raise RuntimeError("Could not shrink val after enforcing locked_train.")
        val_mask[int(candidates[0])] = False

    val_counts = y_criteria[val_mask].sum(axis=0).astype(int)

    for c in need_val_positive:
        if val_counts[c] > 0:
            continue

        train_pos = np.where((~val_mask) & (y_criteria[:, c] == 1) & (~locked_train))[0]
        if train_pos.size == 0:
            unresolved.append(c)
            continue

        best_pair: tuple[int, int] | None = None
        best_score = float("inf")

        for i in train_pos:
            b = score_bins[i]
            val_in_bin = np.where(val_mask & (score_bins == b) & (y_criteria[:, c] == 0))[0]
            if val_in_bin.size == 0:
                continue

            for j in val_in_bin:
                if locked_train[j]:
                    continue

                # Do not remove the sole val positive of any >=3-positive criterion.
                j_labels = np.where(y_criteria[j] > 0)[0]
                if j_labels.size > 0 and must_keep_nonzero.size > 0:
                    protected = j_labels[np.isin(j_labels, must_keep_nonzero)]
                    if protected.size > 0 and np.any(val_counts[protected] <= 1):
                        continue

                trial_counts = val_counts + y_criteria[i] - y_criteria[j]
                criterion_error = float(np.abs(trial_counts - target_val).sum())

                if criterion_error < best_score:
                    best_score = criterion_error
                    best_pair = (int(i), int(j))

        if best_pair is None:
            unresolved.append(c)
            continue

        i, j = best_pair
        val_mask[i] = True
        val_mask[j] = False
        val_counts = val_counts + y_criteria[i] - y_criteria[j]
        swaps_done += 1

    if int(val_mask.sum()) != n_val:
        raise RuntimeError("Constraint repair changed val size unexpectedly.")

    info = {
        "singleton_criteria_indices": singleton_criteria,
        "criteria_ge3_indices": need_val_positive,
        "constraint_swaps_done": int(swaps_done),
        "unresolved_ge3_missing_val_indices": unresolved,
    }
    return val_mask, info


def ensure_dirs(base: Path) -> tuple[Path, Path]:
    train_dir = base / "train"
    val_dir = base / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir


def save_id_files(train_ids: list[int], val_ids: list[int], out_dir: Path) -> None:
    with open(out_dir / TRAIN_IDS_FILENAME, "w", encoding="utf-8") as f:
        for x in train_ids:
            f.write(f"{int(x)}\n")
    with open(out_dir / VAL_IDS_FILENAME, "w", encoding="utf-8") as f:
        for x in val_ids:
            f.write(f"{int(x)}\n")


def load_id_file(path: Path) -> list[int]:
    vals: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals.append(int(s))
    return vals


def build_or_load_split(df: pd.DataFrame, n_train: int, n_val: int, seed: int, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    train_ids_path = out_dir / TRAIN_IDS_FILENAME
    val_ids_path = out_dir / VAL_IDS_FILENAME

    if FREEZE_IF_MANIFEST_EXISTS and train_ids_path.exists() and val_ids_path.exists():
        train_ids = set(load_id_file(train_ids_path))
        val_ids = set(load_id_file(val_ids_path))
        if train_ids & val_ids:
            raise ValueError("Existing train/val id manifests overlap.")

        df = df.copy()
        df["split"] = "drop"
        df.loc[df["image_id"].isin(train_ids), "split"] = "train"
        df.loc[df["image_id"].isin(val_ids), "split"] = "val"
        df = df[df["split"].isin(["train", "val"])].copy()

        if int((df["split"] == "train").sum()) != len(train_ids):
            raise ValueError("Some ids in train_ids.txt are not present in current labeled+available data.")
        if int((df["split"] == "val").sum()) != len(val_ids):
            raise ValueError("Some ids in val_ids.txt are not present in current labeled+available data.")
        if len(train_ids) != n_train or len(val_ids) != n_val:
            raise ValueError(
                f"Existing manifests do not match required sizes train/val={n_train}/{n_val}. "
                f"Found {len(train_ids)}/{len(val_ids)}."
            )

        return df.reset_index(drop=True), {"reused_existing_manifest": True}

    df = df.copy().reset_index(drop=True)
    y_criteria = df[[f"c{i}" for i in range(NUM_CRITERIA_ROWS)]].to_numpy(dtype=np.int32)
    score_bins = df["score_bin"].to_numpy(dtype=np.int32)

    # Practical trick: append score-bin one-hot to criteria labels.
    n_bins = int(score_bins.max()) + 1
    y_bins = np.eye(n_bins, dtype=np.int32)[score_bins]
    y_aug = np.concatenate([y_criteria, y_bins], axis=1)

    pos_totals = y_criteria.sum(axis=0)
    singleton_criteria = np.where(pos_totals == 1)[0]
    locked_train = np.zeros(len(df), dtype=bool)
    for c in singleton_criteria:
        idx = np.where(y_criteria[:, c] == 1)[0]
        if idx.size == 1:
            locked_train[idx[0]] = True

    val_mask = iterative_multilabel_val_split(
        y_aug=y_aug,
        n_val=n_val,
        locked_train=locked_train,
        seed=seed,
    )

    val_mask, constraint_info = enforce_hard_constraints(
        y_criteria=y_criteria,
        score_bins=score_bins,
        val_mask=val_mask,
        locked_train=locked_train,
        n_val=n_val,
    )

    df["split"] = np.where(val_mask, "val", "train")
    if int((df["split"] == "train").sum()) != n_train or int((df["split"] == "val").sum()) != n_val:
        raise RuntimeError("Split sizes do not match target after stratification/repair.")

    train_ids = sorted(df.loc[df["split"] == "train", "image_id"].astype(int).tolist())
    val_ids = sorted(df.loc[df["split"] == "val", "image_id"].astype(int).tolist())
    save_id_files(train_ids, val_ids, out_dir)

    return df, {"reused_existing_manifest": False, **constraint_info}


def copy_split(df: pd.DataFrame, source_index: dict[int, list[Path]], out_dir: Path) -> dict:
    train_dir, val_dir = ensure_dirs(out_dir)

    missing_images: list[int] = []
    duplicate_id_choices: dict[int, int] = {}
    copied = 0
    skipped_exists = 0

    manifest_rows = []
    for _, r in df.sort_values(["split", "score_bin", "total_score", "image_id"]).iterrows():
        img_id = int(r["image_id"])
        src_candidates = source_index.get(img_id, [])
        if not src_candidates:
            missing_images.append(img_id)
            continue

        if len(src_candidates) > 1:
            duplicate_id_choices[img_id] = len(src_candidates)
        src = src_candidates[0]

        split = str(r["split"])
        dst_root = val_dir if split == "val" else train_dir
        dst = dst_root / src.name

        if dst.exists() and not OVERWRITE_EXISTING:
            skipped_exists += 1
        else:
            shutil.copy2(src, dst)
            copied += 1

        manifest_rows.append(
            {
                "image_id": img_id,
                "image_col": str(r["image_col"]),
                "total_score": float(r["total_score"]),
                "score_bin": int(r["score_bin"]),
                "split": split,
                "src_path": str(src),
                "dst_path": str(dst),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_dir / MANIFEST_FILENAME, index=False)

    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())

    y_criteria = df[[f"c{i}" for i in range(NUM_CRITERIA_ROWS)]].to_numpy(dtype=np.int32)
    val_mask = (df["split"] == "val").to_numpy()
    crit_pos_total = y_criteria.sum(axis=0).astype(int)
    crit_pos_val = y_criteria[val_mask].sum(axis=0).astype(int) if n_val > 0 else np.zeros(NUM_CRITERIA_ROWS, dtype=int)
    ge3_missing = [int(i) for i in np.where((crit_pos_total >= 3) & (crit_pos_val == 0))[0].tolist()]

    bin_counts_total = df["score_bin"].value_counts().sort_index().to_dict()
    bin_counts_val = df.loc[df["split"] == "val", "score_bin"].value_counts().sort_index().to_dict()
    bin_counts_train = df.loc[df["split"] == "train", "score_bin"].value_counts().sort_index().to_dict()

    summary = {
        "labels_xlsx": str(LABELS_XLSX),
        "source_images_dir": str(SOURCE_IMAGES_DIR),
        "output_dataset_dir": str(OUTPUT_DATASET_DIR),
        "num_criteria_rows": int(NUM_CRITERIA_ROWS),
        "num_score_bins": int(NUM_SCORE_BINS),
        "target_train": int(TARGET_TRAIN),
        "target_val": int(TARGET_VAL),
        "strict_target_counts": bool(STRICT_TARGET_COUNTS),
        "num_labeled_available_images": int(len(df)),
        "planned_train": int(n_train),
        "planned_val": int(n_val),
        "copied_files": int(copied),
        "skipped_existing_files": int(skipped_exists),
        "missing_images_count": int(len(missing_images)),
        "missing_images": sorted(set(missing_images)),
        "duplicate_id_choices_count": int(len(duplicate_id_choices)),
        "duplicate_id_choices": {str(k): int(v) for k, v in sorted(duplicate_id_choices.items())},
        "score_bin_counts_total": {str(k): int(v) for k, v in bin_counts_total.items()},
        "score_bin_counts_train": {str(k): int(v) for k, v in bin_counts_train.items()},
        "score_bin_counts_val": {str(k): int(v) for k, v in bin_counts_val.items()},
        "criteria_ge3_missing_val_indices": ge3_missing,
        "manifest_path": str(out_dir / MANIFEST_FILENAME),
        "train_ids_path": str(out_dir / TRAIN_IDS_FILENAME),
        "val_ids_path": str(out_dir / VAL_IDS_FILENAME),
    }

    with open(out_dir / COPY_STATS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    if not LABELS_XLSX.exists():
        raise FileNotFoundError(f"Label file not found: {LABELS_XLSX}")
    if not SOURCE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Source image folder not found: {SOURCE_IMAGES_DIR}")

    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    df = load_label_rows(LABELS_XLSX, NUM_CRITERIA_ROWS)
    source_index = index_source_images(SOURCE_IMAGES_DIR)

    # Split only over labeled images that actually exist in source dir.
    available_ids = set(source_index.keys())
    df = df[df["image_id"].isin(available_ids)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No overlap between labeled image ids and source images.")

    n_train, n_val = determine_split_sizes(len(df))
    df["score_bin"] = assign_score_bins(df["total_score"].to_numpy(dtype=np.float64), NUM_SCORE_BINS)

    df_split, split_info = build_or_load_split(
        df=df,
        n_train=n_train,
        n_val=n_val,
        seed=SPLIT_SEED,
        out_dir=OUTPUT_DATASET_DIR,
    )

    summary = copy_split(df_split, source_index, OUTPUT_DATASET_DIR)
    summary.update(split_info)
    with open(OUTPUT_DATASET_DIR / COPY_STATS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Split finished.")
    print(f"Output: {OUTPUT_DATASET_DIR}")
    print(f"Train/Val planned: {summary['planned_train']}/{summary['planned_val']}")
    print(f"Copied files: {summary['copied_files']} | Skipped existing: {summary['skipped_existing_files']}")
    print(f"Missing images: {summary['missing_images_count']}")
    print(f"Reused manifest: {summary.get('reused_existing_manifest', False)}")
    print(f"Summary: {OUTPUT_DATASET_DIR / COPY_STATS_FILENAME}")
    print(f"Manifest CSV: {OUTPUT_DATASET_DIR / MANIFEST_FILENAME}")
    print(f"Train IDs: {OUTPUT_DATASET_DIR / TRAIN_IDS_FILENAME}")
    print(f"Val IDs: {OUTPUT_DATASET_DIR / VAL_IDS_FILENAME}")


if __name__ == "__main__":
    main()
