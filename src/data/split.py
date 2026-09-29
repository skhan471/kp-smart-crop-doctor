"""Leak-free stratified dataset partitioning module for KP Smart Crop Doctor.

Implements Group-Stratified splitting to prevent data leakage between train/val/test splits:
- Images originating from the same physical leaf or plant are strictly isolated within a single split.
- Class distributions are maintained proportionally across splits (70% train / 15% val / 15% test).
- Cryptographic SHA256 checksums are logged for every file to guarantee reproducibility.
"""

from pathlib import Path
import json
import csv
import random
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from src import config
from src.utils import set_seed
from src.data.profiling import compute_file_hash

logger = logging.getLogger(__name__)


def extract_leaf_id(filename: str) -> str:
    """Extract physical leaf identifier from image filename to prevent same-leaf leakage.

    In the PlantVillage benchmark, filenames follow the convention:
    `<uuid>___<leaf_identifier>.JPG`
    e.g. `001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG`
    The leaf identifier `RS_Early.B 8178` connects all multi-angle captures of that leaf.

    Args:
        filename: Name of the image file.

    Returns:
        String representing the unique physical leaf ID.
    """
    stem = Path(filename).stem
    if "___" in stem:
        parts = stem.split("___", 1)
        return parts[1].strip().lower()
    return stem.lower()


def create_group_stratified_splits(
    image_records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition image records into train, validation, and test sets with group-leakage prevention.

    Args:
        image_records: List of image metadata dictionaries (must have 'class_name' and 'filename').
        train_ratio: Target proportion for training set (default: 0.70).
        val_ratio: Target proportion for validation set (default: 0.15).
        test_ratio: Target proportion for test set (default: 0.15).
        random_seed: Deterministic random seed.

    Returns:
        Tuple of (train_records, val_records, test_records).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    set_seed(random_seed)
    rng = random.Random(random_seed)

    # Step 1: Organize images into leaf groups within each class
    # class_name -> leaf_id -> list of image_records
    class_to_leaf_groups = defaultdict(lambda: defaultdict(list))
    for record in image_records:
        class_name = record["class_name"]
        leaf_id = record.get("leaf_id") or extract_leaf_id(record["filename"])
        record["leaf_id"] = leaf_id
        class_to_leaf_groups[class_name][leaf_id].append(record)

    train_records = []
    val_records = []
    test_records = []

    # Step 2: For each class, partition leaf groups into train, val, test
    for class_name, leaf_dict in class_to_leaf_groups.items():
        # Get list of unique leaf IDs for this class and shuffle deterministically
        leaf_ids = list(leaf_dict.keys())
        rng.shuffle(leaf_ids)

        total_images_in_class = sum(len(records) for records in leaf_dict.values())
        target_train = int(round(total_images_in_class * train_ratio))
        target_val = int(round(total_images_in_class * val_ratio))

        c_train_records = []
        c_val_records = []
        c_test_records = []

        for leaf_id in leaf_ids:
            records = leaf_dict[leaf_id]
            current_train_count = len(c_train_records)
            current_val_count = len(c_val_records)

            if current_train_count + len(records) <= target_train or (current_train_count < target_train and current_val_count >= target_val):
                c_train_records.extend(records)
            elif current_val_count + len(records) <= target_val or current_val_count < target_val:
                c_val_records.extend(records)
            else:
                c_test_records.extend(records)

        # Fallback safeguard: if test got 0 images due to rounding on tiny classes, balance from val
        if len(c_test_records) == 0 and len(c_val_records) > 1:
            moved_records = c_val_records.pop()
            c_test_records.append(moved_records)

        train_records.extend(c_train_records)
        val_records.extend(c_val_records)
        test_records.extend(c_test_records)

    # Step 3: Verify zero data leakage
    train_leaves = {r["leaf_id"] for r in train_records}
    val_leaves = {r["leaf_id"] for r in val_records}
    test_leaves = {r["leaf_id"] for r in test_records}

    train_files = {r["file_path"] for r in train_records}
    val_files = {r["file_path"] for r in val_records}
    test_files = {r["file_path"] for r in test_records}

    assert len(train_files.intersection(val_files)) == 0, "FATAL: File overlap between train and validation!"
    assert len(train_files.intersection(test_files)) == 0, "FATAL: File overlap between train and test!"
    assert len(val_files.intersection(test_files)) == 0, "FATAL: File overlap between validation and test!"

    # Verify no leaf identity crosses split boundary
    leak_train_val = train_leaves.intersection(val_leaves)
    leak_train_test = train_leaves.intersection(test_leaves)
    leak_val_test = val_leaves.intersection(test_leaves)

    if leak_train_val or leak_train_test or leak_val_test:
        logger.error(f"Leaf leakage detected! Train-Val: {len(leak_train_val)}, Train-Test: {len(leak_train_test)}")
        raise AssertionError("Data leakage detected: Same physical leaf images appeared in multiple splits.")

    logger.info(
        f"Partitioning successful. Train: {len(train_records)}, "
        f"Val: {len(val_records)}, Test: {len(test_records)}. Zero leakage verified."
    )
    return train_records, val_records, test_records


def save_splits_manifests(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    output_dir: Path,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Export partitioned splits to CSV manifests and JSON summary.

    Args:
        train_records: List of train image metadata.
        val_records: List of validation image metadata.
        test_records: List of test image metadata.
        output_dir: Directory where splits/ will be saved.
        random_seed: The random seed used.

    Returns:
        Summary metrics dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_path",
        "relative_path",
        "filename",
        "class_name",
        "crop",
        "condition",
        "leaf_id",
        "split",
        "sha256",
        "width",
        "height",
    ]

    all_records = []
    for split_name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        for r in records:
            r_copy = dict(r)
            r_copy["split"] = split_name
            # Relative path from project root for portability
            try:
                r_copy["relative_path"] = str(Path(r["file_path"]).relative_to(config.PROJECT_ROOT))
            except Exception:
                r_copy["relative_path"] = r["file_path"]
            all_records.append(r_copy)

    # Save full dataset manifest
    manifest_path = output_dir / "dataset_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    # Save individual split CSVs
    for split_name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        split_path = output_dir / f"{split_name}.csv"
        with open(split_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                r_copy = dict(r)
                r_copy["split"] = split_name
                try:
                    r_copy["relative_path"] = str(Path(r["file_path"]).relative_to(config.PROJECT_ROOT))
                except Exception:
                    r_copy["relative_path"] = r["file_path"]
                writer.writerow(r_copy)

    # Compute split statistics per class
    class_stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})
    for r in all_records:
        c = r["class_name"]
        s = r["split"]
        class_stats[c][s] += 1
        class_stats[c]["total"] += 1

    total_count = len(all_records)
    summary = {
        "random_seed": random_seed,
        "split_strategy": "Group-Stratified (Leaf-Group Isolated)",
        "train_target_ratio": 0.70,
        "val_target_ratio": 0.15,
        "test_target_ratio": 0.15,
        "counts": {
            "total": total_count,
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "actual_ratios": {
            "train": round(len(train_records) / max(total_count, 1), 4),
            "val": round(len(val_records) / max(total_count, 1), 4),
            "test": round(len(test_records) / max(total_count, 1), 4),
        },
        "class_breakdown": dict(class_stats),
        "manifest_path": str(manifest_path),
        "leak_free_verified": True,
    }

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Split manifests and summary saved to {output_dir}")
    return summary
