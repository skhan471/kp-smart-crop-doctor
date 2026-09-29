"""Unit tests for dataset profiling, hashing, and group-stratified partitioning."""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.profiling import compute_file_hash, compute_dhash
from src.data.split import extract_leaf_id, create_group_stratified_splits


def test_extract_leaf_id():
    """Verify physical leaf ID extraction from standard PlantVillage filename format."""
    filename = "001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG"
    leaf_id = extract_leaf_id(filename)
    assert leaf_id == "rs_early.b 8178"

    filename2 = "123e4567-e89b-12d3-a456-426614174000___Corn_Rust_042.jpg"
    assert extract_leaf_id(filename2) == "corn_rust_042"

    simple_filename = "sample_leaf.png"
    assert extract_leaf_id(simple_filename) == "sample_leaf"


def test_dhash_consistency(tmp_path):
    """Verify that perceptual dHash produces deterministic output for identical images."""
    img_array = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    img1 = Image.fromarray(img_array)
    img2 = Image.fromarray(img_array)

    hash1 = compute_dhash(img1)
    hash2 = compute_dhash(img2)

    assert isinstance(hash1, str)
    assert len(hash1) == 16
    assert hash1 == hash2


def test_group_stratified_splits_zero_leakage():
    """Verify that multiple images of the same physical leaf NEVER cross split boundaries."""
    # Create 30 synthetic leaf groups across 2 classes
    records = []
    # Class A: 15 leaves, each photographed 3 times (45 images)
    for leaf_idx in range(15):
        leaf_id = f"leaf_a_{leaf_idx}"
        for photo_idx in range(3):
            records.append({
                "file_path": f"/data/class_a/img_{leaf_idx}_{photo_idx}.jpg",
                "filename": f"uuid_{leaf_idx}_{photo_idx}___{leaf_id}.jpg",
                "class_name": "Crop___Disease_A",
                "leaf_id": leaf_id,
            })

    # Class B: 15 leaves, each photographed 2 times (30 images)
    for leaf_idx in range(15):
        leaf_id = f"leaf_b_{leaf_idx}"
        for photo_idx in range(2):
            records.append({
                "file_path": f"/data/class_b/img_{leaf_idx}_{photo_idx}.jpg",
                "filename": f"uuid_{leaf_idx}_{photo_idx}___{leaf_id}.jpg",
                "class_name": "Crop___Disease_B",
                "leaf_id": leaf_id,
            })

    train_rec, val_rec, test_rec = create_group_stratified_splits(
        records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    total_images = len(records)
    assert len(train_rec) + len(val_rec) + len(test_rec) == total_images

    train_leaves = {r["leaf_id"] for r in train_rec}
    val_leaves = {r["leaf_id"] for r in val_rec}
    test_leaves = {r["leaf_id"] for r in test_rec}

    # Strict assertion: zero intersection among leaf IDs
    assert len(train_leaves.intersection(val_leaves)) == 0, "Leakage between train and val!"
    assert len(train_leaves.intersection(test_leaves)) == 0, "Leakage between train and test!"
    assert len(val_leaves.intersection(test_leaves)) == 0, "Leakage between val and test!"

    # Verify both classes are present in every split
    assert len({r["class_name"] for r in train_rec}) == 2
    assert len({r["class_name"] for r in val_rec}) == 2
    assert len({r["class_name"] for r in test_rec}) == 2
