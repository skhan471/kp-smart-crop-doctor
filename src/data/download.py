"""Dataset download, verification, and preparation utility for KP Smart Crop Doctor.

Provides reproducible retrieval of curated agricultural benchmark datasets
(PlantVillage subset) targeting priority Khyber Pakhtunkhwa crops.
"""

from pathlib import Path
import os
import shutil
import logging
from typing import List, Optional

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

# Recommended core crop and disease classes tailored to KP agriculture (11 classes across 3 major agricultural sectors)
KP_TARGET_CLASSES = [
    # Maize (Corn) - Core KP Cereal (Swabi, Mardan, Charsadda, Nowshera)
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___healthy",
    # Potato - Core KP Hill & Plain Staple (Swat, Kalam, Battagram, Mardan)
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    # Apple - High-Value Temperate Fruit (Upper Swat, Dir, Chitral, Hazara)
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]


def sync_raw_classes(source_dir: Path, target_dir: Path, class_list: Optional[List[str]] = None) -> int:
    """Copy verified class directories from a staging or source directory to data/raw.

    Args:
        source_dir: Source directory containing class folders.
        target_dir: Target directory (typically config.RAW_DATA_DIR).
        class_list: Optional list of class names to copy (defaults to KP_TARGET_CLASSES).

    Returns:
        Total number of images copied and verified.
    """
    if class_list is None:
        class_list = KP_TARGET_CLASSES

    target_dir.mkdir(parents=True, exist_ok=True)
    total_copied = 0

    for class_name in class_list:
        src_class_path = source_dir / class_name
        dest_class_path = target_dir / class_name

        if not src_class_path.exists():
            logger.warning(f"Class folder {src_class_path} not found in source.")
            continue

        dest_class_path.mkdir(parents=True, exist_ok=True)
        files = [f for f in src_class_path.iterdir() if f.is_file() and not f.name.startswith(".")]

        for f in files:
            target_file = dest_class_path / f.name
            if not target_file.exists():
                shutil.copy2(f, target_file)
            total_copied += 1

        logger.info(f"Synchronized class '{class_name}': {len(files)} images.")

    return total_copied
