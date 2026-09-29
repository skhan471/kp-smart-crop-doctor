"""Dataset profiling and quality assessment pipeline for KP Smart Crop Doctor.

Computes class distributions, image resolution consistency, channel integrity,
file integrity, class imbalance ratios, and identifies exact and near-duplicate images.
"""

from pathlib import Path
import hashlib
import json
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def compute_file_hash(filepath: Path, algorithm: str = "sha256", chunk_size: int = 65536) -> str:
    """Compute cryptographic hash of a file for integrity and exact duplicate detection."""
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for near-duplicate visual image detection.

    Args:
        image: PIL Image instance.
        hash_size: Grid size for hash (default: 8 gives 64-bit hash).

    Returns:
        Hexadecimal hash string representing visual frequency gradient.
    """
    # Resize to (hash_size + 1, hash_size) in grayscale
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.float32)
    # Compare adjacent columns: difference gradient
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert binary array to hex string
    decimal_val = 0
    hex_str = []
    for idx, val in enumerate(diff.flatten()):
        if val:
            decimal_val += 1 << (idx % 4)
        if idx % 4 == 3:
            hex_str.append(hex(decimal_val)[2:])
            decimal_val = 0
    return "".join(hex_str)


def profile_dataset(data_dir: Path) -> Dict[str, Any]:
    """Scan and profile a leaf disease dataset directory organized in class subfolders.

    Args:
        data_dir: Root directory containing class subdirectories.

    Returns:
        Dictionary containing comprehensive profiling statistics, duplicate counts,
        imbalance metrics, and resolution distributions.
    """
    logger.info(f"Initiating dataset profiling for directory: {data_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    class_counts = defaultdict(int)
    image_records = []
    sha256_to_paths = defaultdict(list)
    dhash_to_paths = defaultdict(list)
    resolution_counts = defaultdict(int)
    mode_counts = defaultdict(int)
    corrupted_files = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for class_dir in sorted(subdirs):
        class_name = class_dir.name
        # Parse crop and condition from standard folder format Crop___Condition
        if "___" in class_name:
            crop_name, condition = class_name.split("___", 1)
        else:
            crop_name, condition = "Unknown", class_name

        image_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix in valid_extensions]

        for img_path in image_files:
            try:
                # File integrity and metadata check
                with Image.open(img_path) as img:
                    img.verify()

                # Re-open for mode, size, and hash computation (verify invalidates file pointer)
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    width, height = img.size
                    mode = img.mode

                    sha256_hash = compute_file_hash(img_path, "sha256")
                    dhash_val = compute_dhash(img_rgb)

                    class_counts[class_name] += 1
                    resolution_counts[f"{width}x{height}"] += 1
                    mode_counts[mode] += 1

                    sha256_to_paths[sha256_hash].append(str(img_path))
                    dhash_to_paths[dhash_val].append(str(img_path))

                    image_records.append({
                        "file_path": str(img_path),
                        "filename": img_path.name,
                        "class_name": class_name,
                        "crop": crop_name,
                        "condition": condition,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "sha256": sha256_hash,
                        "dhash": dhash_val,
                        "size_bytes": img_path.stat().st_size
                    })

            except Exception as e:
                logger.warning(f"Corrupted or unreadable image skipped: {img_path} ({e})")
                corrupted_files.append({"path": str(img_path), "error": str(e)})

    # Calculate class imbalance statistics
    counts_list = list(class_counts.values())
    if counts_list:
        max_count = max(counts_list)
        min_count = min(counts_list)
        total_images = sum(counts_list)
        imbalance_ratio = round(max_count / max(min_count, 1), 2)
        mean_per_class = round(np.mean(counts_list), 1)
        std_per_class = round(np.std(counts_list), 1)
    else:
        max_count = min_count = total_images = imbalance_ratio = mean_per_class = std_per_class = 0

    # Duplicate findings
    exact_duplicates = {k: v for k, v in sha256_to_paths.items() if len(v) > 1}
    near_duplicates = {k: v for k, v in dhash_to_paths.items() if len(v) > 1 and k not in exact_duplicates}

    total_exact_duplicate_files = sum(len(v) - 1 for v in exact_duplicates.values())
    total_near_duplicate_files = sum(len(v) - 1 for v in near_duplicates.values())

    report = {
        "dataset_root": str(data_dir),
        "total_images": total_images,
        "total_classes": len(class_counts),
        "class_distribution": dict(class_counts),
        "class_imbalance": {
            "min_samples": min_count,
            "max_samples": max_count,
            "mean_samples": mean_per_class,
            "std_samples": std_per_class,
            "imbalance_ratio": imbalance_ratio,
        },
        "duplicate_analysis": {
            "exact_duplicate_clusters": len(exact_duplicates),
            "redundant_exact_duplicate_images": total_exact_duplicate_files,
            "near_duplicate_clusters": len(near_duplicates),
            "redundant_near_duplicate_images": total_near_duplicate_files,
        },
        "resolution_distribution": dict(resolution_counts),
        "color_mode_distribution": dict(mode_counts),
        "corrupted_images_detected": len(corrupted_files),
        "corrupted_files": corrupted_files,
    }

    logger.info(
        f"Profiling complete. Total images: {total_images} across {len(class_counts)} classes. "
        f"Imbalance ratio: {imbalance_ratio}:1. Exact duplicates: {total_exact_duplicate_files}."
    )
    return report, image_records
