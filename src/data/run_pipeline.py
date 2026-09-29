"""Execution script for dataset staging, profiling, and group-stratified splitting."""

import json
from pathlib import Path
from src import config
from src.utils import get_logger, set_seed
from src.data.download import sync_raw_classes, KP_TARGET_CLASSES
from src.data.profiling import profile_dataset
from src.data.split import create_group_stratified_splits, save_splits_manifests

logger = get_logger("dataset_pipeline")


def run_pipeline(staging_dir: Path):
    """Run full profiling and partitioning pipeline."""
    set_seed(config.RANDOM_SEED)

    # 1. Sync classes to data/raw
    logger.info("Step 1: Synchronizing target classes to data/raw...")
    total_staged = sync_raw_classes(staging_dir, config.RAW_DATA_DIR)
    logger.info(f"Staged {total_staged} images into {config.RAW_DATA_DIR}")

    # 2. Profile dataset
    logger.info("Step 2: Profiling dataset and scanning for duplicates...")
    report, records = profile_dataset(config.RAW_DATA_DIR)

    # Save profiling report
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    profiling_path = config.EVALUATION_DIR / "dataset_profiling_report.json"
    with open(profiling_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Dataset profiling report saved to {profiling_path}")

    # 3. Create group-stratified splits
    logger.info("Step 3: Creating leak-free group-stratified partitions (70/15/15)...")
    train_recs, val_recs, test_recs = create_group_stratified_splits(
        records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=config.RANDOM_SEED,
    )

    # 4. Save CSV manifests and summary
    logger.info("Step 4: Saving split CSV manifests and cryptographic checksums...")
    summary = save_splits_manifests(
        train_recs, val_recs, test_recs, config.SPLITS_DATA_DIR, random_seed=config.RANDOM_SEED
    )

    logger.info("Pipeline execution completed successfully!")
    return report, summary


if __name__ == "__main__":
    staging_path = config.PROJECT_ROOT / "scratch" / "pv_sparse" / "raw" / "color"
    run_pipeline(staging_path)
