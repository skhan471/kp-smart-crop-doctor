"""Scaffolding and environment verification tests."""

import pytest
from pathlib import Path
from src import config
from src.utils import set_seed, get_device


def test_directory_structure_exists():
    """Verify core capstone project directories exist."""
    assert config.PROJECT_ROOT.exists()
    assert config.DATA_DIR.exists()
    assert config.RAW_DATA_DIR.exists()
    assert config.PROCESSED_DATA_DIR.exists()
    assert config.SPLITS_DATA_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert config.KNOWLEDGE_BASE_DIR.exists()
    assert config.KB_DOCUMENTS_DIR.exists()
    assert config.KB_METADATA_DIR.exists()
    assert config.EVALUATION_DIR.exists()


def test_reproducibility_seed():
    """Verify random seed setting behavior."""
    set_seed(42)
    import random
    import numpy as np

    val_random = random.random()
    val_numpy = np.random.rand()

    set_seed(42)
    assert random.random() == val_random
    assert np.random.rand() == val_numpy


def test_device_selection():
    """Verify device selection returns valid device type."""
    device = get_device("auto")
    assert device in ["cuda", "cpu"]

    device_cpu = get_device("cpu")
    assert device_cpu == "cpu"
