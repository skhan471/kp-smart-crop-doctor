"""Utility functions for reproducibility, device selection, and logging."""

import os
import random
import logging
from typing import Optional
import numpy as np

# Configure standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger instance for a given module name."""
    return logging.getLogger(name)


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch (if available) for reproducibility.

    Args:
        seed: Integer random seed (default: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(preference: str = "auto") -> str:
    """Determine the active computation device ('cuda' or 'cpu').

    Args:
        preference: 'auto', 'cuda', or 'cpu'.

    Returns:
        Device string ('cuda' or 'cpu').
    """
    if preference.lower() == "cpu":
        return "cpu"

    try:
        import torch

        if preference.lower() == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preference.lower() == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

    return "cpu"
