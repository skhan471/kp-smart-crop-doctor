"""PyTorch Dataset and transformation pipelines for leaf disease classification."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def get_transforms(split: str = "train", img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Return reproducible torchvision transforms for train or val/test splits.

    Args:
        split: 'train', 'val', or 'test'.
        img_size: Target image dimensions (width, height), default (224, 224).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    normalize = transforms.Normalize(
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD,
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])


def load_or_create_label_encoder(manifest_csv: Path, output_json: Optional[Path] = None) -> Dict[str, int]:
    """Build a deterministic alphabetical class-to-index mapping from a split manifest.

    Args:
        manifest_csv: Path to a CSV split file with 'class_name' column.
        output_json: Path to save the label mapping (default: models/label_encoder.json).

    Returns:
        Dictionary mapping class_name -> int class index.
    """
    if output_json is None:
        output_json = config.MODELS_DIR / "label_encoder.json"

    if output_json.exists():
        with open(output_json, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            return mapping["class_to_idx"]

    df = pd.read_csv(manifest_csv)
    unique_classes = sorted(df["class_name"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": len(class_to_idx),
            "classes": unique_classes,
        }, f, indent=2)

    logger.info(f"Created label encoder with {len(class_to_idx)} classes at {output_json}")
    return class_to_idx


class CropDataset(Dataset):
    """PyTorch Dataset loading images and class labels from a CSV manifest."""

    def __init__(
        self,
        csv_file: Path,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """Initialize dataset.

        Args:
            csv_file: Path to train.csv, val.csv, or test.csv.
            transform: Optional torchvision transform pipeline.
            class_to_idx: Mapping from class name to integer label index.
        """
        self.csv_file = Path(csv_file)
        self.df = pd.read_csv(self.csv_file)
        self.transform = transform

        if class_to_idx is None:
            self.class_to_idx = load_or_create_label_encoder(self.csv_file)
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = sorted(self.class_to_idx.keys(), key=lambda k: self.class_to_idx[k])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        file_path = Path(row["file_path"])

        # Resolve path if file was moved or on another machine
        if not file_path.exists():
            if "relative_path" in row and (config.PROJECT_ROOT / row["relative_path"]).exists():
                file_path = config.PROJECT_ROOT / row["relative_path"]
            elif (config.RAW_DATA_DIR / row["class_name"] / row["filename"]).exists():
                file_path = config.RAW_DATA_DIR / row["class_name"] / row["filename"]
            else:
                raise FileNotFoundError(f"Image not found: {file_path}")

        image = Image.open(file_path).convert("RGB")
        label = self.class_to_idx[row["class_name"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(file_path)


def compute_class_weights(csv_file: Path, class_to_idx: Dict[str, int]) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights for CrossEntropyLoss.

    Args:
        csv_file: Path to train.csv manifest.
        class_to_idx: Mapping from class name to label index.

    Returns:
        torch.FloatTensor of shape [num_classes].
    """
    df = pd.read_csv(csv_file)
    counts = df["class_name"].value_counts()

    num_classes = len(class_to_idx)
    total_samples = len(df)
    weights = np.zeros(num_classes, dtype=np.float32)

    for cls_name, idx in class_to_idx.items():
        cls_count = counts.get(cls_name, 1)
        # Standard balanced weighting: total / (num_classes * count)
        weights[idx] = total_samples / (num_classes * cls_count)

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)
