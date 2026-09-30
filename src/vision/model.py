"""Model architectures for transfer learning vision classification.

Supports MobileNetV3-Large (primary candidate) and EfficientNet-B0 (benchmark).
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights,
)

from src.utils import get_logger

logger = get_logger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable, non-trainable, and total parameters of a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "total_parameters": total,
    }


def get_model_size_mb(file_path: Path) -> float:
    """Return model checkpoint file size in megabytes (MB)."""
    p = Path(file_path)
    if p.exists():
        return round(p.stat().st_size / (1024 * 1024), 2)
    return 0.0


def create_model(
    backbone_name: str = "mobilenet_v3_large",
    num_classes: int = 11,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Instantiate and configure transfer learning model with customized classification head.

    Args:
        backbone_name: 'mobilenet_v3_large' or 'efficientnet_b0'.
        num_classes: Number of target disease/healthy classes (default: 11).
        pretrained: Whether to load ImageNet pre-trained weights (default: True).
        dropout: Dropout probability in classification head (default: 0.2).

    Returns:
        Configured PyTorch nn.Module.
    """
    normalized_name = backbone_name.lower().replace("-", "_")

    if normalized_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)

        # In MobileNetV3-Large, classifier is Sequential:
        # [0] Linear(960, 1024), [1] Hardswish(), [2] Dropout(p=0.2), [3] Linear(1024, 1000)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout)
        model.classifier[3] = nn.Linear(in_features, num_classes)
        model.backbone_name = "mobilenet_v3_large"

    elif normalized_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # In EfficientNet-B0, classifier is Sequential:
        # [0] Dropout(p=0.2), [1] Linear(1280, 1000)
        in_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout)
        model.classifier[1] = nn.Linear(in_features, num_classes)
        model.backbone_name = "efficientnet_b0"

    else:
        raise ValueError(
            f"Unsupported backbone: '{backbone_name}'. "
            f"Supported options: ['mobilenet_v3_large', 'efficientnet_b0']."
        )

    logger.info(
        f"Initialized {model.backbone_name} with {num_classes} classes. "
        f"Parameters: {count_parameters(model)['total_parameters']:,} total."
    )
    return model
