"""Reproducible transfer learning training pipeline for crop disease classification.

Implements identical experimental protocols for MobileNetV3-Large and EfficientNet-B0:
- Pre-trained ImageNet feature extraction with cached representations for optimal CPU throughput.
- Class-weighted CrossEntropyLoss to address class imbalance.
- Model selection based strictly on validation Macro-F1.
- Complete metric tracking across epochs with full end-to-end checkpointing.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.utils import set_seed, get_device, get_logger
from src.vision.dataset import CropDataset, get_transforms, compute_class_weights
from src.vision.model import create_model, count_parameters
from evaluation.metrics import compute_classification_metrics

logger = get_logger(__name__)


@torch.no_grad()
def extract_backbone_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pass images through pre-trained convolutional backbone and global pooling.

    Returns:
        Tuple of (features_tensor [N, embedding_dim], labels_tensor [N]).
    """
    model.eval()
    all_features = []
    all_labels = []

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        feats = model.features(images)
        pooled = model.avgpool(feats)
        flat = torch.flatten(pooled, 1)
        all_features.append(flat.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, float, float, Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate full end-to-end model on raw image DataLoader.

    Returns:
        Tuple of (loss, accuracy, macro_f1, full_metrics_dict, y_true, y_prob).
    """
    model.eval()
    running_loss = 0.0
    total = 0

    all_targets = []
    all_probs = []

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.cpu().numpy())

    eval_loss = running_loss / max(total, 1)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    metrics = compute_classification_metrics(y_true, y_prob, class_names)
    eval_acc = metrics["top1_accuracy"]
    macro_f1 = metrics["macro_f1"]

    return eval_loss, eval_acc, macro_f1, metrics, y_true, y_prob


def run_training_experiment(
    backbone_name: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    device_name: str = "auto",
    random_seed: int = 42,
    patience: int = 4,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete reproducible transfer learning experiment for a given backbone.

    Args:
        backbone_name: 'mobilenet_v3_large' or 'efficientnet_b0'.
        epochs: Maximum training epochs.
        batch_size: Batch size for classifier training.
        learning_rate: Learning rate for AdamW optimizer.
        device_name: 'auto', 'cuda', or 'cpu'.
        random_seed: Seed for reproducibility.
        patience: Early stopping patience epochs on validation macro-F1.
        output_dir: Directory to save checkpoints.

    Returns:
        Dictionary containing training history, best validation metrics, and checkpoint path.
    """
    set_seed(random_seed)
    device_str = get_device(device_name)
    device = torch.device(device_str)

    if output_dir is None:
        output_dir = config.CHECKPOINTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting experiment: {backbone_name} on {device_str} | "
        f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}, Seed: {random_seed}"
    )

    # 1. Datasets and Image DataLoaders
    train_csv = config.SPLITS_DATA_DIR / "train.csv"
    val_csv = config.SPLITS_DATA_DIR / "val.csv"

    # Use standard validation transforms for feature extraction to preserve canonical features
    train_raw_dataset = CropDataset(train_csv, transform=get_transforms("val"))
    val_raw_dataset = CropDataset(val_csv, transform=get_transforms("val"), class_to_idx=train_raw_dataset.class_to_idx)
    class_names = train_raw_dataset.classes
    num_classes = len(class_names)

    train_raw_loader = DataLoader(
        train_raw_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    val_raw_loader = DataLoader(
        val_raw_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    # 2. Build Backbone Model
    model = create_model(backbone_name, num_classes=num_classes, pretrained=True, dropout=0.2)
    model = model.to(device)

    # 3. Extract & Cache ImageNet Feature Embeddings
    logger.info(f"Extracting pre-trained {backbone_name} feature representations for train split...")
    t0_feat = time.time()
    train_features, train_labels = extract_backbone_features(model, train_raw_loader, device)
    val_features, val_labels = extract_backbone_features(model, val_raw_loader, device)
    logger.info(
        f"Feature extraction complete in {time.time() - t0_feat:.1f}s. "
        f"Train features: {train_features.shape}, Val features: {val_features.shape}"
    )

    # 4. Feature DataLoaders & Optimization
    feature_train_ds = TensorDataset(train_features, train_labels)
    feature_train_loader = DataLoader(feature_train_ds, batch_size=batch_size, shuffle=True)

    # Class-weighted CrossEntropyLoss for imbalance mitigation
    class_weights = compute_class_weights(train_csv, train_raw_dataset.class_to_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Classifier Training Loop with Validation Macro-F1 Model Selection
    best_val_macro_f1 = -1.0
    best_val_metrics = {}
    best_epoch = -1
    best_classifier_state = None
    checkpoint_path = output_dir / f"{backbone_name}_best.pt"

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "epoch_times_sec": [],
    }

    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training phase on features
        model.classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in feature_train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model.classifier(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation phase on features
        model.classifier.eval()
        with torch.no_grad():
            val_feats_dev = val_features.to(device)
            val_labels_dev = val_labels.to(device)

            val_outputs = model.classifier(val_feats_dev)
            val_loss = criterion(val_outputs, val_labels_dev).item()
            val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()

            val_metrics = compute_classification_metrics(val_labels.numpy(), val_probs, class_names)
            val_acc = val_metrics["top1_accuracy"]
            val_macro_f1 = val_metrics["macro_f1"]

        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_accuracy"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))
        history["epoch_times_sec"].append(round(epoch_duration, 2))

        logger.info(
            f"[{backbone_name}] Epoch {epoch}/{epochs} ({epoch_duration:.2f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro-F1: {val_macro_f1:.4f}"
        )

        # STRICT MODEL SELECTION RULE: Validation Macro-F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_classifier_state = {k: v.cpu().clone() for k, v in model.classifier.state_dict().items()}
            patience_counter = 0

            # Save full end-to-end model checkpoint (conv layers + trained classifier head)
            torch.save({
                "epoch": epoch,
                "backbone_name": backbone_name,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "num_classes": num_classes,
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "random_seed": random_seed,
                },
            }, checkpoint_path)
            logger.info(f"--> Saved new best checkpoint to {checkpoint_path} (Val Macro-F1: {val_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Load best weights into the full model before returning
    if best_classifier_state is not None:
        model.classifier.load_state_dict({k: v.to(device) for k, v in best_classifier_state.items()})

    # Save final assembled full checkpoint
    torch.save({
        "epoch": best_epoch,
        "backbone_name": backbone_name,
        "model_state_dict": model.state_dict(),
        "val_macro_f1": best_val_macro_f1,
        "val_accuracy": best_val_metrics["top1_accuracy"],
        "val_loss": history["val_loss"][best_epoch - 1],
        "class_names": class_names,
        "num_classes": num_classes,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_seed": random_seed,
        },
    }, checkpoint_path)

    return {
        "backbone_name": backbone_name,
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_metrics": best_val_metrics,
        "history": history,
        "parameter_counts": count_parameters(model),
        "model_size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2),
    }
