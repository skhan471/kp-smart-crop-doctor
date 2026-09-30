"""Explainable AI (XAI) Pipeline using Grad-CAM.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) [Selvaraju et al., ICCV 2017]
for visual interpretability of transfer learning convolutional neural networks.

Includes:
1. GradCAM class using PyTorch forward and backward hooks.
2. Heatmap and overlay generation with configurable alpha and colormaps.
3. Reusable inference and explanation API (predict_and_explain).
4. Model randomization sanity checks [Adebayo et al., NeurIPS 2018].
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.cm as cm

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def get_target_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """Resolve a dot-separated layer name into a PyTorch sub-module.

    Args:
        model: PyTorch nn.Module.
        layer_name: E.g., 'features.16' or 'features[16]'.

    Returns:
        The resolved nn.Module layer.
    """
    normalized_name = layer_name.replace("[", ".").replace("]", "")
    parts = normalized_name.split(".")
    curr = model
    for part in parts:
        if part.isdigit():
            curr = curr[int(part)]
        else:
            curr = getattr(curr, part)
    return curr


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.

    Computes activation maps from the target convolutional feature layer and weights
    them using global-average-pooled gradients of the target class logit score.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[nn.Module, str]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize Grad-CAM for a given model and target layer.

        Args:
            model: PyTorch neural network model.
            target_layer: Target conv layer nn.Module or string path (e.g. 'features.16').
                          If None, defaults to config.GRADCAM_TARGET_LAYER.
            device: torch.device (defaults to cpu if unspecified).
        """
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        if target_layer is None:
            target_layer = config.GRADCAM_TARGET_LAYER

        if isinstance(target_layer, str):
            self.target_layer = get_target_layer_by_name(self.model, target_layer)
            self.target_layer_name = target_layer
        else:
            self.target_layer = target_layer
            self.target_layer_name = target_layer.__class__.__name__

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target convolutional layer."""

        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor):
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]):
            # grad_output[0] contains the gradients with respect to layer activations
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Remove all registered hooks to avoid memory leaks or duplicate triggers."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float, torch.Tensor]:
        """Compute the Grad-CAM activation map for the specified class.

        Args:
            input_tensor: 4D tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses top-1 predicted class.

        Returns:
            Tuple of:
            - cam: 2D numpy array of shape (H, W) in range [0.0, 1.0].
            - target_class: Class index for which CAM was computed.
            - confidence: Softmax probability of target class.
            - logits: Raw output logits tensor of shape (1, num_classes).
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                f"Expected input_tensor of shape (1, C, H, W), got {input_tensor.shape}"
            )

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=-1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        confidence = float(probs[0, target_class].item())

        # Target class score (logit) for backpropagation
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Hooks failed to capture activations or gradients. Check target layer."
            )

        # Global Average Pooling (GAP) of gradients across spatial dimensions (H, W)
        # gradients shape: (1, K, u, v) -> weights shape: (1, K, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to focus on features with a positive influence on the target class
        cam = F.relu(cam)

        # Upsample CAM to match the spatial dimensions of the input image
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Min-max normalize to [0.0, 1.0]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence, logits.detach().cpu()


def generate_heatmap(
    cam: np.ndarray,
    colormap: str = config.HEATMAP_COLORMAP,
) -> np.ndarray:
    """Convert a 2D scalar CAM array into an RGB heatmap image.

    Args:
        cam: 2D numpy array with values in [0.0, 1.0].
        colormap: Matplotlib colormap name (default: 'jet').

    Returns:
        RGB numpy array of shape (H, W, 3) with uint8 values in [0, 255].
    """
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(colormap)
    rgba = cmap(cam)  # shape (H, W, 4) with values in [0.0, 1.0]
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def generate_overlay(
    original_image: Union[Image.Image, np.ndarray],
    cam: np.ndarray,
    alpha: float = config.HEATMAP_ALPHA,
    colormap: str = config.HEATMAP_COLORMAP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Superimpose a Grad-CAM heatmap onto the original leaf image.

    Args:
        original_image: Original leaf image (PIL Image or RGB np.ndarray).
        cam: 2D numpy array in range [0.0, 1.0].
        alpha: Blend weight for heatmap (default: 0.45). Original gets (1 - alpha).
        colormap: Matplotlib colormap name (default: 'jet').

    Returns:
        Tuple of:
        - overlay_rgb: Blended RGB numpy array of shape (H, W, 3) in [0, 255] uint8.
        - heatmap_rgb: Heatmap RGB numpy array of shape (H, W, 3) in [0, 255] uint8.
    """
    if isinstance(original_image, Image.Image):
        orig_np = np.array(original_image.convert("RGB"))
    else:
        orig_np = original_image.copy()

    # Resize CAM to match original image dimensions if necessary
    h, w = orig_np.shape[:2]
    if cam.shape != (h, w):
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = np.array(cam_pil.resize((w, h), Image.Resampling.BILINEAR)) / 255.0
    else:
        cam_resized = cam

    heatmap_rgb = generate_heatmap(cam_resized, colormap=colormap)

    # Linear alpha blending
    blended = (1.0 - alpha) * orig_np.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    overlay_rgb = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return overlay_rgb, heatmap_rgb


def load_label_encoder(label_encoder_path: Optional[Path] = None) -> List[str]:
    """Load class names list from label encoder JSON file.

    Args:
        label_encoder_path: Path to label encoder JSON. If None, checks models/label_encoder.json.

    Returns:
        List of class names ordered by index.
    """
    path = label_encoder_path or (config.MODELS_DIR / "label_encoder.json")
    if not path.exists():
        raise FileNotFoundError(f"Label encoder not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "classes" in data:
        return data["classes"]
    elif "idx_to_class" in data:
        idx_map = data["idx_to_class"]
        return [idx_map[str(i)] for i in range(len(idx_map))]
    else:
        raise KeyError("Invalid label encoder format: missing 'classes' or 'idx_to_class'.")


def preprocess_image_for_model(
    image: Union[Image.Image, np.ndarray, Path, str],
    target_size: Tuple[int, int] = config.DEFAULT_IMAGE_SIZE,
) -> Tuple[torch.Tensor, Image.Image]:
    """Preprocess image into normalized PyTorch tensor ready for model inference.

    Args:
        image: File path, PIL Image, or numpy array.
        target_size: Target (height, width) (default: (224, 224)).

    Returns:
        Tuple of (preprocessed 4D tensor (1, 3, H, W), resized PIL Image in RGB).
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    tensor = transform(pil_resized).unsqueeze(0)  # Shape (1, 3, H, W)
    return tensor, pil_resized


def predict_and_explain(
    model: nn.Module,
    image: Union[Image.Image, np.ndarray, Path, str],
    class_names: Optional[List[str]] = None,
    target_class: Optional[int] = None,
    target_layer: Optional[Union[nn.Module, str]] = None,
    device: Optional[torch.device] = None,
    alpha: float = config.HEATMAP_ALPHA,
    colormap: str = config.HEATMAP_COLORMAP,
) -> Dict[str, Any]:
    """Execute complete inference and Grad-CAM explanation on an input image.

    This is the primary reusable API for downstream modules (Streamlit UI, RAG context,
    and automated evaluation benchmarks).

    Args:
        model: Trained PyTorch vision model.
        image: Input image (PIL Image, numpy array, or file path).
        class_names: List of class labels. If None, loaded from label encoder.
        target_class: Optional explicit class index to explain. If None, top-1 predicted class is used.
        target_layer: Conv layer name or module (default: features.16).
        device: torch.device (defaults to cpu).
        alpha: Heatmap blend weight (default: 0.45).
        colormap: Heatmap colormap (default: 'jet').

    Returns:
        Dictionary containing:
        - 'predicted_class': Name of the top-1 predicted class.
        - 'confidence': Softmax confidence score of top-1 class (float in [0, 1]).
        - 'top2_predictions': List of top-2 predictions with class_name, confidence, class_idx.
        - 'all_predictions': List of all class predictions sorted by confidence.
        - 'original_image': Resized RGB PIL Image (224x224).
        - 'gradcam_heatmap': RGB numpy array uint8 (224x224, 3) of the heatmap.
        - 'gradcam_overlay': RGB numpy array uint8 (224x224, 3) of the blended image.
        - 'raw_cam': 2D float numpy array (224x224) with values in [0.0, 1.0].
        - 'target_class_idx': Class index explained by Grad-CAM.
        - 'target_class_name': Name of class explained by Grad-CAM.
    """
    dev = device or torch.device("cpu")
    if class_names is None:
        class_names = load_label_encoder()

    input_tensor, pil_img = preprocess_image_for_model(image)
    orig_np = np.array(pil_img)

    gradcam = GradCAM(model, target_layer=target_layer, device=dev)
    try:
        cam, explained_idx, target_conf, logits = gradcam.generate_cam(
            input_tensor, target_class=target_class
        )
    finally:
        gradcam.remove_hooks()

    probs = F.softmax(logits, dim=-1)[0].numpy()
    sorted_indices = np.argsort(probs)[::-1]

    top1_idx = int(sorted_indices[0])
    top1_conf = float(probs[top1_idx])
    top1_name = class_names[top1_idx]

    top2_preds = [
        {
            "class_idx": int(idx),
            "class_name": class_names[idx],
            "confidence": float(probs[idx]),
        }
        for idx in sorted_indices[:2]
    ]

    all_preds = [
        {
            "class_idx": int(idx),
            "class_name": class_names[idx],
            "confidence": float(probs[idx]),
        }
        for idx in sorted_indices
    ]

    overlay_rgb, heatmap_rgb = generate_overlay(
        orig_np, cam, alpha=alpha, colormap=colormap
    )

    explained_name = class_names[explained_idx]

    return {
        "predicted_class": top1_name,
        "confidence": top1_conf,
        "top2_predictions": top2_preds,
        "all_predictions": all_preds,
        "original_image": pil_img,
        "gradcam_heatmap": heatmap_rgb,
        "gradcam_overlay": overlay_rgb,
        "raw_cam": cam,
        "target_class_idx": explained_idx,
        "target_class_name": explained_name,
    }


def compute_heatmap_similarity(
    map_a: np.ndarray, map_b: np.ndarray
) -> Dict[str, float]:
    """Compute quantitative similarity metrics between two 2D heatmaps.

    Used for model randomization sanity check (Adebayo et al., NeurIPS 2018).

    Args:
        map_a: First 2D numpy array.
        map_b: Second 2D numpy array.

    Returns:
        Dictionary with Pearson r, Spearman rank rho, and Mean Absolute Error (MAE).
    """
    flat_a = map_a.flatten().astype(np.float64)
    flat_b = map_b.flatten().astype(np.float64)

    # Mean Absolute Error
    mae = float(np.mean(np.abs(flat_a - flat_b)))

    # Pearson correlation coefficient
    std_a = np.std(flat_a)
    std_b = np.std(flat_b)
    if std_a > 1e-8 and std_b > 1e-8:
        pearson_r = float(np.corrcoef(flat_a, flat_b)[0, 1])
    else:
        pearson_r = 0.0

    # Spearman rank correlation
    rank_a = np.argsort(np.argsort(flat_a))
    rank_b = np.argsort(np.argsort(flat_b))
    std_ra = np.std(rank_a)
    std_rb = np.std(rank_b)
    if std_ra > 1e-8 and std_rb > 1e-8:
        spearman_rho = float(np.corrcoef(rank_a, rank_b)[0, 1])
    else:
        spearman_rho = 0.0

    return {
        "pearson_correlation": round(pearson_r, 4),
        "spearman_rank_correlation": round(spearman_rho, 4),
        "mean_absolute_error": round(mae, 4),
    }


def model_randomization_sanity_check(
    trained_model: nn.Module,
    image: Union[Image.Image, np.ndarray, Path, str],
    target_layer: Optional[str] = None,
    target_class: Optional[int] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Perform Adebayo et al. (NeurIPS 2018) model randomization sanity check.

    Compares the Grad-CAM heatmap produced by the trained model against a randomized
    model with randomly re-initialized weights. If Grad-CAM is genuinely sensitive to learned
    features, the randomized heatmap should lose spatial localization to lesions and show
    near-zero or negative correlation with the trained heatmap.

    Args:
        trained_model: Trained PyTorch vision model.
        image: Leaf image.
        target_layer: Conv layer string (default: features.16).
        target_class: Target class index. If None, uses trained model's top-1 prediction.
        random_seed: Seed for random weight initialization.

    Returns:
        Dictionary with:
        - 'trained_cam': 2D float array of trained model's CAM.
        - 'randomized_cam': 2D float array of randomized model's CAM.
        - 'similarity': Similarity metrics (pearson_r, spearman_rho, mae).
        - 'passed': True if Pearson correlation is below 0.35 (demonstrating parameter sensitivity).
    """
    from src.vision.model import create_model

    target_layer_str = target_layer or config.GRADCAM_TARGET_LAYER
    input_tensor, pil_img = preprocess_image_for_model(image)

    # 1. Trained model Grad-CAM
    gradcam_trained = GradCAM(trained_model, target_layer=target_layer_str)
    try:
        trained_cam, explained_class, conf, _ = gradcam_trained.generate_cam(
            input_tensor, target_class=target_class
        )
    finally:
        gradcam_trained.remove_hooks()

    # 2. Instantiate identical randomized model (uninitialized / random weights)
    torch.manual_seed(random_seed)
    randomized_model = create_model(
        backbone_name=getattr(trained_model, "backbone_name", "mobilenet_v3_large"),
        num_classes=config.NUM_CLASSES,
        pretrained=False,
    )
    randomized_model.eval()

    gradcam_random = GradCAM(randomized_model, target_layer=target_layer_str)
    try:
        random_cam, _, _, _ = gradcam_random.generate_cam(
            input_tensor, target_class=explained_class
        )
    finally:
        gradcam_random.remove_hooks()

    similarity = compute_heatmap_similarity(trained_cam, random_cam)
    # Passed if correlation between trained and randomized maps is low (e.g. < 0.35)
    passed = similarity["pearson_correlation"] < 0.35

    logger.info(
        f"Adebayo Sanity Check: Pearson r = {similarity['pearson_correlation']:.4f}, "
        f"Spearman rho = {similarity['spearman_rank_correlation']:.4f}, Passed: {passed}"
    )

    return {
        "target_class": explained_class,
        "trained_cam": trained_cam,
        "randomized_cam": random_cam,
        "similarity": similarity,
        "passed": passed,
    }
