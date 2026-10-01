"""Curated demo sample images and loader for KP Smart Crop Doctor.

Provides deterministic test images for all 11 classes for demonstration
and automated evaluation without requiring external uploads.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

DEMO_SAMPLES: Dict[str, Dict[str, Any]] = {
    "potato_late_blight": {
        "crop": "Potato",
        "disease_class": "Potato___Late_blight",
        "label_en": "Potato: Late Blight (Destructive Water Mold)",
        "label_ur": "آلو: پچھیتی جھلساؤ (لیٹ بلائٹ)",
        "relative_path": "data/raw/Potato___Late_blight/0051e5e8-d1c4-4a84-bf3a-a426cdad6285___RS_LB 4640.JPG",
        "description_en": "Field lesion with typical water-soaked foliar margins.",
        "description_ur": "پتے کے کناروں پر نم آلود اور سیاہ دھبے",
    },
    "potato_early_blight": {
        "crop": "Potato",
        "disease_class": "Potato___Early_blight",
        "label_en": "Potato: Early Blight (Concentric Target Rings)",
        "label_ur": "آلو: اگیتی جھلساؤ (ارلی بلائٹ)",
        "relative_path": "data/raw/Potato___Early_blight/001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG",
        "description_en": "Concentric necrotic rings developing on mature potato foliage.",
        "description_ur": "پتوں پر دائرہ نما ہدف کے نشانات",
    },
    "potato_healthy": {
        "crop": "Potato",
        "disease_class": "Potato___healthy",
        "label_en": "Potato: Healthy Leaf (Control)",
        "label_ur": "آلو: صحت مند پتا (کنٹرول)",
        "relative_path": "data/raw/Potato___healthy/00fc2ee5-729f-4757-8aeb-65c3355874f2___RS_HL 1864.JPG",
        "description_en": "Vigorous, dark-green healthy potato foliage.",
        "description_ur": "صحت مند اور تروتازہ سبز پتا",
    },
    "apple_scab": {
        "crop": "Apple",
        "disease_class": "Apple___Apple_scab",
        "label_en": "Apple: Apple Scab (Velvety Olive Lesions)",
        "label_ur": "سیب: پپڑی روگ (ایپل سکيب)",
        "relative_path": "data/raw/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG",
        "description_en": "Olive-brown velvety scab lesions on apple leaf surface.",
        "description_ur": "سیب کے پتے پر زیتونی بھورے رنگ کے کھرنڈ نما دھبے",
    },
    "apple_black_rot": {
        "crop": "Apple",
        "disease_class": "Apple___Black_rot",
        "label_en": "Apple: Black Rot / Frogeye Leaf Spot",
        "label_ur": "سیب: سیاہ سڑن اور مینڈک کی آنکھ نما دھبے",
        "relative_path": "data/raw/Apple___Black_rot/0090d05d-d797-4c99-abd4-3b9cb323a5fd___JR_FrgE.S 8727.JPG",
        "description_en": "Circular frogeye spots with purple-brown margins.",
        "description_ur": "جامنی بھورے کناروں والے گول دھبے",
    },
    "apple_cedar_rust": {
        "crop": "Apple",
        "disease_class": "Apple___Cedar_apple_rust",
        "label_en": "Apple: Cedar Apple Rust (Bright Orange Spots)",
        "label_ur": "سیب: سیب اور صنوبر کا زنگ",
        "relative_path": "data/raw/Apple___Cedar_apple_rust/025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust 3655.JPG",
        "description_en": "Vibrant orange-yellow rust aecia spots on foliage.",
        "description_ur": "پتے پر چمکدار نارنجی زرد دھبے",
    },
    "apple_healthy": {
        "crop": "Apple",
        "disease_class": "Apple___healthy",
        "label_en": "Apple: Healthy Leaf (Control)",
        "label_ur": "سیب: صحت مند پتا (کنٹرول)",
        "relative_path": "data/raw/Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG",
        "description_en": "Intact, fully expanded healthy apple leaf.",
        "description_ur": "صحت مند اور بے داغ سیب کا پتا",
    },
    "maize_gray_leaf_spot": {
        "crop": "Corn_(maize)",
        "disease_class": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "label_en": "Maize: Gray Leaf Spot (Rectangular Lesions)",
        "label_ur": "مکئی: سرمئی پتا دھبہ روگ (گرے لیف سپاٹ)",
        "relative_path": "data/raw/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/00120a18-ff90-46e4-92fb-2b7a10345bd3___RS_GLSp 9357.JPG",
        "description_en": "Vein-delimited rectangular gray lesions on maize blade.",
        "description_ur": "مکئی کے پتے پر رگوں کے درمیان مستطیل دھبے",
    },
    "maize_common_rust": {
        "crop": "Corn_(maize)",
        "disease_class": "Corn_(maize)___Common_rust_",
        "label_en": "Maize: Common Rust (Cinnamon-Brown Pustules)",
        "label_ur": "مکئی: عام زنگ (کامن رسٹ)",
        "relative_path": "data/raw/Corn_(maize)___Common_rust_/RS_Rust 1563.JPG",
        "description_en": "Raised powdery cinnamon-brown pustules on both leaf sides.",
        "description_ur": "پتے کی دونوں سطحوں پر ابھرے ہوئے دارچینی رنگ کے چھالے",
    },
    "maize_northern_leaf_blight": {
        "crop": "Corn_(maize)",
        "disease_class": "Corn_(maize)___Northern_Leaf_Blight",
        "label_en": "Maize: Northern Corn Leaf Blight (Cigar Lesions)",
        "label_ur": "مکئی: شمالی پتوں کا جھلساؤ (ناردن لیف بلائٹ)",
        "relative_path": "data/raw/Corn_(maize)___Northern_Leaf_Blight/005318c8-a5fa-4420-843b-23bdda7322c2___RS_NLB 3853 copy.jpg",
        "description_en": "Long elliptical cigar-shaped lesions on maize blade.",
        "description_ur": "مکئی کے پتے پر سگار نما لمبے اور بیضوی دھبے",
    },
    "maize_healthy": {
        "crop": "Corn_(maize)",
        "disease_class": "Corn_(maize)___healthy",
        "label_en": "Maize: Healthy Leaf (Control)",
        "label_ur": "مکئی: صحت مند پتا (کنٹرول)",
        "relative_path": "data/raw/Corn_(maize)___healthy/00031d74-076e-4aef-b040-e068cd3576eb___R.S_HL 8315 copy 2.jpg",
        "description_en": "Clean, unblemished healthy maize leaf blade.",
        "description_ur": "صحت مند اور صاف مکئی کا پتا",
    },
}


def get_demo_sample_keys() -> list:
    """Return ordered list of demo sample keys."""
    return list(DEMO_SAMPLES.keys())


def load_demo_image(sample_key: str, base_dir: Optional[Path] = None) -> Optional[Image.Image]:
    """Load sample PIL Image from relative path.

    Args:
        sample_key: Key in DEMO_SAMPLES.
        base_dir: Root project directory (defaults to config.PROJECT_ROOT).

    Returns:
        RGB PIL Image instance or None if not found.
    """
    if sample_key not in DEMO_SAMPLES:
        return None

    root = base_dir or config.PROJECT_ROOT
    rel_path = DEMO_SAMPLES[sample_key]["relative_path"]
    full_path = root / rel_path

    if not full_path.exists():
        logger.warning(f"Demo image not found at: {full_path}")
        return None

    try:
        return Image.open(full_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open demo image at {full_path}: {e}")
        return None
