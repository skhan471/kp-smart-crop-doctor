"""Central configuration module for KP Smart Crop Doctor.

Defines project directory paths, reproducibility seeds, model parameters,
and RAG settings using pathlib and environment variables.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DATA_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
KB_DOCUMENTS_DIR = KNOWLEDGE_BASE_DIR / "documents"
KB_METADATA_DIR = KNOWLEDGE_BASE_DIR / "metadata"
KB_VECTOR_STORE_DIR = KNOWLEDGE_BASE_DIR / "vector_store"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Reproducibility Seed
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# Computer Vision Configuration
DEFAULT_IMAGE_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 15))
DEFAULT_BACKBONE = os.getenv("VISION_MODEL_BACKBONE", "mobilenet_v3_large")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 11))
TOP_K_PREDICTIONS = int(os.getenv("TOP_K", 2))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))

# Image Normalization (ImageNet standards for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Explainability Configuration (Grad-CAM)
GRADCAM_TARGET_LAYER = os.getenv("GRADCAM_TARGET_LAYER", "features.16")
HEATMAP_COLORMAP = "jet"
HEATMAP_ALPHA = 0.45

# RAG & Knowledge Base Configuration
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
RAG_TOP_K_CHUNKS = int(os.getenv("RAG_TOP_K_CHUNKS", 3))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")

# Supported Languages
SUPPORTED_LANGUAGES = ["en", "ur"]
DEFAULT_LANGUAGE = "en"
