"""
Utility functions for the Face Finder application
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from PIL import Image
from io import BytesIO
import pillow_heif

# Register HEIC/HEIF support
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
PHOTOS_DIR = DATA_DIR / "photos"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npz"


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed face embeddings from disk

    Returns:
        Tuple of (embeddings array, filenames array)

    Raises:
        FileNotFoundError: If embeddings file doesn't exist
    """
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}. "
            "Please run preprocessing.py first."
        )

    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    embeddings = data["embeddings"]
    filenames = data["filenames"]

    logger.info(f"Loaded {len(embeddings)} face embeddings from {len(np.unique(filenames))} photos")

    return embeddings, filenames


def read_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Convert image bytes to OpenCV image array
    Supports JPEG, PNG, WebP, HEIC, HEIF formats

    Args:
        image_bytes: Raw image bytes

    Returns:
        OpenCV image array (BGR) or None if failed
    """
    try:
        # Try OpenCV first (faster for common formats)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            return img

        # Fallback to Pillow for HEIC and other formats
        logger.info("OpenCV failed, trying Pillow (possibly HEIC format)")
        pil_img = Image.open(BytesIO(image_bytes))

        # Convert to RGB if needed
        if pil_img.mode in ('RGBA', 'P', 'LA'):
            pil_img = pil_img.convert('RGB')
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Convert PIL to OpenCV (RGB to BGR)
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def get_photo_url(filename: str, base_url: str = "/photos") -> str:
    """
    Get URL path for a photo

    Args:
        filename: Photo filename
        base_url: Base URL path for photos

    Returns:
        Full URL path to the photo
    """
    return f"{base_url}/{filename}"


def validate_image(img: np.ndarray) -> bool:
    """
    Validate that an image is suitable for face detection

    Args:
        img: OpenCV image array

    Returns:
        True if image is valid, False otherwise
    """
    if img is None:
        return False

    if len(img.shape) != 3:
        return False

    height, width = img.shape[:2]

    # Minimum size for face detection
    if height < 50 or width < 50:
        return False

    # Maximum size (to prevent memory issues)
    if height > 10000 or width > 10000:
        return False

    return True


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (-1 to 1)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def batch_cosine_similarity(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and all embeddings

    Args:
        query: Query embedding (512-dim)
        embeddings: Array of embeddings (N x 512)

    Returns:
        Array of similarity scores
    """
    # Normalize query
    query_norm = query / (np.linalg.norm(query) + 1e-8)

    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Compute similarities
    similarities = np.dot(embeddings_norm, query_norm)

    return similarities
