"""
Face matching logic using pre-computed embeddings
"""

import numpy as np
from insightface.app import FaceAnalysis
from typing import List, Dict, Optional
import logging

from .utils import (
    load_embeddings,
    read_image_from_bytes,
    validate_image,
    batch_cosine_similarity,
)

logger = logging.getLogger(__name__)

# Model settings (same as preprocessing)
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
DET_THRESH = 0.5

# Matching threshold
DEFAULT_SIMILARITY_THRESHOLD = 0.4


class FaceMatcher:
    """Match uploaded face against pre-computed embeddings"""

    def __init__(self):
        """Initialize the face matcher with model and embeddings"""
        logger.info("Initializing FaceMatcher...")

        # Initialize face analysis model
        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=DET_THRESH)

        # Load pre-computed embeddings
        self.embeddings, self.filenames = load_embeddings()

        logger.info(f"FaceMatcher ready with {len(self.embeddings)} face embeddings")

    def extract_face_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image

        Args:
            img: OpenCV image array (BGR)

        Returns:
            Face embedding (512-dim) or None if no face detected
        """
        if not validate_image(img):
            logger.warning("Invalid image provided")
            return None

        faces = self.app.get(img)

        if not faces:
            logger.info("No face detected in the uploaded image")
            return None

        if len(faces) > 1:
            logger.info(f"Multiple faces detected ({len(faces)}), using the largest one")
            # Use the face with the largest bounding box
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        return faces[0].embedding

    def find_matches(
        self,
        query_embedding: np.ndarray,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Find photos containing the query face

        Args:
            query_embedding: Face embedding to search for (512-dim)
            threshold: Minimum similarity score (0-1)
            top_k: Maximum number of matches to return (None for all)

        Returns:
            List of matches with filename and similarity score
        """
        # Compute similarities with all stored embeddings
        similarities = batch_cosine_similarity(query_embedding, self.embeddings)

        # Get unique photos with their best match score
        photo_scores = {}
        for i, (sim, filename) in enumerate(zip(similarities, self.filenames)):
            if sim >= threshold:
                # Keep the highest score for each photo
                if filename not in photo_scores or sim > photo_scores[filename]:
                    photo_scores[filename] = float(sim)

        # Sort by similarity score (highest first)
        sorted_matches = sorted(
            photo_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Apply top_k limit if specified
        if top_k is not None:
            sorted_matches = sorted_matches[:top_k]

        # Format results
        matches = [
            {"filename": filename, "similarity": score}
            for filename, score in sorted_matches
        ]

        logger.info(f"Found {len(matches)} matching photos (threshold: {threshold})")

        return matches

    def match_from_bytes(
        self,
        image_bytes: bytes,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Find matching photos from uploaded image bytes

        Args:
            image_bytes: Raw image bytes
            threshold: Minimum similarity score
            top_k: Maximum number of matches

        Returns:
            Dict with matches and metadata
        """
        # Decode image
        img = read_image_from_bytes(image_bytes)
        if img is None:
            return {
                "success": False,
                "error": "Failed to decode image",
                "matches": []
            }

        # Extract face embedding
        embedding = self.extract_face_embedding(img)
        if embedding is None:
            return {
                "success": False,
                "error": "No face detected in the uploaded image",
                "matches": []
            }

        # Find matches
        matches = self.find_matches(embedding, threshold=threshold, top_k=top_k)

        return {
            "success": True,
            "face_detected": True,
            "total_matches": len(matches),
            "matches": matches
        }


# Singleton instance for reuse
_matcher_instance: Optional[FaceMatcher] = None


def get_matcher() -> FaceMatcher:
    """Get or create the singleton FaceMatcher instance"""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = FaceMatcher()
    return _matcher_instance
