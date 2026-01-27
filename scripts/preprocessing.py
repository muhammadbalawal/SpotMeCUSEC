"""
Extracts face embeddings from all photos once and stores them for fast lookup.
"""

import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PHOTOS_DIR = Path("data/photos")
EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npz"

# Model settings
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
DET_THRESH = 0.5


class FaceEmbeddingExtractor:
    """Extract and store face embeddings from photos"""
    
    def __init__(self):
        """Initialize the face analysis model"""
        logger.info("Initializing InsightFace model...")
        
        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Prepare model
        # ctx_id=0 means use GPU 0, ctx_id=-1 means use CPU
        self.app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=DET_THRESH)
        
        logger.info(f"Model initialized: {MODEL_NAME}")
        logger.info(f"Detection size: {DET_SIZE}")
        logger.info(f"Using GPU: CUDA available")
    
    def extract_faces_from_image(self, image_path: Path) -> List[np.ndarray]:
        """
        Extract all faces from a single image

        Args:
            image_path: Path to image file

        Returns:
            List of face embedding arrays (512-dim each)
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))

            if img is None:
                logger.warning(f"Could not read image: {image_path}")
                return []

            # Get all faces in the image
            faces = self.app.get(img)

            # Extract embeddings only
            return [face.embedding for face in faces]
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return []
    
    def process_all_photos(self) -> dict:
        """
        Process all photos in the photos directory
        
        Returns:
            Dictionary containing all face embeddings organized by photo
        """
        if not PHOTOS_DIR.exists():
            raise FileNotFoundError(f"Photos directory not found: {PHOTOS_DIR}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [
            f for f in PHOTOS_DIR.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        total_files = len(image_files)
        logger.info(f"Found {total_files} images to process")
        
        if total_files == 0:
            raise ValueError("No images found in photos directory")
        
        # Process each image
        embeddings_data = {}
        total_faces = 0
        processed = 0
        
        for idx, image_path in enumerate(image_files, 1):
            # Extract faces
            faces = self.extract_faces_from_image(image_path)
            
            if faces:
                embeddings_data[image_path.name] = faces
                total_faces += len(faces)
            
            processed += 1
            
            # Log progress
            if processed % 50 == 0 or processed == total_files:
                logger.info(
                    f"Progress: {processed}/{total_files} images "
                    f"({(processed/total_files)*100:.1f}%) - "
                    f"Total faces: {total_faces}"
                )
        
        logger.info(f"Processing complete!")
        logger.info(f"  - Images processed: {processed}")
        logger.info(f"  - Total faces detected: {total_faces}")
        logger.info(f"  - Average faces per image: {total_faces/processed:.1f}")
        
        return embeddings_data
    
    def save_embeddings(self, embeddings_data: dict):
        """
        Save embeddings to numpy binary file

        Args:
            embeddings_data: Dictionary mapping filename -> list of embeddings
        """
        # Create embeddings directory if it doesn't exist
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        # Flatten to arrays for fast similarity search
        all_embeddings = []
        all_filenames = []

        for filename, embeddings in embeddings_data.items():
            for emb in embeddings:
                all_embeddings.append(emb)
                all_filenames.append(filename)

        # Convert to numpy arrays
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        filenames_array = np.array(all_filenames)

        # Save to compressed numpy binary
        logger.info(f"Saving embeddings to {EMBEDDINGS_FILE}...")
        np.savez_compressed(
            EMBEDDINGS_FILE,
            embeddings=embeddings_array,
            filenames=filenames_array
        )

        # Calculate file size
        file_size_mb = EMBEDDINGS_FILE.stat().st_size / (1024 * 1024)
        logger.info(f"Embeddings saved successfully!")
        logger.info(f"  - File: {EMBEDDINGS_FILE}")
        logger.info(f"  - Size: {file_size_mb:.2f} MB")
        logger.info(f"  - Shape: {embeddings_array.shape}")


def validate_environment():
    """Validate that all required directories and dependencies exist"""
    
    # Check if photos directory exists
    if not PHOTOS_DIR.exists():
        logger.error(f"Photos directory not found: {PHOTOS_DIR}")
        logger.info("Please create the directory and add your photos:")
        logger.info(f"  mkdir -p {PHOTOS_DIR}")
        return False
    
    # Check if photos directory has images
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [
        f for f in PHOTOS_DIR.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if len(image_files) == 0:
        logger.error(f"No images found in {PHOTOS_DIR}")
        logger.info("Please add your hackathon photos to this directory")
        return False
    
    logger.info(f"Found {len(image_files)} images in {PHOTOS_DIR}")
    
    # Check if InsightFace is installed
    try:
        import insightface
        logger.info(f"InsightFace version: {insightface.__version__}")
    except ImportError:
        logger.error("InsightFace not installed!")
        logger.info("Install with: pip install insightface onnxruntime-gpu")
        return False
    
    # Check OpenCV
    try:
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        logger.error("OpenCV not installed!")
        logger.info("Install with: pip install opencv-python")
        return False
    
    return True


def main():
    """Main preprocessing pipeline"""
    
    print("=" * 60)
    print("  Hackathon Photo Finder - Preprocessing")
    print("=" * 60)
    print()
    
    # Validate environment
    logger.info("Validating environment...")
    if not validate_environment():
        logger.error("Environment validation failed. Please fix the issues above.")
        return
    
    print()
    logger.info("Starting face embedding extraction...")
    print()
    
    try:
        # Initialize extractor
        extractor = FaceEmbeddingExtractor()
        
        # Process all photos
        embeddings_data = extractor.process_all_photos()
        
        # Save embeddings
        extractor.save_embeddings(embeddings_data)
        
        print()
        print("=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        print()
        print(f"Embeddings saved to: {EMBEDDINGS_FILE}")
        print()
        print("Next steps:")
        print("  1. Start the FastAPI server:")
        print("     uvicorn app.main:app --reload")
        print("  2. Open your browser and test the face finder!")
        print()
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()