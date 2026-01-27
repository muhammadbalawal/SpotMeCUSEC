"""
FastAPI server for Face Finder application
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
import logging

from .face_matcher import get_matcher, DEFAULT_SIMILARITY_THRESHOLD
from .utils import PHOTOS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Face Finder API",
    description="Find photos where you appear using face recognition",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount photos directory for serving images
if PHOTOS_DIR.exists():
    app.mount("/photos", StaticFiles(directory=str(PHOTOS_DIR)), name="photos")


@app.on_event("startup")
async def startup_event():
    """Initialize the face matcher on startup"""
    logger.info("Starting Face Finder API...")
    try:
        get_matcher()
        logger.info("Face matcher initialized successfully")
    except FileNotFoundError as e:
        logger.error(f"Failed to initialize: {e}")
        logger.error("Please run preprocessing.py first to generate embeddings")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Face Finder API is running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        matcher = get_matcher()
        return {
            "status": "healthy",
            "embeddings_loaded": len(matcher.embeddings),
            "unique_photos": len(set(matcher.filenames))
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/find-me")
async def find_me(
    file: UploadFile = File(..., description="Upload a photo of yourself"),
    threshold: float = Query(
        default=DEFAULT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1)"
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of results to return"
    )
):
    """
    Upload a selfie and find all photos where you appear

    - **file**: Image file (JPEG, PNG, etc.)
    - **threshold**: Minimum similarity score (default: 0.4)
    - **limit**: Maximum results to return (default: 50)
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image"
        )

    try:
        # Read file contents
        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # Get matcher and find matches
        matcher = get_matcher()
        result = matcher.match_from_bytes(
            contents,
            threshold=threshold,
            top_k=limit
        )

        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to process image")
            )

        # Add photo URLs to matches
        for match in result["matches"]:
            match["url"] = f"/photos/{match['filename']}"

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing image"
        )


@app.get("/photos-count")
async def photos_count():
    """Get the total number of processed photos"""
    try:
        matcher = get_matcher()
        unique_photos = len(set(matcher.filenames))
        total_faces = len(matcher.embeddings)
        return {
            "total_photos": unique_photos,
            "total_faces": total_faces
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
