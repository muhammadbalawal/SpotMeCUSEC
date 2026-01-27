"""
FastAPI server for Face Finder application
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, Response
from pathlib import Path
import logging
import json
import httpx
from PIL import Image
from io import BytesIO
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import pillow_heif

from .face_matcher import get_matcher, DEFAULT_SIMILARITY_THRESHOLD

# Register HEIC/HEIF support with Pillow
pillow_heif.register_heif_opener()

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Max file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Static files directory
STATIC_DIR = Path(__file__).parent.parent / "static"
DATA_DIR = Path(__file__).parent.parent / "data"
FONTS_DIR = DATA_DIR / "fonts"
DRIVE_MAPPING_FILE = DATA_DIR / "drive_mapping.json"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Thumbnail settings
THUMB_SIZE = (400, 300)
THUMB_QUALITY = 80

# Load Drive mapping if available
drive_mapping = {}
if DRIVE_MAPPING_FILE.exists():
    with open(DRIVE_MAPPING_FILE) as f:
        drive_mapping = json.load(f)
    logging.info(f"Loaded Drive mapping with {len(drive_mapping)} files")

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

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount fonts directory
if FONTS_DIR.exists():
    app.mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")

# Mount data directory for assets
if DATA_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DATA_DIR)), name="assets")


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
    """Serve the frontend"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
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
@limiter.limit("10/minute")
async def find_me(
    request: Request,
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

    - **file**: Image file (JPEG, PNG, HEIC, etc.)
    - **threshold**: Minimum similarity score (default: 0.4)
    - **limit**: Maximum results to return (default: 50)
    """
    # Validate file type (allow HEIC for iPhone)
    valid_types = ("image/", "application/octet-stream")  # octet-stream for some HEIC uploads
    is_heic = file.filename and file.filename.lower().endswith(('.heic', '.heif'))

    if not is_heic and (not file.content_type or not file.content_type.startswith("image/")):
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

        # Check file size (10MB limit)
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
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

        # Add photo URLs to matches (Drive only)
        for match in result["matches"]:
            filename = match['filename']
            if filename in drive_mapping:
                file_id = drive_mapping[filename]
                match["url"] = f"/drive-image/{file_id}"

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


@app.get("/drive-image/{file_id}")
async def drive_image_proxy(file_id: str, thumb: bool = Query(default=False)):
    """Proxy endpoint to serve Google Drive images with optional thumbnail caching"""

    # Check cache first
    cache_suffix = "_thumb" if thumb else "_full"
    cache_path = CACHE_DIR / f"{file_id}{cache_suffix}.jpg"

    if cache_path.exists():
        return FileResponse(
            cache_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=604800"}  # 7 days
        )

    # Fetch from Google Drive
    drive_url = f"https://drive.google.com/uc?export=view&id={file_id}"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(drive_url)

            if response.status_code != 200:
                raise HTTPException(status_code=404, detail="Image not found")

            content_type = response.headers.get("content-type", "image/jpeg")

            # Check if we got HTML instead of an image (Drive permission issue)
            if "text/html" in content_type:
                raise HTTPException(status_code=403, detail="Drive file not publicly accessible")

            # Process and cache the image
            try:
                img = Image.open(BytesIO(response.content))

                # Convert to RGB if necessary (handles PNG with transparency, etc.)
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')

                if thumb:
                    # Generate thumbnail with high-quality downscaling
                    img.thumbnail(THUMB_SIZE, Image.LANCZOS, reducing_gap=2.0)
                    img.save(cache_path, "JPEG", quality=THUMB_QUALITY, optimize=True)
                else:
                    # Cache full image with moderate compression
                    img.save(cache_path, "JPEG", quality=90, optimize=True)

                return FileResponse(
                    cache_path,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=604800"}
                )
            except Exception as img_error:
                logger.warning(f"Failed to process image, returning raw: {img_error}")
                # Fallback: return original content without caching
                return Response(
                    content=response.content,
                    media_type=content_type,
                    headers={"Cache-Control": "public, max-age=86400"}
                )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout fetching image")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error proxying Drive image: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch image")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
