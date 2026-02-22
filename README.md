---
title: SpotMe CUSEC
---

# SpotMe CUSEC

Find yourself in CUSEC 2026 photos using face recognition.

Upload a selfie and instantly discover all event photos where you appear.

---

## Try Out

- [cusecphotofinder-production.up.railway.app](https://cusecphotofinder-production.up.railway.app)
- [www.spotmecusec.xyz](https://www.spotmecusec.xyz)

---

## Features

- **Face Recognition** - Uses InsightFace (buffalo_l model) for accurate face detection and matching
- **Fast Search** - Pre-computed embeddings enable instant matching against thousands of photos
- **iPhone Support** - Accepts HEIC/HEIF images from iPhones
- **Thumbnail Caching** - Optimized image loading with server-side thumbnail generation
- **Rate Limiting** - Protected API with 10 requests/minute limit
- **Privacy First** - No uploaded images are stored

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python, FastAPI, Uvicorn |
| **Face Recognition** | InsightFace, ONNX Runtime |
| **Image Processing** | Pillow, OpenCV, pillow-heif |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Storage** | Google Drive (images), NumPy (embeddings) |

---

## How It Works

1. **Preprocessing** - Face embeddings are extracted from all event photos and stored in `data/embeddings/embeddings.npz`

2. **Upload** - User uploads a selfie through the web interface

3. **Match** - The uploaded face is compared against all stored embeddings using cosine similarity

4. **Results** - Photos with similarity above the threshold are returned, sorted by match confidence

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Threshold | 0.4 | Minimum similarity score (0-1). Lower = more results |
| Max Results | 50 | Maximum photos to return |
| File Size Limit | 10MB | Maximum upload size |
| Rate Limit | 10/min | Requests per minute per IP |

---

## Privacy

- Uploaded selfies are processed in memory and **never stored**
- All event photos are from publicly available sources
- No personal data is collected or retained

---

## Author

Made for CUSEC 2026 by Balawal
