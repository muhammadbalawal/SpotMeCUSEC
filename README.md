# Face Finder

A web-based tool that allows a user to upload a photo of themselves and automatically find all photos (including group photos) where they appear, using face recognition.

---

## Project Goal

- Process a large folder of event photos once
- Detect and store face embeddings for every face in every photo
- Provide a web interface where a user uploads a photo of themselves
- Instantly return all photos where the user appears

---

## Tech Stack

### Backend
- Python 3.10+
- FastAPI
- DeepFace
- RetinaFace
- OpenCV
- NumPy

### Frontend
- React or Next.js
- Simple file upload UI
- Grid display of matching photos

### Storage
- Local filesystem for photos
- JSON or SQLite for embeddings
- Optional: FAISS for fast vector search

---

## High-Level Architecture

1. One-time photo preprocessing
2. Face detection and embedding extraction
3. Embedding storage
4. User photo upload
5. Face matching against stored embeddings
6. Return matching photos

---

## Phase 1: Data Preparation (One-Time)

### 1. Download Photos
- Download the entire Google Drive folder locally
- Store all images in a single directory (e.g. `data/photos/`)

### 2. Create Reference Folder
- Create a folder for test selfies (e.g. `data/reference/`)
- Add 3–5 clear photos of the same person
- Use different angles and lighting

---

## Phase 2: Preprocessing Pipeline

### Goal
Extract face embeddings from every photo once and store them for fast lookup.

### Steps
1. Loop through every photo in `data/photos/`
2. Detect all faces in the photo
3. Generate an embedding for each detected face
4. Save embeddings with the photo filename

### Output Format Example
```json
{
  "photo_001.jpg": [
    [0.12, -0.33, 0.88, ...],
    [0.55, 0.01, -0.22, ...]
  ],
  "photo_002.jpg": [
    [...]
  ]
}

