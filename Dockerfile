FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, image processing, and building InsightFace
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create cache directory
RUN mkdir -p data/cache

# Expose port
EXPOSE 8000

# Run the app (PORT env var set by Railway/HF)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
