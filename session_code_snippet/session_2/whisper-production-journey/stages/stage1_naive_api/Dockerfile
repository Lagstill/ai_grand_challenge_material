# Stage 1: Naive Dockerfile
# "It works, ship it!"

FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg for audio processing
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Single worker, no optimization
# This is what tutorials show you
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Problems:
# 1. No health check in Dockerfile
# 2. No resource limits
# 3. Running as root
# 4. No multi-stage build (image is huge)
# 5. Model downloads on first request
