# Stage 6: Production Dockerfile
# Multi-stage build for smaller image

FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# ===========================================
# Runtime Stage
# ===========================================
FROM python:3.10-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL version="1.0.0"
LABEL description="Production Whisper Transcription API"

# Security: Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python dependencies
COPY --from=builder /build/deps /usr/local/lib/python3.10/site-packages

# Copy application
COPY main.py .

# Setup directories with proper permissions
RUN mkdir -p /tmp/whisper_models /tmp/uploads /app/logs && \
    chown -R appuser:appuser /tmp /app

# Switch to non-root user
USER appuser

# Environment defaults
ENV WHISPER_MODEL=base \
    COMPUTE_TYPE=int8 \
    MAX_WORKERS=4 \
    MAX_QUEUE_SIZE=50 \
    REQUEST_TIMEOUT=120 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

EXPOSE 8000

# Use tini for proper signal handling (graceful shutdown)
ENTRYPOINT ["/usr/bin/tini", "--"]

# Production uvicorn settings
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--limit-concurrency", "100", \
     "--timeout-keep-alive", "30", \
     "--access-log"]
