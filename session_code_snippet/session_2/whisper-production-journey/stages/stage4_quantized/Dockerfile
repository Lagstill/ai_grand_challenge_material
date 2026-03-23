# Stage 4: Production-Oriented Dockerfile
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

# Security: Non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY main.py .

# Model cache and temp directories
RUN mkdir -p /tmp/whisper_models /tmp/uploads && \
    chown -R appuser:appuser /tmp/whisper_models /tmp/uploads /app

USER appuser

# Pre-download model during build (optional, uncomment if needed)
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

EXPOSE 8000

# uvicorn with optimal settings for CPU
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--limit-concurrency", "20", \
     "--timeout-keep-alive", "30"]
