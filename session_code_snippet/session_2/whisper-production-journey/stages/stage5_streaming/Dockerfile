# Stage 5: Streaming-optimized Dockerfile
FROM python:3.10-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN mkdir -p /tmp/whisper_models && chown -R appuser:appuser /tmp /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Single worker for streaming (multiple workers can cause issues with SSE)
# In production, use a reverse proxy to handle multiple connections
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "120"]
