# Stage 3: Optimized Dockerfile with Faster-Whisper
FROM python:3.10-slim

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Create model cache directory with correct permissions
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Health check - now actually useful
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Multiple workers for concurrent requests
# Workers = (2 × CPU cores) + 1 is a common formula
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
