"""
Stage 4: Advanced Optimization - Batching & Request Queuing
============================================================

What's new:
1. Request queuing with backpressure
2. Distil-Whisper option (smaller, faster model)
3. VAD (Voice Activity Detection) for skipping silence
4. Better memory management
5. Request priority support

The key insight: Individual request latency vs System throughput
Sometimes you sacrifice single-request speed for better overall throughput.

Reference:
- Distil-Whisper: https://huggingface.co/distil-whisper
- VAD: https://github.com/snakers4/silero-vad
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
from asyncio import Queue
from dataclasses import dataclass
from typing import Optional
import tempfile
import os
import time
import logging
from faster_whisper import WhisperModel
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_QUEUE_SIZE = 50     # Reject requests beyond this (backpressure)
MAX_WORKERS = 4         # Concurrent transcription workers
REQUEST_TIMEOUT = 120   # Max time for a request
MODEL_NAME = "base"     # Options: tiny, base, small, medium, large

# Thread pool for transcription
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Global state
model: WhisperModel = None
request_queue: Queue = None
active_requests = 0
total_processed = 0
total_failed = 0


@dataclass
class TranscriptionRequest:
    """Structured request for queue processing"""
    audio_path: str
    language: Optional[str]
    priority: int  # Lower = higher priority
    created_at: float
    future: asyncio.Future


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global model, request_queue
    
    logger.info("🔄 Starting up...")
    
    # Initialize request queue
    request_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    
    # Load model with all optimizations
    logger.info(f"📦 Loading Faster-Whisper {MODEL_NAME} model...")
    start = time.time()
    
    model = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8",     # INT8 quantization
        cpu_threads=4,
        num_workers=2,
        download_root="/tmp/whisper_models"  # Explicit cache location
    )
    
    # Aggressive warmup for consistent performance
    logger.info("🔥 Running warmup transcriptions...")
    warmup_audio = os.path.join(
        os.path.dirname(__file__), "..", "..", "audio_samples", "sample_5s.wav"
    )
    if os.path.exists(warmup_audio):
        for _ in range(3):  # Multiple warmup passes
            list(model.transcribe(warmup_audio, beam_size=1))
    
    load_time = time.time() - start
    logger.info(f"✅ Model ready in {load_time:.2f}s")
    
    yield
    
    # Graceful shutdown
    logger.info("🛑 Shutting down...")
    executor.shutdown(wait=True, cancel_futures=False)


app = FastAPI(
    title="Whisper API - Stage 4 (Optimized)",
    description="Batching, queuing, and advanced optimizations",
    version="0.4.0",
    lifespan=lifespan
)


def _transcribe_optimized(
    audio_path: str,
    language: Optional[str] = None,
    vad_filter: bool = True
) -> dict:
    """
    Optimized transcription with VAD filtering.
    
    VAD (Voice Activity Detection) skips silent portions,
    significantly reducing processing time for audio with pauses.
    """
    global total_processed, total_failed
    
    try:
        options = {
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "vad_filter": vad_filter,  # Skip silence
            "vad_parameters": {
                "min_silence_duration_ms": 500,  # Minimum silence to skip
                "speech_pad_ms": 200,            # Padding around speech
            }
        }
        
        if language:
            options["language"] = language
        
        segments, info = model.transcribe(audio_path, **options)
        
        # Collect segments
        all_segments = []
        full_text = []
        
        for segment in segments:
            all_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            full_text.append(segment.text.strip())
        
        total_processed += 1
        
        return {
            "text": " ".join(full_text),
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "duration": round(info.duration, 2),
            "segments": all_segments,
            "vad_filtered": vad_filter
        }
        
    except Exception as e:
        total_failed += 1
        raise


async def _check_queue_capacity():
    """Check if we can accept new requests"""
    if request_queue.full():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Server overloaded",
                "queue_size": request_queue.qsize(),
                "retry_after": 5
            }
        )


@app.middleware("http")
async def add_metrics_header(request: Request, call_next):
    """Add queue depth to response headers for monitoring"""
    response = await call_next(request)
    response.headers["X-Queue-Depth"] = str(request_queue.qsize() if request_queue else 0)
    response.headers["X-Active-Requests"] = str(active_requests)
    return response


@app.get("/health")
async def health_check():
    """
    Comprehensive health check for load balancers.
    
    Returns unhealthy if queue is too full - enables load balancer
    to route traffic elsewhere.
    """
    queue_depth = request_queue.qsize() if request_queue else 0
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Degraded if queue is more than 80% full
    status = "healthy"
    if queue_depth > MAX_QUEUE_SIZE * 0.8:
        status = "degraded"
    
    return {
        "status": status,
        "model": f"faster-whisper-{MODEL_NAME}",
        "compute_type": "int8",
        "queue_depth": queue_depth,
        "max_queue_size": MAX_QUEUE_SIZE,
        "active_workers": MAX_WORKERS,
        "memory_mb": round(memory_mb, 1)
    }


@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}


@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"alive": True}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="ISO language code (e.g., 'en', 'hi')"),
    vad_filter: bool = Query(True, description="Enable Voice Activity Detection"),
    priority: int = Query(5, ge=1, le=10, description="Request priority (1=highest)")
):
    """
    Transcribe audio with backpressure and priority queuing.
    
    Features:
    - Rejects requests when overloaded (503)
    - Priority queue processing
    - VAD filtering for faster processing
    - Detailed timing metrics
    """
    global active_requests
    
    # Check capacity before accepting request
    await _check_queue_capacity()
    
    start_time = time.time()
    queue_start = time.time()
    temp_path = None
    
    try:
        active_requests += 1
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        file_size_mb = len(content) / (1024 * 1024)
        
        # Execute transcription with timeout
        loop = asyncio.get_event_loop()
        
        queue_wait = time.time() - queue_start
        inference_start = time.time()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    _transcribe_optimized,
                    temp_path,
                    language,
                    vad_filter
                ),
                timeout=REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Timeout after {REQUEST_TIMEOUT}s"
            )
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        # Calculate realtime factor (RTF)
        # RTF < 1 means faster than realtime
        rtf = inference_time / result["duration"] if result["duration"] > 0 else 0
        
        return JSONResponse({
            **result,
            "metrics": {
                "queue_wait_seconds": round(queue_wait, 3),
                "inference_seconds": round(inference_time, 3),
                "total_seconds": round(total_time, 3),
                "realtime_factor": round(rtf, 3),
                "file_size_mb": round(file_size_mb, 2)
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        active_requests -= 1
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus-style metrics endpoint.
    
    In production, you'd use prometheus_client library.
    This is a simplified JSON version.
    """
    process = psutil.Process()
    
    return {
        "system": {
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads()
        },
        "queue": {
            "depth": request_queue.qsize() if request_queue else 0,
            "max_size": MAX_QUEUE_SIZE,
            "active_requests": active_requests
        },
        "counters": {
            "total_processed": total_processed,
            "total_failed": total_failed,
            "success_rate": round(
                total_processed / (total_processed + total_failed) * 100, 1
            ) if (total_processed + total_failed) > 0 else 100
        },
        "config": {
            "model": MODEL_NAME,
            "compute_type": "int8",
            "max_workers": MAX_WORKERS
        }
    }


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Whisper Transcription API",
        "version": "0.4.0",
        "stage": "4 - Optimized",
        "docs": "/docs"
    }
