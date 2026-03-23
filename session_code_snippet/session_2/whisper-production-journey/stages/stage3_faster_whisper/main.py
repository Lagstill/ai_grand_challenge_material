"""
Stage 3: Faster-Whisper - The First Optimization
=================================================

What changed:
1. Switched from OpenAI Whisper to Faster-Whisper (CTranslate2 backend)
2. Added concurrent request handling with ThreadPoolExecutor
3. Proper async/await patterns
4. Request timeout handling

Real-world reference:
- SYSTRAN's CTranslate2: https://github.com/OpenNMT/CTranslate2
- Faster-Whisper: https://github.com/guillaumekln/faster-whisper

Expected improvement: 4x faster inference on CPU
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import tempfile
import os
import time
from typing import Optional
from faster_whisper import WhisperModel

# Thread pool for CPU-bound transcription work
# Key insight: Don't block the event loop, offload to threads
executor = ThreadPoolExecutor(max_workers=4)  # Tune based on CPU cores

# Global model instance
model: WhisperModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Proper lifecycle management.
    
    Model loading happens ONCE at startup, not per-request.
    Graceful shutdown ensures in-flight requests complete.
    """
    global model
    
    print("🔄 Loading Faster-Whisper model...")
    start = time.time()
    
    # CTranslate2 backend with CPU optimization
    # "auto" will use INT8 on CPU for best performance
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",  # Key optimization!
        cpu_threads=4,        # Match thread pool size
        num_workers=2         # Parallel beam search workers
    )
    
    # Warmup - important for consistent first-request latency
    # Without warmup, first request is 2-3x slower
    print("🔥 Warming up model...")
    warmup_audio = os.path.join(os.path.dirname(__file__), "..", "..", "audio_samples", "sample_10s.wav")
    if os.path.exists(warmup_audio):
        list(model.transcribe(warmup_audio))
    
    load_time = time.time() - start
    print(f"✅ Model loaded and warmed up in {load_time:.2f}s")
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    print("🛑 Shutting down gracefully...")
    executor.shutdown(wait=True)


app = FastAPI(
    title="Whisper API - Stage 3 (Faster-Whisper)",
    description="4x faster with CTranslate2 backend and proper async",
    version="0.3.0",
    lifespan=lifespan
)


def _transcribe_sync(audio_path: str, language: Optional[str] = None) -> dict:
    """
    Synchronous transcription - runs in thread pool.
    
    Separated from async handler for cleaner code and better profiling.
    """
    transcribe_options = {
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,  # Deterministic output
    }
    
    if language:
        transcribe_options["language"] = language
    
    segments, info = model.transcribe(audio_path, **transcribe_options)
    
    # Collect all segments
    all_segments = []
    full_text = []
    
    for segment in segments:
        all_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
        full_text.append(segment.text.strip())
    
    return {
        "text": " ".join(full_text),
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": all_segments
    }


@app.get("/health")
async def health_check():
    """
    Improved health check - actually verifies model is loaded.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": "faster-whisper-base",
        "compute_type": "int8",
        "device": "cpu"
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    """
    Transcribe audio with proper async handling.
    
    Key improvements:
    1. Runs transcription in thread pool (non-blocking)
    2. Proper timeout handling via asyncio
    3. Cleanup guaranteed via try/finally
    """
    start_time = time.time()
    temp_path = None
    
    try:
        # Save file (still synchronous, but fast)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Run transcription in thread pool with timeout
        loop = asyncio.get_event_loop()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    _transcribe_sync,
                    temp_path,
                    language
                ),
                timeout=60.0  # 60 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Transcription timeout - file may be too long"
            )
        
        elapsed = time.time() - start_time
        
        return JSONResponse({
            **result,
            "processing_time_seconds": round(elapsed, 2),
            "realtime_factor": round(result["duration"] / elapsed, 2) if elapsed > 0 else 0
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/metrics")
async def get_metrics():
    """
    Basic metrics endpoint for monitoring.
    
    In production, you'd use Prometheus + Grafana.
    This is a simplified version for the demo.
    """
    import psutil
    
    process = psutil.Process()
    
    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "threads": process.num_threads(),
        "executor_queue_size": executor._work_queue.qsize() if hasattr(executor._work_queue, 'qsize') else "unknown"
    }
