"""
Stage 6: Production-Ready Whisper API
======================================

This is the culmination of our journey. Everything we learned:
1. Faster-Whisper with CTranslate2 (Stage 3)
2. INT8 quantization (Stage 3)
3. Backpressure and queuing (Stage 4)
4. VAD filtering (Stage 4)
5. Streaming support (Stage 5)
6. NEW: Structured logging
7. NEW: Prometheus metrics
8. NEW: Graceful shutdown
9. NEW: Request tracing
10. NEW: Circuit breaker pattern

This is what you'd deploy to production.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import tempfile
import os
import time
import json
import uuid
import logging
from typing import Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import threading
from faster_whisper import WhisperModel
import psutil

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Centralized configuration - in production, use environment variables"""
    MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
    COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "50"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ============================================================
# STRUCTURED LOGGING
# ============================================================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging - easily parsed by log aggregators"""
    
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        return json.dumps(log_data)


# Setup logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("whisper_api")
logger.addHandler(handler)
logger.setLevel(getattr(logging, Config.LOG_LEVEL))


# ============================================================
# METRICS (Prometheus-compatible)
# ============================================================

class Metrics:
    """Simple metrics collector - in production, use prometheus_client"""
    
    def __init__(self):
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.requests_timeout = 0
        self.requests_rejected = 0  # Backpressure rejections
        self.total_audio_seconds = 0.0
        self.total_processing_seconds = 0.0
        self.active_requests = 0
        self._lock = threading.Lock()
    
    def increment(self, metric: str, value: float = 1):
        with self._lock:
            current = getattr(self, metric, 0)
            setattr(self, metric, current + value)
    
    def to_prometheus(self) -> str:
        """Export in Prometheus text format"""
        lines = [
            f"# HELP whisper_requests_total Total requests",
            f"# TYPE whisper_requests_total counter",
            f"whisper_requests_total {self.requests_total}",
            f"",
            f"# HELP whisper_requests_success Successful requests",
            f"# TYPE whisper_requests_success counter",
            f"whisper_requests_success {self.requests_success}",
            f"",
            f"# HELP whisper_requests_failed Failed requests",
            f"# TYPE whisper_requests_failed counter",
            f"whisper_requests_failed {self.requests_failed}",
            f"",
            f"# HELP whisper_active_requests Current active requests",
            f"# TYPE whisper_active_requests gauge",
            f"whisper_active_requests {self.active_requests}",
            f"",
            f"# HELP whisper_audio_seconds_total Total audio processed",
            f"# TYPE whisper_audio_seconds_total counter",
            f"whisper_audio_seconds_total {self.total_audio_seconds:.2f}",
        ]
        return "\n".join(lines)


metrics = Metrics()


# ============================================================
# CIRCUIT BREAKER
# ============================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker pattern from DDIA.
    
    Prevents cascading failures by rejecting requests when
    the system is in a degraded state.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = 0
        self.half_open_successes = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                # Check if we should try half-open
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_successes = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                    return True
                return False
            
            # HALF_OPEN: Allow limited requests
            return True
    
    def record_success(self):
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_requests:
                    self.state = CircuitState.CLOSED
                    self.failures = 0
                    logger.info("Circuit breaker: HALF_OPEN -> CLOSED (recovered)")
            else:
                self.failures = 0
    
    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker: HALF_OPEN -> OPEN (still failing)")
            elif self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker: CLOSED -> OPEN (threshold reached)")


circuit_breaker = CircuitBreaker()


# ============================================================
# GLOBAL STATE
# ============================================================

executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
model: WhisperModel = None
request_queue: asyncio.Queue = None
shutdown_event: asyncio.Event = None


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list
    processing_time: float
    realtime_factor: float


# ============================================================
# APPLICATION LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Production lifecycle management"""
    global model, request_queue, shutdown_event
    
    logger.info("Starting application", extra={"extra": {"config": {
        "model": Config.MODEL_NAME,
        "compute_type": Config.COMPUTE_TYPE,
        "max_workers": Config.MAX_WORKERS
    }}})
    
    # Initialize
    request_queue = asyncio.Queue(maxsize=Config.MAX_QUEUE_SIZE)
    shutdown_event = asyncio.Event()
    
    # Load model
    start = time.time()
    model = WhisperModel(
        Config.MODEL_NAME,
        device="cpu",
        compute_type=Config.COMPUTE_TYPE,
        cpu_threads=Config.MAX_WORKERS,
        num_workers=2
    )
    
    # Warmup
    warmup_audio = os.path.join(
        os.path.dirname(__file__), "..", "..", "audio_samples", "sample_10s.wav"
    )
    if os.path.exists(warmup_audio):
        list(model.transcribe(warmup_audio, beam_size=1))
    
    load_time = time.time() - start
    logger.info(f"Model loaded", extra={"extra": {"load_time_seconds": load_time}})
    
    yield
    
    # Graceful shutdown
    logger.info("Initiating graceful shutdown")
    shutdown_event.set()
    
    # Wait for in-flight requests (with timeout)
    shutdown_start = time.time()
    while metrics.active_requests > 0 and (time.time() - shutdown_start) < 30:
        await asyncio.sleep(0.5)
        logger.info(f"Waiting for {metrics.active_requests} requests to complete")
    
    executor.shutdown(wait=True, cancel_futures=False)
    logger.info("Shutdown complete")


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Whisper Transcription API",
    description="Production-ready speech-to-text API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MIDDLEWARE
# ============================================================

@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Add request ID and timing to all requests"""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    request.state.start_time = time.time()
    
    response = await call_next(request)
    
    duration_ms = (time.time() - request.state.start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Duration-MS"] = str(int(duration_ms))
    
    # Log request completion
    logger.info(
        f"{request.method} {request.url.path}",
        extra={"extra": {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": int(duration_ms)
        }}
    )
    
    return response


# ============================================================
# HEALTH ENDPOINTS
# ============================================================

@app.get("/health")
async def health():
    """Comprehensive health check"""
    queue_depth = request_queue.qsize() if request_queue else 0
    queue_capacity = queue_depth / Config.MAX_QUEUE_SIZE
    
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Determine health status
    status = "healthy"
    if circuit_breaker.state == CircuitState.OPEN:
        status = "unhealthy"
    elif queue_capacity > 0.8:
        status = "degraded"
    
    return {
        "status": status,
        "circuit_breaker": circuit_breaker.state.value,
        "queue_depth": queue_depth,
        "queue_capacity_percent": round(queue_capacity * 100, 1),
        "active_requests": metrics.active_requests,
        "memory_mb": round(memory_mb, 1),
        "model": Config.MODEL_NAME,
        "uptime_seconds": round(time.time() - getattr(app.state, "start_time", time.time()), 1)
    }


@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe"""
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if circuit_breaker.state == CircuitState.OPEN:
        raise HTTPException(503, "Circuit breaker open")
    return {"ready": True}


@app.get("/health/live")  
async def liveness():
    """Kubernetes liveness probe"""
    return {"alive": True}


# ============================================================
# METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint"""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=metrics.to_prometheus(),
        media_type="text/plain"
    )


# ============================================================
# TRANSCRIPTION CORE
# ============================================================

def _transcribe_core(
    audio_path: str,
    language: Optional[str] = None,
    vad_filter: bool = True
) -> TranscriptResult:
    """Core transcription logic - runs in thread pool"""
    
    options = {
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "vad_filter": vad_filter,
    }
    
    if language:
        options["language"] = language
    
    start = time.time()
    segments_gen, info = model.transcribe(audio_path, **options)
    
    segments = []
    full_text = []
    
    for i, seg in enumerate(segments_gen):
        segments.append(TranscriptSegment(
            index=i,
            start=round(seg.start, 2),
            end=round(seg.end, 2),
            text=seg.text.strip()
        ))
        full_text.append(seg.text.strip())
    
    processing_time = time.time() - start
    
    return TranscriptResult(
        text=" ".join(full_text),
        language=info.language,
        language_probability=round(info.language_probability, 2),
        duration=round(info.duration, 2),
        segments=[asdict(s) for s in segments],
        processing_time=round(processing_time, 2),
        realtime_factor=round(processing_time / info.duration, 3) if info.duration > 0 else 0
    )


# ============================================================
# API ENDPOINTS
# ============================================================

@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="ISO language code"),
    vad_filter: bool = Query(True, description="Enable VAD filtering"),
):
    """
    Production transcription endpoint.
    
    Features:
    - Request tracing
    - Backpressure
    - Circuit breaker
    - Timeout handling
    - Structured response
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Circuit breaker check
    if not circuit_breaker.can_execute():
        metrics.increment("requests_rejected")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service temporarily unavailable",
                "reason": "circuit_breaker_open",
                "retry_after": 30
            }
        )
    
    # Backpressure check
    if request_queue.full():
        metrics.increment("requests_rejected")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Server overloaded",
                "queue_depth": request_queue.qsize(),
                "retry_after": 5
            }
        )
    
    # File size check
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > Config.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {Config.MAX_FILE_SIZE_MB}MB"
        )
    
    temp_path = None
    metrics.increment("requests_total")
    metrics.increment("active_requests")
    
    try:
        # Save file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(content)
            temp_path = f.name
        
        # Execute with timeout
        loop = asyncio.get_event_loop()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    _transcribe_core,
                    temp_path,
                    language,
                    vad_filter
                ),
                timeout=Config.REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            metrics.increment("requests_timeout")
            circuit_breaker.record_failure()
            raise HTTPException(504, f"Timeout after {Config.REQUEST_TIMEOUT}s")
        
        # Record success
        circuit_breaker.record_success()
        metrics.increment("requests_success")
        metrics.increment("total_audio_seconds", result.duration)
        metrics.increment("total_processing_seconds", result.processing_time)
        
        return JSONResponse({
            "request_id": request_id,
            "text": result.text,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration_seconds": result.duration,
            "segments": result.segments,
            "metrics": {
                "processing_seconds": result.processing_time,
                "realtime_factor": result.realtime_factor,
                "file_size_mb": round(file_size_mb, 2)
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.increment("requests_failed")
        circuit_breaker.record_failure()
        logger.error(f"Transcription error: {e}", extra={"extra": {"request_id": request_id}})
        raise HTTPException(500, str(e))
    
    finally:
        metrics.increment("active_requests", -1)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/transcribe/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None),
):
    """Streaming transcription with SSE"""
    
    if not circuit_breaker.can_execute():
        raise HTTPException(503, "Service unavailable")
    
    content = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(content)
        temp_path = f.name
    
    async def generate():
        result_queue = queue.Queue()
        
        def transcribe_worker():
            try:
                segments, info = model.transcribe(
                    temp_path,
                    beam_size=5,
                    vad_filter=True
                )
                
                for i, seg in enumerate(segments):
                    result_queue.put(("segment", {
                        "index": i,
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": seg.text.strip()
                    }))
                
                result_queue.put(("metadata", {
                    "language": info.language,
                    "duration": round(info.duration, 2)
                }))
                result_queue.put(("done", None))
                
            except Exception as e:
                result_queue.put(("error", str(e)))
        
        # Start worker
        thread = threading.Thread(target=transcribe_worker)
        thread.start()
        
        # Stream results
        while True:
            try:
                event_type, data = result_queue.get(timeout=0.1)
                
                if event_type == "segment":
                    yield f"event: segment\ndata: {json.dumps(data)}\n\n"
                elif event_type == "metadata":
                    yield f"event: metadata\ndata: {json.dumps(data)}\n\n"
                elif event_type == "done":
                    yield f"event: done\ndata: {{}}\n\n"
                    break
                elif event_type == "error":
                    yield f"event: error\ndata: {json.dumps({'error': data})}\n\n"
                    break
            except queue.Empty:
                yield f": keepalive\n\n"
                if not thread.is_alive():
                    break
        
        # Cleanup
        thread.join(timeout=5)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Whisper Transcription API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }
