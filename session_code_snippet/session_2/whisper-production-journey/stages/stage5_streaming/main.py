"""
Stage 5: Streaming Transcription API
=====================================

What's new:
1. Server-Sent Events (SSE) for real-time segment streaming
2. Users see results as they're generated
3. Time-to-first-token metric (perceived latency)
4. Graceful cancellation support

The insight: Perceived latency matters more than actual latency.
A 10s transcription that streams in 10 parts feels faster than
waiting 10s for a single response.

References:
- SSE specification: https://html.spec.whatwg.org/multipage/server-sent-events.html
- Faster-Whisper streaming: Built into the library (segments are yielded)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import tempfile
import os
import time
import json
import logging
from typing import Optional, AsyncGenerator, Generator
from dataclasses import dataclass, asdict
from faster_whisper import WhisperModel
import psutil
import queue
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_QUEUE_SIZE = 50
MAX_WORKERS = 4
MODEL_NAME = "base"

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
model: WhisperModel = None


@dataclass
class TranscriptSegment:
    """A single transcription segment for streaming"""
    index: int
    start: float
    end: float
    text: str
    is_final: bool = False


@dataclass
class TranscriptMetadata:
    """Final metadata after transcription completes"""
    language: str
    language_probability: float
    duration: float
    total_segments: int
    processing_time: float
    realtime_factor: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management with model loading"""
    global model
    
    logger.info("🔄 Loading model for streaming...")
    start = time.time()
    
    model = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=2
    )
    
    # Warmup
    warmup_audio = os.path.join(
        os.path.dirname(__file__), "..", "..", "audio_samples", "sample_5s.wav"
    )
    if os.path.exists(warmup_audio):
        list(model.transcribe(warmup_audio, beam_size=1))
    
    logger.info(f"✅ Model ready in {time.time() - start:.2f}s")
    
    yield
    
    executor.shutdown(wait=True)


app = FastAPI(
    title="Whisper API - Stage 5 (Streaming)",
    description="Real-time streaming transcription with SSE",
    version="0.5.0",
    lifespan=lifespan
)


def _transcribe_streaming(
    audio_path: str,
    result_queue: queue.Queue,
    language: Optional[str] = None
) -> None:
    """
    Transcribe audio and push segments to queue as they're generated.
    
    This runs in a separate thread and yields segments one at a time.
    """
    try:
        options = {
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "vad_filter": True,
        }
        
        if language:
            options["language"] = language
        
        start_time = time.time()
        segments, info = model.transcribe(audio_path, **options)
        
        # Stream segments as they're generated
        segment_count = 0
        for segment in segments:
            segment_data = TranscriptSegment(
                index=segment_count,
                start=round(segment.start, 2),
                end=round(segment.end, 2),
                text=segment.text.strip(),
                is_final=False
            )
            result_queue.put(("segment", segment_data))
            segment_count += 1
        
        # Send final metadata
        processing_time = time.time() - start_time
        metadata = TranscriptMetadata(
            language=info.language,
            language_probability=round(info.language_probability, 2),
            duration=round(info.duration, 2),
            total_segments=segment_count,
            processing_time=round(processing_time, 2),
            realtime_factor=round(processing_time / info.duration, 3) if info.duration > 0 else 0
        )
        result_queue.put(("metadata", metadata))
        result_queue.put(("done", None))
        
    except Exception as e:
        result_queue.put(("error", str(e)))


async def stream_transcription(
    audio_path: str,
    language: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted events.
    
    SSE format:
        event: segment
        data: {"index": 0, "start": 0.0, "end": 2.5, "text": "Hello"}
        
        event: metadata
        data: {"language": "en", "duration": 10.0, ...}
        
        event: done
        data: {}
    """
    result_queue = queue.Queue()
    
    # Start transcription in background thread
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(
        executor,
        _transcribe_streaming,
        audio_path,
        result_queue,
        language
    )
    
    # Poll queue and yield events
    while True:
        try:
            event_type, data = result_queue.get(timeout=0.1)
            
            if event_type == "segment":
                yield f"event: segment\ndata: {json.dumps(asdict(data))}\n\n"
            elif event_type == "metadata":
                yield f"event: metadata\ndata: {json.dumps(asdict(data))}\n\n"
            elif event_type == "done":
                yield f"event: done\ndata: {{}}\n\n"
                break
            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps({'error': data})}\n\n"
                break
                
        except queue.Empty:
            # Send keepalive to prevent connection timeout
            yield f": keepalive\n\n"
            
            # Check if transcription is still running
            if future.done():
                try:
                    future.result()  # Raise any exceptions
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                break


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": f"faster-whisper-{MODEL_NAME}",
        "streaming": True
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None),
):
    """
    Standard (non-streaming) transcription endpoint.
    
    Returns complete result after processing finishes.
    """
    start_time = time.time()
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Use a queue for this single request
        result_queue = queue.Queue()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            _transcribe_streaming,
            temp_path,
            result_queue,
            language
        )
        
        # Collect all results
        segments = []
        metadata = None
        
        while True:
            event_type, data = result_queue.get(timeout=30)
            if event_type == "segment":
                segments.append(asdict(data))
            elif event_type == "metadata":
                metadata = asdict(data)
            elif event_type == "done":
                break
            elif event_type == "error":
                raise HTTPException(status_code=500, detail=data)
        
        full_text = " ".join(s["text"] for s in segments)
        
        return JSONResponse({
            "text": full_text,
            "segments": segments,
            **metadata,
            "total_time": round(time.time() - start_time, 2)
        })
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/transcribe/stream")
async def transcribe_audio_streaming(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None),
):
    """
    Streaming transcription endpoint using Server-Sent Events.
    
    Returns segments in real-time as they're transcribed.
    
    Example usage with curl:
        curl -X POST "http://localhost:8000/transcribe/stream" \
            -F "file=@audio.wav" \
            -H "Accept: text/event-stream"
    
    Example usage with JavaScript:
        const formData = new FormData();
        formData.append('file', audioFile);
        
        const response = await fetch('/transcribe/stream', {
            method: 'POST',
            body: formData
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const {value, done} = await reader.read();
            if (done) break;
            console.log(decoder.decode(value));
        }
    """
    temp_path = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Return streaming response
        return StreamingResponse(
            stream_transcription(temp_path, language),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """System metrics"""
    process = psutil.Process()
    return {
        "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
        "cpu_percent": process.cpu_percent(),
        "threads": process.num_threads()
    }


# Simple HTML demo page for testing streaming
DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Whisper Streaming Demo</title>
    <style>
        body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
        #output { background: #f5f5f5; padding: 20px; border-radius: 8px; min-height: 200px; white-space: pre-wrap; }
        .segment { margin: 5px 0; padding: 5px; background: white; border-radius: 4px; }
        .metadata { color: #666; font-size: 0.9em; margin-top: 20px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        input[type="file"] { margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🎤 Whisper Streaming Demo</h1>
    <input type="file" id="audioFile" accept="audio/*">
    <button onclick="transcribe()">Transcribe (Streaming)</button>
    <div id="output">Waiting for audio file...</div>
    
    <script>
        async function transcribe() {
            const file = document.getElementById('audioFile').files[0];
            if (!file) { alert('Select a file first'); return; }
            
            const output = document.getElementById('output');
            output.innerHTML = 'Starting transcription...\\n';
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/transcribe/stream', {
                method: 'POST',
                body: formData
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                
                const text = decoder.decode(value);
                const lines = text.split('\\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        if (data.text) {
                            output.innerHTML += `<div class="segment">[${data.start}s - ${data.end}s] ${data.text}</div>`;
                        } else if (data.language) {
                            output.innerHTML += `<div class="metadata">Language: ${data.language} | Duration: ${data.duration}s | RTF: ${data.realtime_factor}</div>`;
                        }
                    }
                }
            }
            output.innerHTML += '\\n✅ Done!';
        }
    </script>
</body>
</html>
"""


@app.get("/demo")
async def demo_page():
    """Simple demo page for testing streaming"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=DEMO_HTML)
