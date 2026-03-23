"""
Stage 1: The Naive API
======================

"It works in my notebook, let me just wrap it in FastAPI"

This is what most tutorials teach you. 
This is what will fail in production.

Problems you'll discover:
1. Synchronous - blocks the entire server
2. No concurrency handling
3. Model reloaded on import (cold start issues)
4. No timeout handling
5. No error recovery
6. Memory grows unbounded with file uploads
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
import time
from typing import Optional

app = FastAPI(
    title="Whisper API - Stage 1 (Naive)",
    description="The 'it works on my machine' version",
    version="0.1.0"
)

# Global model loading - happens once at startup
# But what if it takes 30 seconds? What if it fails?
print("🔄 Loading Whisper model... (this blocks everything)")
start_load = time.time()
model = whisper.load_model("base")
print(f"✅ Model loaded in {time.time() - start_load:.2f}s")


@app.get("/health")
def health_check():
    """Basic health check - doesn't actually verify model is working"""
    return {"status": "healthy"}  # Lies. We don't know if it's healthy.


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    """
    Transcribe audio file.
    
    Problems with this implementation:
    1. 'async' is a lie - whisper.transcribe() is blocking
    2. Entire file loaded into memory
    3. No file size limits
    4. No timeout
    5. Temp file might not get cleaned up on error
    """
    start_time = time.time()
    
    # Save uploaded file to temp location
    # Problem: What if disk is full? What if file is 10GB?
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()  # Entire file in memory!
            temp_file.write(content)
            temp_path = temp_file.name
        
        # This is synchronous and BLOCKS the event loop
        # No other requests can be processed while this runs
        transcribe_options = {}
        if language:
            transcribe_options["language"] = language
            
        result = model.transcribe(temp_path, **transcribe_options)
        
        elapsed = time.time() - start_time
        
        return JSONResponse({
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "processing_time_seconds": round(elapsed, 2),
            "segments": len(result.get("segments", [])),
            # No mention of queue depth, memory usage, or system health
        })
        
    except Exception as e:
        # Generic exception handling - not production ready
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup - but this might not run if server crashes
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/transcribe_batch")
async def transcribe_batch(files: list[UploadFile] = File(...)):
    """
    Batch transcription - even worse!
    
    This will:
    1. Process files sequentially (no parallelism)
    2. Block for the entire duration
    3. Timeout on most load balancers
    4. Give no progress feedback
    """
    results = []
    total_start = time.time()
    
    for file in files:
        # Sequential processing - waste of potential
        result = await transcribe_audio(file)
        results.append(result)
    
    return {
        "results": results,
        "total_time": time.time() - total_start,
        "files_processed": len(files)
    }


# What's missing:
# - Request queuing
# - Concurrent processing  
# - Backpressure (rejecting requests when overloaded)
# - Metrics/monitoring
# - Proper async execution
# - Memory management
# - Graceful shutdown
# - Model warmup verification
