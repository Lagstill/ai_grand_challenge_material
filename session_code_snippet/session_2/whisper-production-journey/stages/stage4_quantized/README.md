# Stage 4: Advanced Optimization - Backpressure & VAD ⚡

## The Story

Stage 3 gave us 4x speedup. But we're still getting failures under heavy load.

The problem: We accept every request, even when we can't handle them.

**Solution**: Backpressure + Smarter Processing

## What Changed

### 1. Backpressure (Reject When Full)
```python
MAX_QUEUE_SIZE = 50

async def _check_queue_capacity():
    if request_queue.full():
        raise HTTPException(
            status_code=503,
            detail={"error": "Server overloaded", "retry_after": 5}
        )
```

**Why this matters**: 
- Without backpressure: Queue grows infinitely → Memory exhaustion → Crash
- With backpressure: Reject early → Client retries → System stays healthy

### 2. VAD (Voice Activity Detection)
```python
options = {
    "vad_filter": True,
    "vad_parameters": {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    }
}
```

**Why this matters**:
- Audio with pauses/silence: VAD skips non-speech portions
- 30s audio with 10s silence → Only process 20s of speech
- 33% less computation for typical audio

### 3. Health Check Gradations
```python
@app.get("/health")
async def health_check():
    queue_depth = request_queue.qsize()
    
    if queue_depth > MAX_QUEUE_SIZE * 0.8:
        status = "degraded"  # Load balancer can route elsewhere
    else:
        status = "healthy"
```

### 4. Rich Metrics
Every response includes:
```json
{
  "metrics": {
    "queue_wait_seconds": 0.023,
    "inference_seconds": 1.234,
    "total_seconds": 1.257,
    "realtime_factor": 0.12,
    "file_size_mb": 0.45
  }
}
```

## The Numbers

| Metric | Stage 3 | Stage 4 | Improvement |
|--------|---------|---------|-------------|
| p99 latency | 6s | 4s | 33% better |
| Requests/min (sustained) | 150 | 250 | 67% more |
| Failure rate (100 users) | 5% | <1% | 5x better |
| Memory stability | Growing | Stable | ✓ |

## Run It

```bash
docker build -t whisper-optimized .
docker run -p 8000:8000 whisper-optimized

# Test with VAD enabled (default)
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@../../audio_samples/sample_10s.wav"

# Check metrics
curl http://localhost:8000/metrics

# Check queue status
curl http://localhost:8000/health
```

## Load Test Again

```bash
cd ../stage2_load_test_fail
locust -f locustfile.py --host=http://localhost:8000
```

Start with 50 users. Watch:
- 503 responses when queue fills (intended!)
- Consistent p99 latency (no runaway growth)
- Memory stays flat

## The DDIA Pattern: Backpressure

From "Designing Data-Intensive Applications":

> "If a system is designed to handle a certain request rate, but it receives a higher rate, 
> it has two choices: queue the requests or reject them. 
> If queues grow without bound, the system will eventually run out of memory.
> **A better approach is to reject requests when the system is overloaded.**"

This is what we implemented. The 503 responses aren't failures - they're the system protecting itself.

## Real-World Examples

### AssemblyAI's Architecture
- Uses VAD to segment audio before processing
- Processes segments in parallel
- Rejects requests at edge when overloaded

### Hugging Face Inference Endpoints
- Queue-based request handling
- Autoscaling based on queue depth
- 503 responses trigger client retries with backoff

## Next: Stage 5 - Streaming

Right now, users wait for the entire transcription to complete.

What if we could stream results as they're generated?
- Better perceived latency
- Earlier error detection
- Progressive UI updates
