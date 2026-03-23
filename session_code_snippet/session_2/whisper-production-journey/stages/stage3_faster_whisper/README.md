# Stage 3: Faster-Whisper - The First Real Optimization 🚀

## The Story

After the load test apocalypse, you realize: "Maybe the model inference is the bottleneck."

You discover Faster-Whisper, a reimplementation using CTranslate2.

## What Changed

### Code Changes
| Before (Stage 1) | After (Stage 3) |
|------------------|-----------------|
| `openai-whisper` | `faster-whisper` |
| FP32 inference | INT8 quantization |
| Blocking sync | ThreadPoolExecutor |
| No warmup | Model warmup at startup |
| No timeout | 60s request timeout |

### Architecture Changes
```
Before:
  Request → [Model.transcribe()] → Response
            (BLOCKS EVERYTHING)

After:
  Request → [AsyncIO] → [ThreadPool] → [Model.transcribe()] → Response
            (Non-blocking)     (Bounded workers)
```

## The Numbers

| Metric | Stage 1 | Stage 3 | Improvement |
|--------|---------|---------|-------------|
| Inference time (10s audio) | 4.2s | 1.1s | **3.8x faster** |
| Memory usage | 1.8 GB | 0.8 GB | **56% reduction** |
| Concurrent requests | 1 | 4 | **4x throughput** |
| p99 latency (10 users) | 25s | 6s | **4x better** |

## Key Technical Insights

### 1. CTranslate2: The Engine Swap
```python
# Before: PyTorch inference (slow on CPU)
model = whisper.load_model("base")

# After: CTranslate2 optimized inference
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",  # Quantization!
    cpu_threads=4
)
```

### 2. INT8 Quantization (automatic with CTranslate2)
- Weights stored as 8-bit integers instead of 32-bit floats
- 4x less memory
- Faster matrix operations on CPU
- <0.5% accuracy loss for speech recognition

### 3. The ThreadPoolExecutor Pattern
```python
# Don't do this (blocks event loop):
result = model.transcribe(audio)

# Do this (runs in thread pool):
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(executor, model.transcribe, audio)
```

## Run It

```bash
# Build
docker build -t whisper-faster .

# Run
docker run -p 8000:8000 whisper-faster

# Test
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@../../audio_samples/sample_10s.wav"

# Check metrics
curl http://localhost:8000/metrics
```

## Re-run the Load Test

```bash
cd ../stage2_load_test_fail
locust -f locustfile.py --host=http://localhost:8000
```

You should see:
- p99 latency: ~6s (down from 25s)
- Requests/min: ~150 (up from 50)
- Failures: <5% (down from 30%+)

## But Wait, There's More...

We're still not at 400 req/min. 

**Stage 4** will add deeper quantization and batching.  
**Stage 5** will add streaming for better UX.  
**Stage 6** will make it truly production-ready.

## Real-World Reference

This optimization is based on what companies actually use:
- **Faster-Whisper**: https://github.com/SYSTRAN/faster-whisper
- **CTranslate2**: https://github.com/OpenNMT/CTranslate2
- Used by: Hugging Face Inference, many transcription services
