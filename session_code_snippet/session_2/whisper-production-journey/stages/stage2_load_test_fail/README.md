# Stage 2: The Load Test Apocalypse 💥

## The Story

"It works in my tests!" → "Why is production on fire?"

This is the moment of truth. We're going to hit your naive API with realistic traffic and watch it crumble.

## What You'll See

### Before (expectations):
- ✅ 10 users
- ✅ ~50 requests/minute
- ✅ p99 latency: 3-5 seconds
- ✅ 0% failures

### Reality (what happens):
- ❌ Requests queue up
- ❌ p99 latency: 15-30+ seconds  
- ❌ Timeouts everywhere
- ❌ Server becomes unresponsive
- ❌ Health checks fail

## Run The Test

### Step 1: Start the naive API
```bash
cd ../stage1_naive_api
docker build -t whisper-naive .
docker run -p 8000:8000 whisper-naive
```

### Step 2: Run Locust
```bash
cd ../stage2_load_test_fail
pip install locust

# Start Locust
locust -f locustfile.py --host=http://localhost:8000
```

### Step 3: Open the Locust UI
Go to http://localhost:8089

### Step 4: Configure the test
- **Number of users**: 10
- **Spawn rate**: 2
- Click "Start swarming"

### Step 5: Watch it burn 🔥

## What's Happening (The Technical Explanation)

### The Blocking Problem
```python
# This line is the killer:
result = model.transcribe(temp_path)
```

Even though FastAPI is async, `whisper.transcribe()` is **synchronous**.  
It blocks the Python GIL. No other requests can be processed.

### The Queue Builds Up
```
Request 1: Processing... (3s)
Request 2: Waiting...
Request 3: Waiting...
Request 4: Waiting...
Request 5: Waiting...   ← Already 12s in queue
...
Request 10: Waiting... ← 30s wait time
```

### The Death Spiral
1. Requests queue up
2. Timeouts start happening
3. Clients retry (more requests!)
4. Memory pressure builds
5. Server crashes or becomes unresponsive

## The Metrics That Matter

| Metric | Target | Naive Reality |
|--------|--------|---------------|
| p99 latency | < 5s | 15-30s |
| Requests/min | 400 | 50 |
| Failure rate | 0% | 10-40% |
| CPU utilization | 80% | 100% (but only 1 core) |

## The "Designing Data-Intensive Applications" Lesson

> "An architecture that scales well will probably have **several mechanisms for preventing overload**:
> - Load shedding
> - Backpressure  
> - Rate limiting
> - Queueing with bounded queues"

Our naive API has **none of these**.

## Next: Stage 3 - The First Optimization

We'll introduce Faster-Whisper and CTranslate2 for a 4x speedup.
