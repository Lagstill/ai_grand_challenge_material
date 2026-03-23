# Stage 1: The Naive API 🎭

## The Story

You've got a working notebook. Your PM is excited. Stakeholders want a demo.

"Just wrap it in an API" they said. "It'll be fine" they said.

## What This Stage Demonstrates

✅ **What works:**
- Single requests complete successfully
- Returns correct transcriptions
- "Looks" production-ready

❌ **What will break:**
- Concurrent requests
- Large files
- Sustained load
- Network timeouts

## Run It

```bash
# Build
docker build -t whisper-naive .

# Run
docker run -p 8000:8000 whisper-naive

# Test single request (works!)
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@../../audio_samples/sample_10s.wav"
```

## The False Confidence

Try this and watch it "work":

```bash
# Single request - beautiful
time curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@../../audio_samples/sample_10s.wav"
```

Output:
```json
{
  "text": "Hello, this is a test transcription.",
  "processing_time_seconds": 2.34
}
```

🎉 Ship it! ...right?

## Next: Stage 2 - The Load Test Apocalypse

We'll use Locust to simulate real traffic and watch this API crumble.
