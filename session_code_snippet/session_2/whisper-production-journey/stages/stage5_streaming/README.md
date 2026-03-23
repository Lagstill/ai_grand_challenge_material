# Stage 5: Streaming Transcription 🌊

## The Story

Users are complaining: "I upload a 2-minute audio and stare at a spinner for 30 seconds."

Even though we're fast, the UX is bad. Users don't see progress.

**Solution**: Stream results as they're generated.

## The Psychology of Perceived Latency

| Scenario | Actual Time | Perceived Time |
|----------|-------------|----------------|
| Wait 10s → Get full result | 10s | "Forever" |
| See progress every 1s | 10s | "Fast!" |

**Time to First Byte (TTFB)** matters more than total time for user experience.

## How Streaming Works

### Server-Sent Events (SSE)
```
Client → POST /transcribe/stream (with audio)
Server → HTTP 200, text/event-stream

event: segment
data: {"index": 0, "start": 0.0, "end": 2.5, "text": "Hello"}

event: segment  
data: {"index": 1, "start": 2.5, "end": 5.0, "text": "World"}

event: metadata
data: {"language": "en", "duration": 5.0, "realtime_factor": 0.12}

event: done
data: {}
```

### The Code Pattern
```python
async def stream_transcription(audio_path: str):
    # Transcription yields segments one at a time
    segments, info = model.transcribe(audio_path)
    
    for segment in segments:  # This is a generator!
        yield f"event: segment\ndata: {json.dumps(segment)}\n\n"
    
    yield f"event: done\ndata: {{}}\n\n"
```

## Run It

```bash
docker build -t whisper-streaming .
docker run -p 8000:8000 whisper-streaming

# Open the demo page in browser
open http://localhost:8000/demo

# Or test with curl
curl -X POST "http://localhost:8000/transcribe/stream" \
  -F "file=@../../audio_samples/sample_30s.wav" \
  -H "Accept: text/event-stream"
```

## JavaScript Client Example

```javascript
async function transcribeWithStreaming(audioFile) {
    const formData = new FormData();
    formData.append('file', audioFile);
    
    const response = await fetch('/transcribe/stream', {
        method: 'POST',
        body: formData
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    let fullText = '';
    
    while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        // Parse SSE events
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                if (data.text) {
                    fullText += data.text + ' ';
                    updateUI(fullText);  // Show progress!
                }
            }
        }
    }
    
    return fullText;
}
```

## The Numbers

| Metric | Non-Streaming | Streaming | Impact |
|--------|---------------|-----------|--------|
| Time to first text | 10s | 1.2s | **88% faster perceived** |
| Total time | 10s | 10.5s | 5% slower (overhead) |
| User satisfaction | "Slow" | "Responsive" | 📈 |

## When to Use Streaming

✅ **Use streaming when:**
- User is watching/waiting
- Audio is long (>30s)
- You want to show progress
- Building real-time applications

❌ **Don't use streaming when:**
- Batch processing (no one watching)
- API consumers want simple JSON
- Very short audio (<5s)

## Production Considerations

### 1. Proxy Buffering
nginx and other proxies buffer responses by default. Disable it:
```nginx
location /transcribe/stream {
    proxy_buffering off;
    proxy_cache off;
}
```

### 2. Connection Timeouts
Long transcriptions need longer timeouts:
```python
CMD ["uvicorn", "main:app", "--timeout-keep-alive", "120"]
```

### 3. Error Handling
SSE can't use HTTP status codes mid-stream. Send error events:
```python
yield f"event: error\ndata: {{'error': 'Something went wrong'}}\n\n"
```

## Real-World Examples

### YouTube Auto-Captions
- Streams captions in real-time during live streams
- Shows partial results that update

### Google Speech-to-Text
- Offers both streaming and batch modes
- Streaming for real-time, batch for archives

### Assembly AI
- WebSocket-based streaming
- Sub-second latency for live transcription

## Next: Stage 6 - Production Ready

We have a fast, streaming API. Now we need to make it bulletproof:
- Docker Compose with all services
- Health checks that actually work
- Graceful shutdown
- Logging and monitoring
- Ready for Kubernetes
