# 🎤 Whisper Production Journey

**From Notebook to Production: A Hands-On Workshop**

> "Your model works in Jupyter. PM says ship it. What could go wrong?"  
> — Every ML Engineer, before learning the hard way

---

## 🎯 Workshop Overview

This repository contains a complete, incremental journey of deploying a Whisper speech-to-text model from a working notebook to a production-ready API.

**Duration**: 60 minutes (hands-on demo)  
**Audience**: AI/ML teams transitioning research models to production  
**Prerequisites**: Docker, Python basics, familiarity with APIs

---

## 📊 The Results

| Metric | Start (Stage 1) | End (Stage 6) | Improvement |
|--------|-----------------|---------------|-------------|
| **Requests/min** | 50 | 400 | **8x** |
| **p99 Latency** | 25s | 1.8s | **14x** |
| **Failure Rate** | 30% | <0.5% | **60x** |
| **Cost/1000 req** | $0.057 | $0.007 | **8x cheaper** |

---

## 🗺️ The Journey

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1          STAGE 2         STAGE 3         STAGE 4          │
│  Notebook    →    Load Test   →   Faster      →   Quantized +     │
│  "It works!"      "It dies!"      Whisper         Backpressure     │
│                                   (4x faster)     (stability)      │
└─────────────────────────────────────────────────────────────────────┘
                                        │
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5          STAGE 6                                          │
│  Streaming    →   Production     "Ship it (for real this time)"   │
│  (better UX)      Ready                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Required
docker --version  # Docker Desktop
python --version  # Python 3.10+

# Optional (for audio generation)
pip install gtts pydub
```

### Generate Test Audio
```bash
cd scripts
python generate_audio.py
```

### Run the Demo (Stage by Stage)

```bash
# Stage 1: The naive API (watch it work... for one user)
cd stages/stage1_naive_api
docker build -t whisper-naive .
docker run -p 8000:8000 whisper-naive

# Test it
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@../../audio_samples/sample_10s.wav"
```

```bash
# Stage 2: Load test (watch it fail)
cd stages/stage2_load_test_fail
pip install locust
locust -f locustfile.py --host=http://localhost:8000
# Open http://localhost:8089, set 10 users, watch the chaos
```

```bash
# Stage 6: Production (watch it handle load)
cd stages/stage6_production
docker compose up -d
# Open http://localhost:8000/docs
```

---

## 📁 Repository Structure

```
whisper-production-journey/
├── README.md                    # You are here
├── notebooks/
│   └── 01_notebook_hero.ipynb   # The starting point
├── stages/
│   ├── stage1_naive_api/        # "It works on my machine"
│   ├── stage2_load_test_fail/   # The reality check (Locust)
│   ├── stage3_faster_whisper/   # First optimization (4x speedup)
│   ├── stage4_quantized/        # Backpressure + VAD
│   ├── stage5_streaming/        # Real-time streaming (SSE)
│   └── stage6_production/       # Full production setup
├── audio_samples/               # Test audio files
├── benchmarks/
│   └── results.md               # Performance data
└── scripts/
    └── generate_audio.py        # Audio generation utility
```

---

## 🔑 Key Concepts Covered

### From "Designing Data-Intensive Applications"

1. **Backpressure**
   > "If a system cannot keep up with the rate of incoming requests, it has three choices: drop requests, queue them, or apply backpressure."
   
   We implement backpressure in Stage 4 - rejecting requests early instead of queueing them infinitely.

2. **Circuit Breaker**
   > "If a downstream service is struggling, protect it by failing fast."
   
   Implemented in Stage 6 - after 5 failures, we stop trying for 30 seconds.

3. **Health Checks**
   > "A service should honestly report its health status."
   
   Three levels: `/health` (detailed), `/health/ready` (load balancer), `/health/live` (Kubernetes)

### Production Optimizations

| Technique | Stage | Impact |
|-----------|-------|--------|
| **Faster-Whisper (CTranslate2)** | 3 | 4x faster inference |
| **INT8 Quantization** | 3 | 56% memory reduction |
| **VAD Filtering** | 4 | Skip silence, 20-30% faster |
| **ThreadPoolExecutor** | 3 | Concurrent processing |
| **Streaming (SSE)** | 5 | 88% better perceived latency |
| **Circuit Breaker** | 6 | Prevent cascade failures |
| **Graceful Shutdown** | 6 | Zero dropped requests |

---

## 🎯 Applying to Your Problem Statement

### For Audio/Speech Teams (PS06, PS08, PS12)
- Use Faster-Whisper patterns for your own audio models
- Streaming is critical for real-time transcription
- VAD preprocessing can significantly reduce processing time

### For LLM/RAG Teams (PS01, PS04)
- Same patterns apply: quantization, batching, streaming
- Replace Whisper with your LLM in the same architecture
- Backpressure is even more important for long-running LLM inference

### For Vision Teams (PS03, PS09, PS10, PS11)
- Batching matters more for vision (vs sequential audio)
- Consider ONNX Runtime instead of CTranslate2
- Streaming can show detection results progressively

---

## 📚 References

### Books
- **Designing Data-Intensive Applications** by Martin Kleppmann
  - Chapter 1: Reliability, Scalability, Maintainability
  - Chapter 4: Encoding and Evolution

### Real-World Case Studies
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - The runtime we use
- [Distil-Whisper](https://huggingface.co/distil-whisper) - Smaller, faster models
- [AssemblyAI Architecture](https://www.assemblyai.com/blog/) - Production speech-to-text

### Tools
- [Locust](https://locust.io/) - Load testing
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Inference optimization
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python API framework

---

## 🤝 Workshop Flow (For Presenter)

### 0-5 min: The Hook
- Show notebook working perfectly
- "Ship it?" 
- Run load test, watch it die

### 5-15 min: Understanding the Problem
- Explain blocking I/O
- Show queue buildup
- Introduce DDIA concepts

### 15-30 min: The Optimization Journey
- Stage 3: Faster-Whisper (live build)
- Stage 4: Show backpressure working
- Re-run load test, show improvement

### 30-45 min: Production Patterns
- Stage 5: Streaming demo (browser)
- Stage 6: Docker Compose, monitoring

### 45-55 min: The Numbers
- Show benchmark results
- Cost analysis
- Connect to their Problem Statements

### 55-60 min: Handoff
- Share repo link
- Q&A
- "Now apply this to YOUR model"

---

## 🏆 What You'll Take Away

1. **A working reference implementation** - Clone and adapt
2. **Benchmark methodology** - How to measure your own system
3. **Production patterns** - Circuit breaker, backpressure, streaming
4. **The confidence** - To ship your model to real users

---

## 📝 License

MIT License - Use freely for your projects.

---

**Built for the AI Grand Challenge Workshop @ IIT Delhi**

*Questions? Open an issue or reach out during the session.*
