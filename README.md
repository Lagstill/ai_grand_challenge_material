# AI Grand Challenge Material

**IIT Delhi AI Grand Challenge — Workshop Materials**

> Comprehensive hands-on materials for building production-ready AI/ML systems.

---

## 📚 Sessions Overview

| Session | Topic | Key Focus |
|---------|-------|-----------|
| **Session 1** | MLOps/LLMOps Best Practices | Reproducibility, experiment tracking, model registry |
| **Session 2** | Whisper Production Journey | Taking a model from notebook to production |
| **Session 3** | Ray Serve for Model Serving | Efficient, scalable model deployment |

---

## 🗂️ Repository Structure

```
├── presentation_deck/           # Slide decks for sessions
└── session_code_snippet/
    ├── session_1/
    │   └── mlops-demo/          # MLOps/LLMOps demo with W&B + Langfuse
    ├── session_2/
    │   └── whisper-production-journey/  # Notebook → Production workshop
    └── session_3/
        └── ray_serve_demo/      # Ray Serve model serving demo
```

---

## Session 1: MLOps/LLMOps Best Practices

**Problem**: Teams get LLMs working in notebooks but can't reproduce results, compare experiments, or safely deploy to production.

**Solution**: A complete MLOps stack:
- **Config-driven training** — `train_config.yaml` for reproducibility
- **Experiment tracking** — Weights & Biases for run comparison
- **Model registry** — W&B Model Registry with staging/production aliases
- **Eval gates** — GitHub Actions blocks PRs if metrics regress
- **Production observability** — Langfuse traces every inference

📁 [Session 1 Code](session_code_snippet/session_1/mlops-demo-final/mlops-demo/)

---

## Session 2: Whisper Production Journey

**Problem**: "Your model works in Jupyter. PM says ship it. What could go wrong?"

**Journey**: 6-stage progression from notebook to production-ready API:

| Metric | Stage 1 (Notebook) | Stage 6 (Production) | Improvement |
|--------|-------------------|----------------------|-------------|
| Requests/min | 50 | 400 | **8x** |
| p99 Latency | 25s | 1.8s | **14x** |
| Failure Rate | 30% | <0.5% | **60x** |

Stages: Naive API → Load Test Failures → Faster Whisper → Quantization → Streaming → Production

📁 [Session 2 Code](session_code_snippet/session_2/whisper-production-journey/)

---

## Session 3: Ray Serve for Model Serving

**Problem**: How do you serve ML models efficiently at scale?

**Solution**: Ray Serve with:
- **Dynamic batching** — 8-32x throughput improvement
- **Autoscaling** — 1-10 replicas based on load
- **Multi-model pipelines** — Parallel model execution
- **LLM serving** — Serve local models via Ollama

Learning path:
```bash
python 01_what_is_ray.py       # Ray basics
python 02_why_ray_serve.py     # Why Ray Serve
python 03_use_cases.py         # Real-world applications
```

📁 [Session 3 Code](session_code_snippet/session_3/ray_serve_demo/)

---

## 🚀 Getting Started

Each session has its own setup instructions. Navigate to the session folder and follow the README:

```bash
# Session 1 - MLOps
cd session_code_snippet/session_1/mlops-demo-final/mlops-demo
cat README.md

# Session 2 - Whisper
cd session_code_snippet/session_2/whisper-production-journey
cat README.md

# Session 3 - Ray Serve
cd session_code_snippet/session_3/ray_serve_demo
cat README.md
```

---

## 📋 Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Basic familiarity with ML/AI concepts
- (Session-specific requirements in each folder)

---

## 📝 License

Materials for IIT Delhi AI Grand Challenge Workshop.