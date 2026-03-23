# Ray Serve Efficient Model Serving Demo

This demo showcases Ray Serve's capabilities for efficient, production-grade model serving.

## Learning Path (Start Here!)

If you're new to Ray, follow these files in order:

| Step | File | What You'll Learn |
|------|------|-------------------|
| 1 | `01_what_is_ray.py` | What Ray is, why Python needs it, parallelism basics |
| 2 | `02_why_ray_serve.py` | Why Ray Serve exists, comparison with Flask, scaling |
| 3 | `03_use_cases.py` | Real-world scenarios: LLMs, pipelines, A/B testing |
| 4 | `basic_serve.py` | Production-ready batching and autoscaling |
| 5 | `ollama_serve.py` | Serve your local LLMs with Ollama |

```bash
# Run the learning path
python 01_what_is_ray.py      # 2 min - understand Ray basics
python 02_why_ray_serve.py    # 3 min - understand Ray Serve
python 03_use_cases.py        # 3 min - see real applications
```

## Key Features Demonstrated

| Feature | File | Benefit |
|---------|------|---------|
| **Dynamic Batching** | `basic_serve.py` | 8-32x throughput for ML inference |
| **Autoscaling** | `basic_serve.py` | 1-10 replicas based on load |
| **Multi-Model Pipelines** | `composed_pipeline.py` | Parallel model execution |
| **LLM Serving** | `ollama_serve.py` | Serve Ollama models efficiently |
| **Load Testing** | `load_test.py` | Benchmark with Locust |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start with Docker (Ollama + Ray)
docker-compose up -d

# Or run locally
python basic_serve.py

# Load test
locust -f load_test.py --host=http://localhost:8000
```

## Architecture

```
                    +------------------+
                    |   Ray Serve      |
                    |   (Autoscaling)  |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
   +-----v-----+      +------v------+     +------v------+
   | Batched   |      |   Ollama    |     |  Composed   |
   | Inference |      | Integration |     |   Pipeline  |
   +-----------+      +-------------+     +-------------+
```

## Files

**Learning (Start Here):**
- `01_what_is_ray.py` - Introduction to Ray concepts
- `02_why_ray_serve.py` - Why Ray Serve for ML serving
- `03_use_cases.py` - Real-world production patterns

**Production Examples:**
- `basic_serve.py` - Batching and autoscaling demo
- `ollama_serve.py` - Ollama LLM serving with Ray
- `composed_pipeline.py` - Multi-model deployment graph
- `load_test.py` - Locust load testing script
- `deploy_all.py` - Deploy all services at once
- `docker-compose.yml` - Full stack deployment
