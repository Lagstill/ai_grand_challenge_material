"""
============================================================================
WHY RAY SERVE? - The Model Serving Problem
============================================================================

This file explains WHY Ray Serve exists and WHEN to use it.

The Problem: You trained an ML model. Now what?
- How do you make it accessible to users?
- How do you handle 1000 requests per second?
- How do you update the model without downtime?

Run:
    python 02_why_ray_serve.py
============================================================================
"""

import time
import asyncio
import ray
from ray import serve
import requests
from concurrent.futures import ThreadPoolExecutor

print("""
============================================================================
THE ML SERVING JOURNEY
============================================================================

Stage 1: "It works on my laptop!"
         - model.predict(input) in a Jupyter notebook
         
Stage 2: "Let's make it a REST API!"
         - Flask/FastAPI wrapper around the model
         - Problem: One process, one request at a time (or threading issues)
         
Stage 3: "We need to scale!"
         - Multiple workers, load balancer, Kubernetes...
         - Problem: Complex infrastructure, hard to manage
         
Stage 4: "Ray Serve handles all of this!"
         - Automatic scaling, batching, deployment management
         - From laptop to cluster with the same code
         
Let's see each stage in action:
""")


# ============================================================================
# STAGE 1: Direct Model Calls (The Baseline)
# ============================================================================

class SimpleModel:
    """Simulates an ML model."""
    
    def __init__(self):
        print("  Loading model weights...")
        time.sleep(0.5)
        print("  Model ready!")
    
    def predict(self, text):
        time.sleep(0.05)  # Simulate inference time
        return {"input": text, "sentiment": "positive", "score": 0.92}


print("\n--- STAGE 1: Direct Model Call ---")
model = SimpleModel()
result = model.predict("Ray is amazing!")
print(f"  Result: {result}")
print("  Problem: Not accessible over the network!")


# ============================================================================
# STAGE 2: Flask-Style API (Naive Approach)
# ============================================================================

print("""
--- STAGE 2: Flask-Style API ---

Typical Flask/FastAPI code:

    from flask import Flask, request
    app = Flask(__name__)
    model = load_model()  # Loaded once at startup
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        result = model.predict(data['text'])
        return result

Problems with this approach:
  1. Single-threaded: One request blocks others
  2. No batching: Each request = one model call (inefficient for GPUs)
  3. Scaling = manual: Need Gunicorn, workers, load balancers
  4. No autoscaling: Fixed number of workers
""")


# ============================================================================
# STAGE 3: The Scaling Nightmare
# ============================================================================

print("""
--- STAGE 3: Traditional Scaling (Without Ray) ---

To handle production traffic, you typically need:

  +------------------+
  |  Load Balancer   |  (HAProxy, Nginx, AWS ALB)
  +--------+---------+
           |
     +-----+-----+-----+
     |           |     |
  +--v---+  +---v--+  +v----+
  |Flask |  |Flask |  |Flask|   (Multiple replicas)
  |+Model|  |+Model|  |+Model|
  +------+  +------+  +------+
  
Infrastructure needed:
  - Kubernetes cluster
  - Container orchestration
  - Service mesh (Istio)
  - Monitoring (Prometheus, Grafana)
  - Auto-scaling rules (HPA)
  - Load balancer configuration
  
This is A LOT of infrastructure for "just serving a model"!
""")


# ============================================================================
# STAGE 4: Ray Serve (The Solution)
# ============================================================================

print("""
--- STAGE 4: Ray Serve (Elegant Solution) ---

Ray Serve provides:
  - Automatic scaling (no Kubernetes needed until very large scale)
  - Request batching (critical for GPU efficiency)
  - Model composition (chain multiple models)
  - Zero-downtime updates
  - Built-in load balancing

Let's see it in action:
""")

ray.init(ignore_reinit_error=True)


@serve.deployment(
    num_replicas=2,  # Start with 2 replicas
    ray_actor_options={"num_cpus": 0.25},
)
class SmartModel:
    """A Ray Serve deployment with built-in optimizations."""
    
    def __init__(self):
        print("  SmartModel: Loading...")
        time.sleep(0.3)
        self.request_count = 0
        print("  SmartModel: Ready!")
    
    async def __call__(self, request):
        """Handle HTTP request."""
        self.request_count += 1
        data = await request.json()
        await asyncio.sleep(0.03)  # Simulate inference
        return {
            "input": data.get("text"),
            "sentiment": "positive",
            "score": 0.95,
            "served_by": f"replica-{id(self) % 1000}",
            "total_served": self.request_count,
        }


# Deploy the model
print("\nDeploying model with Ray Serve...")
app = SmartModel.bind()
serve.run(app, name="smart_model", route_prefix="/smart")
print("Deployed at http://localhost:8000/smart")


# Test load balancing across replicas
print("\nTesting load balancing (10 concurrent requests):")


def send_request(i):
    return requests.post(
        "http://localhost:8000/smart",
        json={"text": f"Request {i}"}
    ).json()


with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(send_request, range(10)))

replicas_used = set(r["served_by"] for r in results)
print(f"  Requests distributed across {len(replicas_used)} replicas: {replicas_used}")


# ============================================================================
# WHY RAY SERVE SCALES BETTER
# ============================================================================

print("""
============================================================================
RAY SERVE ADVANTAGES DEMONSTRATED
============================================================================
""")


# ADVANTAGE 1: Automatic Batching
print("ADVANTAGE 1: Automatic Request Batching")
print("-" * 40)


@serve.deployment
class BatchedModel:
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def predict_batch(self, inputs):
        """Process multiple inputs at once - critical for GPU efficiency!"""
        batch_size = len(inputs)
        print(f"  Processing batch of {batch_size} items together")
        await asyncio.sleep(0.05)  # Same time regardless of batch size!
        return [{"input": inp, "batch_size": batch_size} for inp in inputs]
    
    async def __call__(self, request):
        data = await request.json()
        return await self.predict_batch(data.get("text", ""))


batched_app = BatchedModel.bind()
serve.run(batched_app, name="batched", route_prefix="/batch")

print("\nSending 16 concurrent requests to batched endpoint...")


def send_batch_request(i):
    return requests.post(
        "http://localhost:8000/batch",
        json={"text": f"Input {i}"}
    ).json()


start = time.time()
with ThreadPoolExecutor(max_workers=16) as executor:
    batch_results = list(executor.map(send_batch_request, range(16)))

print(f"  Total time: {time.time()-start:.2f}s")
batch_sizes = [r["batch_size"] for r in batch_results]
print(f"  Batch sizes used: {sorted(set(batch_sizes))}")
print("  Without batching: 16 x 50ms = 800ms")
print("  With batching: ~2 batches x 50ms = 100ms (8x faster!)")


# ADVANTAGE 2: Dynamic Scaling
print("\n\nADVANTAGE 2: Autoscaling Configuration")
print("-" * 40)
print("""
@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,        # Scale to 1 when idle
        "max_replicas": 20,       # Scale up to 20 under load
        "target_num_ongoing_requests_per_replica": 5,
    }
)
class AutoscaledModel:
    ...

Ray Serve automatically:
  - Monitors request queue depth
  - Adds replicas when load increases
  - Removes replicas when load decreases
  - No manual intervention needed!
""")


# ADVANTAGE 3: Model Composition
print("\nADVANTAGE 3: Model Composition")
print("-" * 40)
print("""
Chain multiple models in a pipeline:

    @serve.deployment
    class Pipeline:
        def __init__(self, preprocessor, model_a, model_b, postprocessor):
            self.preprocessor = preprocessor
            self.model_a = model_a
            self.model_b = model_b
            self.postprocessor = postprocessor
        
        async def __call__(self, request):
            # Sequential
            data = await self.preprocessor.process.remote(request)
            
            # Parallel (both models run simultaneously!)
            result_a, result_b = await asyncio.gather(
                self.model_a.predict.remote(data),
                self.model_b.predict.remote(data),
            )
            
            return await self.postprocessor.combine.remote(result_a, result_b)

See composed_pipeline.py for a full working example!
""")


# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("""
============================================================================
COMPARISON: Flask vs Ray Serve
============================================================================

Feature              | Flask/FastAPI        | Ray Serve
---------------------|----------------------|------------------------
Scaling              | Manual (Gunicorn)    | Automatic
Batching             | Must implement       | @serve.batch decorator
Load Balancing       | External (Nginx)     | Built-in
GPU Efficiency       | One request/GPU      | Batched requests
Model Composition    | Manual routing       | Deployment graphs
Zero-downtime Deploy | Complex              | serve.run() again
Monitoring           | Add Prometheus       | Built-in dashboard
From Laptop to Cloud | Rewrite needed       | Same code!

============================================================================
WHEN TO USE RAY SERVE
============================================================================

PERFECT FOR:
  - ML model serving at scale
  - LLM inference (batching is critical)
  - Multi-model pipelines (NLP, vision)
  - Dynamic scaling workloads
  - GPU-intensive inference

OVERKILL FOR:
  - Simple CRUD APIs (use FastAPI)
  - Static content (use CDN)
  - <10 requests/second (Flask is fine)

NEXT: Run basic_serve.py and load_test.py to see it handle real load!
============================================================================
""")

# Cleanup
serve.shutdown()
ray.shutdown()
