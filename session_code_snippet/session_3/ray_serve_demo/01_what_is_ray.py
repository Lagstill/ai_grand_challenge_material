"""
============================================================================
WHAT IS RAY? - A Hands-On Introduction
============================================================================

Ray is a framework for scaling Python applications from laptop to cluster.
Think of it as "multiprocessing on steroids" that works across machines.

This file demonstrates Ray's core concepts through runnable examples.

Run sections individually or the whole file:
    python 01_what_is_ray.py
============================================================================
"""

import time
import ray

print("""
============================================================================
PART 1: THE PROBLEM - Python is Single-Threaded
============================================================================

Python's GIL (Global Interpreter Lock) means only one thread runs at a time.
This is a major bottleneck for CPU-intensive and I/O-intensive tasks.

Let's see the problem:
""")


# ----- WITHOUT RAY: Sequential Processing -----

def slow_function(x):
    """Simulate a slow computation (e.g., ML inference, API call)."""
    time.sleep(1)  # Simulates 1 second of work
    return x * x


def run_sequential():
    """Process items one at a time - SLOW!"""
    start = time.time()
    results = [slow_function(i) for i in range(4)]
    duration = time.time() - start
    return results, duration


print("Running 4 tasks sequentially...")
results, duration = run_sequential()
print(f"  Results: {results}")
print(f"  Time: {duration:.2f}s (expected ~4s because 4 x 1s)")


print("""
============================================================================
PART 2: THE SOLUTION - Ray Parallelizes Everything
============================================================================

Ray lets you:
1. Run functions in parallel with @ray.remote
2. Distribute work across CPU cores (and machines!)
3. Share data efficiently between tasks

Let's parallelize the same work:
""")


# ----- WITH RAY: Parallel Processing -----

# Initialize Ray (only needs to be done once)
ray.init(ignore_reinit_error=True)


@ray.remote
def slow_function_parallel(x):
    """Same function, but now it can run in parallel!"""
    time.sleep(1)
    return x * x


def run_parallel():
    """Process items in parallel - FAST!"""
    start = time.time()
    
    # Launch all tasks at once (non-blocking)
    futures = [slow_function_parallel.remote(i) for i in range(4)]
    
    # Wait for all results
    results = ray.get(futures)
    
    duration = time.time() - start
    return results, duration


print("Running 4 tasks in parallel with Ray...")
results, duration = run_parallel()
print(f"  Results: {results}")
print(f"  Time: {duration:.2f}s (expected ~1s because all run simultaneously)")
print(f"\n  SPEEDUP: ~4x faster!")


print("""
============================================================================
PART 3: REAL-WORLD USE CASE - Batch ML Inference
============================================================================

Imagine you have an ML model and need to process 100 images.
Without Ray: Process one at a time (slow)
With Ray: Process all in parallel across all CPU cores

This is exactly what Ray Serve does for web APIs!
""")


@ray.remote
def process_image(image_id):
    """Simulate ML model inference on an image."""
    time.sleep(0.1)  # Simulates inference time
    return {"image_id": image_id, "prediction": "cat", "confidence": 0.95}


def batch_inference_demo():
    start = time.time()
    
    # Process 20 images in parallel
    futures = [process_image.remote(i) for i in range(20)]
    results = ray.get(futures)
    
    duration = time.time() - start
    print(f"  Processed 20 images in {duration:.2f}s")
    print(f"  Throughput: {20/duration:.1f} images/second")
    print(f"  Sample result: {results[0]}")


print("Running batch inference demo...")
batch_inference_demo()


print("""
============================================================================
PART 4: RAY ACTORS - Stateful Distributed Objects
============================================================================

Sometimes you need to maintain state (like a loaded ML model).
Ray Actors are classes that live in the cluster and maintain state.
""")


@ray.remote
class ModelServer:
    """A stateful actor that holds a model in memory."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.request_count = 0
        print(f"Loading model: {model_name}")
        time.sleep(0.5)  # Simulate model loading
        print(f"Model {model_name} ready!")
    
    def predict(self, input_data):
        self.request_count += 1
        time.sleep(0.05)  # Simulate inference
        return {
            "model": self.model_name,
            "input": input_data,
            "output": f"prediction_{self.request_count}",
        }
    
    def get_stats(self):
        return {"model": self.model_name, "requests_served": self.request_count}


# Create actor instances (they run in separate processes)
print("\nCreating model server actors...")
server1 = ModelServer.remote("sentiment-model")
server2 = ModelServer.remote("ner-model")

# Use the actors
print("\nSending requests to actors...")
results = ray.get([
    server1.predict.remote("I love Ray!"),
    server1.predict.remote("This is great"),
    server2.predict.remote("Apple Inc. in California"),
])

for r in results:
    print(f"  {r}")

# Check stats
stats = ray.get([server1.get_stats.remote(), server2.get_stats.remote()])
print(f"\nActor stats: {stats}")


print("""
============================================================================
SUMMARY: When to Use Ray
============================================================================

USE RAY WHEN YOU NEED TO:
  1. Parallelize CPU-intensive work (ML training, data processing)
  2. Scale beyond a single machine (distributed computing)
  3. Serve ML models with high throughput (Ray Serve)
  4. Build complex ML pipelines (Ray Tune, Ray Train)
  5. Process data in parallel (Ray Data)

DON'T USE RAY WHEN:
  - Your task is already fast enough
  - You're doing simple I/O (use asyncio instead)
  - You only have tiny amounts of data

NEXT STEPS:
  - Run 02_why_ray_serve.py to see how Ray Serve builds on these concepts
  - Run basic_serve.py to see production-ready model serving
============================================================================
""")

ray.shutdown()
