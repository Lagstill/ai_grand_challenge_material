"""
Load Testing for Ray Serve with Locust

Run with:
    locust -f load_test.py --host=http://localhost:8000

Or headless:
    locust -f load_test.py --host=http://localhost:8000 \
           --users 100 --spawn-rate 10 --run-time 60s --headless
"""

from locust import HttpUser, task, between, events
import random
import time
import json


class RayServeUser(HttpUser):
    """Simulate users hitting Ray Serve endpoints."""
    
    wait_time = between(0.1, 0.5)  # Wait between requests
    
    # Sample prompts for testing
    prompts = [
        "Explain machine learning in simple terms",
        "What is the capital of France?",
        "Write a Python function to sort a list",
        "How does distributed computing work?",
        "Describe the benefits of containerization",
    ]
    
    texts_for_nlp = [
        "Ray Serve is great for serving ML models at scale!",
        "Python is a terrible language for performance",
        "Google and Microsoft are leading cloud providers",
        "This product exceeded my expectations",
        "Docker and Kubernetes are essential tools",
    ]
    
    @task(3)
    def test_batched_predict(self):
        """Test the batched prediction endpoint."""
        self.client.post(
            "/predict",
            json={"text": f"Request {random.randint(1, 1000)}"},
            name="/predict (batched)",
        )
    
    @task(2)
    def test_nlp_pipeline(self):
        """Test the NLP pipeline endpoint."""
        self.client.post(
            "/nlp",
            json={"text": random.choice(self.texts_for_nlp)},
            name="/nlp (pipeline)",
        )
    
    @task(1)
    def test_ollama(self):
        """Test Ollama endpoint (lower frequency due to latency)."""
        self.client.post(
            "/v1",
            json={
                "prompt": random.choice(self.prompts),
                "max_tokens": 50,
            },
            name="/v1 (ollama)",
            timeout=60,
        )


class HighThroughputUser(HttpUser):
    """User profile for throughput testing."""
    
    wait_time = between(0.01, 0.05)  # Minimal wait for max throughput
    
    @task
    def blast_requests(self):
        """Send requests as fast as possible."""
        self.client.post(
            "/predict",
            json={"text": f"Throughput test {time.time()}"},
            name="/predict (throughput)",
        )


class BatchingTestUser(HttpUser):
    """User profile to demonstrate batching efficiency."""
    
    wait_time = between(0, 0.01)  # Near-simultaneous requests
    
    @task
    def concurrent_batch(self):
        """Send concurrent requests to trigger batching."""
        self.client.post(
            "/predict",
            json={"text": f"Batch test {random.random()}"},
            name="/predict (batch-test)",
        )


# Statistics tracking
request_stats = {
    "total_requests": 0,
    "total_latency": 0,
    "batch_sizes": [],
}


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """Track request statistics."""
    request_stats["total_requests"] += 1
    request_stats["total_latency"] += response_time
    
    # Try to extract batch size from response
    if kwargs.get("response"):
        try:
            data = kwargs["response"].json()
            if "batch_size" in data:
                request_stats["batch_sizes"].append(data["batch_size"])
        except:
            pass


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Load Test Summary")
    print("="*60)
    print(f"Total requests: {request_stats['total_requests']}")
    
    if request_stats['total_requests'] > 0:
        avg_latency = request_stats['total_latency'] / request_stats['total_requests']
        print(f"Average latency: {avg_latency:.2f}ms")
    
    if request_stats['batch_sizes']:
        import statistics
        print(f"Batch size - mean: {statistics.mean(request_stats['batch_sizes']):.1f}, "
              f"max: {max(request_stats['batch_sizes'])}")
    print("="*60)


# Quick benchmark script (without Locust)
def quick_benchmark():
    """Run a quick benchmark without Locust."""
    import requests
    import concurrent.futures
    import statistics
    
    BASE_URL = "http://localhost:8000"
    
    def send_request(endpoint, payload):
        start = time.time()
        try:
            resp = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=30)
            return time.time() - start, resp.status_code, resp.json()
        except Exception as e:
            return time.time() - start, 0, {"error": str(e)}
    
    print("\n" + "="*60)
    print("Quick Benchmark")
    print("="*60)
    
    # Test batching efficiency
    n_requests = 100
    print(f"\nSending {n_requests} concurrent requests to /predict...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [
            executor.submit(send_request, "/predict", {"text": f"Request {i}"})
            for i in range(n_requests)
        ]
        results = [f.result() for f in futures]
    
    latencies = [r[0] * 1000 for r in results]  # Convert to ms
    successful = [r for r in results if r[1] == 200]
    batch_sizes = [r[2].get("batch_size", 1) for r in successful if "batch_size" in r[2]]
    
    print(f"\nResults:")
    print(f"  Successful requests: {len(successful)}/{n_requests}")
    print(f"  Latency (ms):")
    print(f"    Mean:   {statistics.mean(latencies):.2f}")
    print(f"    Median: {statistics.median(latencies):.2f}")
    print(f"    P95:    {sorted(latencies)[int(0.95*len(latencies))]:.2f}")
    print(f"    P99:    {sorted(latencies)[int(0.99*len(latencies))]:.2f}")
    
    if batch_sizes:
        print(f"  Batching:")
        print(f"    Avg batch size: {statistics.mean(batch_sizes):.1f}")
        print(f"    Max batch size: {max(batch_sizes)}")
    
    # Calculate throughput
    total_time = max(r[0] for r in results)
    throughput = n_requests / total_time
    print(f"  Throughput: {throughput:.1f} req/s")


if __name__ == "__main__":
    print("Run with Locust:")
    print("  locust -f load_test.py --host=http://localhost:8000")
    print("\nOr run quick benchmark:")
    quick_benchmark()
