"""
Ray Serve Basic Deployment with Dynamic Batching

Demonstrates:
- @serve.batch decorator for automatic request batching
- Autoscaling configuration
- Async request handling
- Performance metrics
"""

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
import asyncio
import time
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@serve.deployment(
    # Autoscaling configuration
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
        "upscale_delay_s": 10,
        "downscale_delay_s": 30,
    },
    # Resource allocation per replica
    ray_actor_options={"num_cpus": 0.5},
    # Maximum concurrent requests per replica
    max_ongoing_requests=100,
)
class BatchedModelServer:
    """
    Demonstrates efficient batching for ML inference.
    Simulates a model that benefits from batch processing (like GPU inference).
    """
    
    def __init__(self):
        logger.info("Initializing BatchedModelServer")
        self.request_count = 0
        self.batch_sizes = []
        
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def batched_predict(self, inputs: List[str]) -> List[dict]:
        """
        Process multiple requests in a single batch.
        
        In real ML scenarios, this enables:
        - Efficient GPU utilization
        - Reduced per-request overhead
        - Higher throughput
        """
        batch_size = len(inputs)
        self.batch_sizes.append(batch_size)
        self.request_count += batch_size
        
        logger.info(f"Processing batch of size {batch_size}")
        
        # Simulate batch inference (GPU operations benefit from batching)
        # Real scenario: model.predict(np.stack(inputs))
        await asyncio.sleep(0.05)  # Simulate processing time
        
        results = []
        for i, text in enumerate(inputs):
            results.append({
                "input": text,
                "embedding": np.random.randn(768).tolist()[:5],  # Truncated for demo
                "batch_size": batch_size,
                "batch_position": i,
                "total_processed": self.request_count,
            })
        
        return results
    
    async def __call__(self, request):
        """Handle HTTP requests."""
        data = await request.json()
        text = data.get("text", "default input")
        return await self.batched_predict(text)
    
    async def get_stats(self) -> dict:
        """Return serving statistics."""
        return {
            "total_requests": self.request_count,
            "batch_count": len(self.batch_sizes),
            "avg_batch_size": np.mean(self.batch_sizes) if self.batch_sizes else 0,
            "max_batch_size": max(self.batch_sizes) if self.batch_sizes else 0,
        }


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.25},
)
class StreamingServer:
    """
    Demonstrates streaming responses for LLM-style outputs.
    """
    
    async def generate_stream(self, prompt: str):
        """Simulate token-by-token generation."""
        words = f"Response to: {prompt}. This is a streaming demo.".split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.1)
    
    async def __call__(self, request):
        from starlette.responses import StreamingResponse
        data = await request.json()
        prompt = data.get("prompt", "Hello")
        return StreamingResponse(
            self.generate_stream(prompt),
            media_type="text/plain"
        )


# Application setup
app_batched = BatchedModelServer.bind()
app_streaming = StreamingServer.bind()


def deploy_and_test():
    """Deploy and run basic tests."""
    ray.init(ignore_reinit_error=True)
    
    # Deploy
    serve.run(app_batched, name="batched", route_prefix="/predict")
    serve.run(app_streaming, name="streaming", route_prefix="/stream")
    
    print("\n" + "="*60)
    print("Ray Serve deployments running:")
    print("  - Batched inference: http://localhost:8000/predict")
    print("  - Streaming:         http://localhost:8000/stream")
    print("  - Dashboard:         http://localhost:8265")
    print("="*60 + "\n")
    
    # Test batching efficiency
    import requests
    import concurrent.futures
    
    print("Testing batching efficiency...")
    
    def send_request(i):
        start = time.time()
        resp = requests.post(
            "http://localhost:8000/predict",
            json={"text": f"Request {i}"}
        )
        return time.time() - start, resp.json()
    
    # Send concurrent requests to demonstrate batching
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(send_request, i) for i in range(50)]
        results = [f.result() for f in futures]
    
    latencies = [r[0] for r in results]
    batch_sizes = [r[1].get("batch_size", 1) for r in results]
    
    print(f"\nResults from 50 concurrent requests:")
    print(f"  Average latency: {np.mean(latencies)*1000:.2f}ms")
    print(f"  P50 latency:     {np.percentile(latencies, 50)*1000:.2f}ms")
    print(f"  P99 latency:     {np.percentile(latencies, 99)*1000:.2f}ms")
    print(f"  Batch sizes:     {set(batch_sizes)}")
    print(f"  Avg batch size:  {np.mean(batch_sizes):.1f}")
    
    return serve


if __name__ == "__main__":
    serve_instance = deploy_and_test()
    print("\nPress Ctrl+C to stop serving...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve.shutdown()
        ray.shutdown()
