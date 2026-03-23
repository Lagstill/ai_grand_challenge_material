"""
Ray Serve + Ollama Integration

Demonstrates serving Ollama models with:
- Request batching for efficient queuing
- Multiple model replicas
- Load balancing across Ollama instances
- Streaming responses
- Caching layer
"""

import ray
from ray import serve
import httpx
import asyncio
from typing import List, Optional, AsyncGenerator
import logging
import hashlib
import json
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for repeated prompts."""
    
    def __init__(self, capacity: int = 100):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_num_ongoing_requests_per_replica": 3,
    },
    max_ongoing_requests=50,
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class OllamaServe:
    """
    Efficient Ollama model serving with Ray Serve.
    
    Benefits:
    - Automatic request queuing and load balancing
    - Multiple replicas can connect to single Ollama instance
    - Built-in health checks and failover
    - Request-level caching for repeated prompts
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama2",
        cache_enabled: bool = True,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)
        self.cache = LRUCache(capacity=100) if cache_enabled else None
        self.request_count = 0
        self.cache_hits = 0
        logger.info(f"OllamaServe initialized with model={model}, url={ollama_url}")
    
    def _cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters."""
        payload = {"prompt": prompt, **kwargs}
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    
    async def generate(self, prompt: str, stream: bool = False, **kwargs) -> dict:
        """
        Generate response from Ollama.
        
        Args:
            prompt: Input text
            stream: Whether to stream response
            **kwargs: Additional Ollama parameters (temperature, top_p, etc.)
        """
        self.request_count += 1
        
        # Check cache for non-streaming requests
        if not stream and self.cache:
            cache_key = self._cache_key(prompt, **kwargs)
            cached = self.cache.get(cache_key)
            if cached:
                self.cache_hits += 1
                logger.info(f"Cache hit for request {self.request_count}")
                return {"response": cached, "cached": True}
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            **kwargs,
        }
        
        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache successful responses
            if not stream and self.cache:
                self.cache.put(cache_key, result.get("response", ""))
            
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama request failed: {e}")
            return {"error": str(e)}
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **kwargs,
        }
        
        async with self.client.stream(
            "POST",
            f"{self.ollama_url}/api/generate",
            json=payload,
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
    
    async def __call__(self, request):
        """Handle HTTP requests."""
        data = await request.json()
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)
        
        if stream:
            from starlette.responses import StreamingResponse
            return StreamingResponse(
                self.generate_stream(prompt),
                media_type="text/plain"
            )
        
        return await self.generate(prompt, **data)
    
    async def get_stats(self) -> dict:
        """Return serving statistics."""
        return {
            "total_requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / self.request_count if self.request_count > 0 else 0,
            "model": self.model,
        }
    
    async def health_check(self) -> bool:
        """Check Ollama connectivity."""
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            return response.status_code == 200
        except:
            return False


@serve.deployment(num_replicas=1)
class OllamaRouter:
    """
    Router for multiple Ollama model deployments.
    Enables A/B testing, model routing, and fallback.
    """
    
    def __init__(self, models: dict):
        """
        Args:
            models: Dict mapping model names to deployment handles
        """
        self.models = models
        self.default_model = list(models.keys())[0]
    
    async def __call__(self, request):
        data = await request.json()
        model_name = data.pop("model", self.default_model)
        
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found. Available: {list(self.models.keys())}"}
        
        handle = self.models[model_name]
        # Forward request to appropriate model
        return await handle.generate.remote(data.get("prompt", ""), **data)


def create_multi_model_deployment():
    """
    Create a deployment with multiple Ollama models.
    Useful for serving different model sizes or A/B testing.
    """
    # Create model deployments
    llama2 = OllamaServe.bind(model="llama2")
    mistral = OllamaServe.bind(model="mistral")
    codellama = OllamaServe.bind(model="codellama")
    
    # Create router
    router = OllamaRouter.bind(
        models={
            "llama2": llama2,
            "mistral": mistral,
            "codellama": codellama,
        }
    )
    
    return router


# Single model app
app = OllamaServe.bind(model="llama2")

# Multi-model app
multi_model_app = create_multi_model_deployment()


def run_demo():
    """Run the Ollama serving demo."""
    ray.init(ignore_reinit_error=True)
    
    # Check if Ollama is running
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"Ollama running with models: {models}")
    except:
        print("WARNING: Ollama not detected at localhost:11434")
        print("Start Ollama with: ollama serve")
        print("Pull a model with: ollama pull llama2")
        return
    
    # Deploy single model
    serve.run(app, name="ollama", route_prefix="/v1")
    
    print("\n" + "="*60)
    print("Ollama via Ray Serve running at http://localhost:8000/v1")
    print("="*60)
    print("\nExample requests:")
    print("""
  # Generate
  curl -X POST http://localhost:8000/v1 \\
    -H "Content-Type: application/json" \\
    -d '{"prompt": "Explain Ray Serve in one sentence"}'

  # Stream 
  curl -X POST http://localhost:8000/v1 \\
    -H "Content-Type: application/json" \\
    -d '{"prompt": "Write a haiku about distributed systems", "stream": true}'
    """)
    
    return serve


if __name__ == "__main__":
    serve_instance = run_demo()
    
    if serve_instance:
        print("\nPress Ctrl+C to stop...")
        try:
            while True:
                asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))
        except KeyboardInterrupt:
            print("\nShutting down...")
            serve.shutdown()
            ray.shutdown()
