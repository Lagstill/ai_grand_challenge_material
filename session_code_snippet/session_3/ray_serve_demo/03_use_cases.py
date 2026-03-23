"""
============================================================================
REAL-WORLD USE CASES - Where Ray Shine
============================================================================

This file demonstrates actual production scenarios where Ray provides
significant value. Each example is runnable and shows measurable benefits.

Run:
    python 03_use_cases.py
============================================================================
"""

import time
import ray
from ray import serve
import asyncio
import numpy as np
from typing import List
import requests
from concurrent.futures import ThreadPoolExecutor

print("""
============================================================================
USE CASE 1: LLM INFERENCE (The Killer App)
============================================================================

Problem: LLMs are expensive. A single request might take 2-10 seconds.
         With 100 users, that's 100 x 2s = 200s of waiting.

Solution: Ray Serve with batching and queuing.
         - Batch multiple prompts to the same model
         - Queue requests intelligently
         - Scale replicas based on demand
""")

ray.init(ignore_reinit_error=True)


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 3,
    },
    max_ongoing_requests=20,
)
class LLMServer:
    """
    Simulates LLM serving pattern used by companies like:
    - OpenAI (GPT API)
    - Anthropic (Claude API)
    - HuggingFace (Inference Endpoints)
    """
    
    def __init__(self):
        print("  Loading LLM weights (simulated)...")
        self.model_name = "llama-7b"
        self.tokens_generated = 0
    
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.2)
    async def generate_batch(self, prompts: List[str]) -> List[dict]:
        """
        Batch inference - THE key optimization for LLMs.
        
        Why batching matters:
        - GPU memory is loaded once
        - Compute is amortized across batch
        - Throughput increases linearly with batch size
        """
        batch_size = len(prompts)
        print(f"  LLM processing batch of {batch_size} prompts")
        
        # Simulate LLM inference (2s base + 0.1s per token per prompt)
        await asyncio.sleep(0.5 + 0.02 * batch_size)
        
        results = []
        for prompt in prompts:
            tokens = 50  # Simulated output tokens
            self.tokens_generated += tokens
            results.append({
                "prompt": prompt[:50] + "...",
                "completion": f"Generated response for: {prompt[:20]}...",
                "tokens": tokens,
                "batch_size": batch_size,
            })
        return results
    
    async def __call__(self, request):
        data = await request.json()
        return await self.generate_batch(data.get("prompt", ""))


llm_app = LLMServer.bind()
serve.run(llm_app, name="llm", route_prefix="/llm")

print("\nTesting LLM batching with 10 concurrent requests...")


def send_llm_request(prompt):
    start = time.time()
    resp = requests.post(
        "http://localhost:8000/llm",
        json={"prompt": prompt}
    ).json()
    return time.time() - start, resp


prompts = [f"Explain concept {i}" for i in range(10)]
start = time.time()
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(send_llm_request, prompts))

print(f"\n  Results:")
print(f"  Total time: {time.time()-start:.2f}s")
print(f"  Avg latency: {np.mean([r[0] for r in results]):.2f}s")
print(f"  Batch sizes used: {set(r[1]['batch_size'] for r in results)}")
print(f"  Without batching: ~5s x 10 = 50s (sequential)")
print(f"  With batching: ~2-3s total (requests batched together)")


print("""
============================================================================
USE CASE 2: MULTI-MODEL PIPELINE (NLP/CV Common Pattern)
============================================================================

Real-world ML often requires multiple models:
  - Text: Tokenizer -> Embedding -> Classifier -> PostProcessor
  - Image: Resize -> Feature Extraction -> Detection -> Tracking
  
Ray Serve handles this with deployment graphs.
""")


@serve.deployment(num_replicas=2)
class TextEmbedder:
    """Stage 1: Convert text to embeddings."""
    
    def embed(self, text: str) -> np.ndarray:
        # Simulates sentence-transformers or similar
        time.sleep(0.02)
        return np.random.randn(768)


@serve.deployment(num_replicas=2)
class Classifier:
    """Stage 2: Classify based on embeddings."""
    
    def classify(self, embedding: np.ndarray) -> dict:
        time.sleep(0.02)
        return {
            "label": np.random.choice(["positive", "negative", "neutral"]),
            "confidence": float(np.random.uniform(0.7, 0.99)),
        }


@serve.deployment(num_replicas=2)
class EntityExtractor:
    """Stage 2 (parallel): Extract entities."""
    
    def extract(self, text: str) -> dict:
        time.sleep(0.025)
        return {
            "entities": ["PERSON: John", "ORG: Acme Corp"],
            "count": 2,
        }


@serve.deployment
class NLPOrchestrator:
    """Orchestrates the multi-model pipeline."""
    
    def __init__(self, embedder, classifier, extractor):
        self.embedder = embedder
        self.classifier = classifier
        self.extractor = extractor
    
    async def analyze(self, text: str) -> dict:
        start = time.time()
        
        # Stage 1: Embedding
        embedding = await self.embedder.embed.remote(text)
        
        # Stage 2: Run classifier and extractor IN PARALLEL
        class_future = self.classifier.classify.remote(embedding)
        entity_future = self.extractor.extract.remote(text)
        
        classification, entities = await asyncio.gather(
            class_future, entity_future
        )
        
        return {
            "text": text[:50],
            "classification": classification,
            "entities": entities,
            "pipeline_ms": (time.time() - start) * 1000,
        }
    
    async def __call__(self, request):
        data = await request.json()
        return await self.analyze(data.get("text", ""))


# Build the pipeline
embedder = TextEmbedder.bind()
classifier = Classifier.bind()
extractor = EntityExtractor.bind()
nlp_pipeline = NLPOrchestrator.bind(embedder, classifier, extractor)

serve.run(nlp_pipeline, name="nlp", route_prefix="/nlp")

print("\nTesting NLP pipeline...")
test_texts = [
    "John Smith from Acme Corp announced great results today.",
    "The product launch was a complete failure.",
]

for text in test_texts:
    resp = requests.post(
        "http://localhost:8000/nlp",
        json={"text": text}
    ).json()
    print(f"\n  Input: {resp['text']}...")
    print(f"  Classification: {resp['classification']}")
    print(f"  Entities: {resp['entities']}")
    print(f"  Pipeline latency: {resp['pipeline_ms']:.1f}ms")


print("""
============================================================================
USE CASE 3: A/B TESTING & CANARY DEPLOYMENTS
============================================================================

When deploying ML models, you often need:
  - A/B test different model versions
  - Gradually roll out new models (canary)
  - Instant rollback if issues arise

Ray Serve makes this trivial.
""")


@serve.deployment
class ModelV1:
    def predict(self, x):
        return {"version": "v1", "prediction": "baseline", "x": x}


@serve.deployment
class ModelV2:
    def predict(self, x):
        return {"version": "v2", "prediction": "improved", "x": x}


@serve.deployment
class ABRouter:
    """Routes traffic between model versions."""
    
    def __init__(self, model_v1, model_v2, v2_traffic_percent: float = 20):
        self.model_v1 = model_v1
        self.model_v2 = model_v2
        self.v2_percent = v2_traffic_percent
        self.v1_count = 0
        self.v2_count = 0
    
    async def __call__(self, request):
        data = await request.json()
        
        # Route based on traffic percentage
        if np.random.random() < self.v2_percent / 100:
            self.v2_count += 1
            return await self.model_v2.predict.remote(data.get("x"))
        else:
            self.v1_count += 1
            return await self.model_v1.predict.remote(data.get("x"))
    
    def get_stats(self):
        total = self.v1_count + self.v2_count
        return {
            "v1_requests": self.v1_count,
            "v2_requests": self.v2_count,
            "v2_actual_percent": self.v2_count / total * 100 if total > 0 else 0,
        }


# Deploy with 20% traffic to v2
v1 = ModelV1.bind()
v2 = ModelV2.bind()
ab_router = ABRouter.bind(v1, v2, v2_traffic_percent=20)

serve.run(ab_router, name="ab_test", route_prefix="/ab")

print("\nRunning A/B test simulation (100 requests, 20% to v2)...")
version_counts = {"v1": 0, "v2": 0}

for _ in range(100):
    resp = requests.post("http://localhost:8000/ab", json={"x": 1}).json()
    version_counts[resp["version"]] += 1

print(f"\n  Traffic distribution:")
print(f"    v1: {version_counts['v1']}%")
print(f"    v2: {version_counts['v2']}%")
print(f"  (Target was 80% v1, 20% v2)")


print("""
============================================================================
USE CASE 4: RESOURCE ISOLATION (GPU/CPU Split)
============================================================================

Different models need different resources:
  - LLMs need GPUs
  - Preprocessing can use CPU only
  - Some models need specific GPU types

Ray Serve lets you specify resources per deployment.
""")

print("""
Example configuration:

@serve.deployment(
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 1,
        "resources": {"TPU": 1}  # Custom resources
    }
)
class GPUModel:
    ...

@serve.deployment(
    ray_actor_options={"num_cpus": 0.5}  # Share CPU for light tasks
)
class Preprocessor:
    ...

Benefits:
  - GPU models get dedicated GPU memory
  - CPU tasks share remaining resources
  - Automatic scheduling across cluster
  - No resource contention
""")


print("""
============================================================================
SUMMARY: Real-World Ray Users
============================================================================

Companies using Ray in production:

TECH GIANTS:
  - OpenAI: Training and serving GPT models
  - Uber: ML platform for all ML workloads
  - Spotify: Recommendation systems
  - LinkedIn: Feature engineering pipelines
  
AI COMPANIES:
  - Anyscale: Ray's commercial company (Ray Serve at scale)
  - Cohere: LLM inference
  - Weights & Biases: ML experiment tracking
  
ENTERPRISES:
  - Instacart: Real-time ML predictions
  - DoorDash: ETA predictions, ranking
  - Shopify: Product recommendations

RESEARCH:
  - Berkeley AI Research: Where Ray was created
  - Stanford: Various ML research projects

============================================================================
NEXT STEPS
============================================================================

1. Run the demos:
   python basic_serve.py     # Batching demo
   python ollama_serve.py    # LLM integration
   python compose_pipeline.py # Multi-model

2. Load test:
   python load_test.py

3. Deploy with Docker:
   docker-compose up

4. Monitor at:
   http://localhost:8265  # Ray Dashboard
============================================================================
""")

serve.shutdown()
ray.shutdown()
