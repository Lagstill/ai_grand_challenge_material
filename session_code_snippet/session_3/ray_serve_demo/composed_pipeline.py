"""
Ray Serve Composed Pipeline

Demonstrates deployment graphs for:
- Multi-stage ML pipelines
- Preprocessing + Model + Postprocessing chains
- Parallel model ensemble
- Conditional routing
"""

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
import asyncio
import time
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@serve.deployment(num_replicas=2)
class Preprocessor:
    """Text preprocessing: tokenization, cleaning, embedding lookup."""
    
    def __init__(self):
        logger.info("Preprocessor initialized")
        # Simulate loading vocabulary/tokenizer
        self.vocab_size = 50000
    
    async def preprocess(self, text: str) -> Dict:
        """Clean and tokenize text."""
        # Simulate preprocessing time
        await asyncio.sleep(0.01)
        
        tokens = text.lower().split()
        # Simulate token IDs
        token_ids = [hash(t) % self.vocab_size for t in tokens]
        
        return {
            "original": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "length": len(tokens),
        }


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1},
)
class SentimentModel:
    """Sentiment analysis model."""
    
    def __init__(self):
        logger.info("SentimentModel initialized")
        # Simulate loading model weights
        self.positive_words = {"good", "great", "excellent", "amazing", "love"}
        self.negative_words = {"bad", "terrible", "awful", "hate", "poor"}
    
    async def predict(self, preprocessed: Dict) -> Dict:
        """Run sentiment inference."""
        await asyncio.sleep(0.02)  # Simulate inference
        
        tokens = set(preprocessed["tokens"])
        pos_count = len(tokens & self.positive_words)
        neg_count = len(tokens & self.negative_words)
        
        if pos_count > neg_count:
            sentiment = "positive"
            confidence = 0.7 + (pos_count * 0.1)
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = 0.7 + (neg_count * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": min(confidence, 0.99),
            "pos_signals": pos_count,
            "neg_signals": neg_count,
        }


@serve.deployment(num_replicas=2)
class EntityExtractor:
    """Named entity recognition."""
    
    def __init__(self):
        logger.info("EntityExtractor initialized")
        # Simple entity patterns
        self.entities = {
            "ORG": {"google", "microsoft", "apple", "amazon", "ray"},
            "TECH": {"python", "kubernetes", "docker", "ml", "ai"},
        }
    
    async def extract(self, preprocessed: Dict) -> Dict:
        """Extract named entities."""
        await asyncio.sleep(0.015)
        
        tokens = set(preprocessed["tokens"])
        found = {}
        
        for entity_type, patterns in self.entities.items():
            matches = tokens & patterns
            if matches:
                found[entity_type] = list(matches)
        
        return {"entities": found, "count": sum(len(v) for v in found.values())}


@serve.deployment
class Postprocessor:
    """Combine results and format output."""
    
    async def combine(
        self,
        preprocessed: Dict,
        sentiment: Dict,
        entities: Dict,
    ) -> Dict:
        """Merge all model outputs."""
        return {
            "input": preprocessed["original"],
            "token_count": preprocessed["length"],
            "sentiment": sentiment,
            "entities": entities,
            "timestamp": time.time(),
        }


@serve.deployment
class NLPPipeline:
    """
    Composed NLP pipeline demonstrating Ray Serve deployment graph.
    
    Structure:
        Input -> Preprocessor -> [SentimentModel, EntityExtractor] -> Postprocessor -> Output
                                  (parallel execution)
    """
    
    def __init__(
        self,
        preprocessor: DeploymentHandle,
        sentiment_model: DeploymentHandle,
        entity_extractor: DeploymentHandle,
        postprocessor: DeploymentHandle,
    ):
        self.preprocessor = preprocessor
        self.sentiment_model = sentiment_model
        self.entity_extractor = entity_extractor
        self.postprocessor = postprocessor
    
    async def analyze(self, text: str) -> Dict:
        """Run full NLP pipeline."""
        start = time.time()
        
        # Step 1: Preprocess
        preprocessed = await self.preprocessor.preprocess.remote(text)
        
        # Step 2: Run models in parallel (key optimization)
        sentiment_future = self.sentiment_model.predict.remote(preprocessed)
        entity_future = self.entity_extractor.extract.remote(preprocessed)
        
        sentiment, entities = await asyncio.gather(
            sentiment_future,
            entity_future,
        )
        
        # Step 3: Postprocess
        result = await self.postprocessor.combine.remote(
            preprocessed, sentiment, entities
        )
        
        result["pipeline_latency_ms"] = (time.time() - start) * 1000
        return result
    
    async def __call__(self, request):
        data = await request.json()
        text = data.get("text", "")
        return await self.analyze(text)


# Create the deployment graph
def create_pipeline():
    """Build the NLP pipeline deployment graph."""
    preprocessor = Preprocessor.bind()
    sentiment_model = SentimentModel.bind()
    entity_extractor = EntityExtractor.bind()
    postprocessor = Postprocessor.bind()
    
    pipeline = NLPPipeline.bind(
        preprocessor=preprocessor,
        sentiment_model=sentiment_model,
        entity_extractor=entity_extractor,
        postprocessor=postprocessor,
    )
    
    return pipeline


app = create_pipeline()


def run_demo():
    """Run the composed pipeline demo."""
    ray.init(ignore_reinit_error=True)
    
    serve.run(app, name="nlp_pipeline", route_prefix="/nlp")
    
    print("\n" + "="*60)
    print("NLP Pipeline running at http://localhost:8000/nlp")
    print("="*60)
    
    # Test the pipeline
    import requests
    
    test_texts = [
        "Ray Serve is great for ML deployment!",
        "Python and Docker are excellent tools.",
        "This product is terrible and bad.",
        "Google and Microsoft use Kubernetes.",
    ]
    
    print("\nTesting pipeline:")
    for text in test_texts:
        resp = requests.post(
            "http://localhost:8000/nlp",
            json={"text": text}
        )
        result = resp.json()
        print(f"\n  Input: {text}")
        print(f"  Sentiment: {result['sentiment']['sentiment']} "
              f"(confidence: {result['sentiment']['confidence']:.2f})")
        print(f"  Entities: {result['entities']['entities']}")
        print(f"  Latency: {result['pipeline_latency_ms']:.2f}ms")
    
    return serve


if __name__ == "__main__":
    serve_instance = run_demo()
    
    print("\nPress Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve.shutdown()
        ray.shutdown()
