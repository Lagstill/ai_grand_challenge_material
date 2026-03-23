"""
Deploy all Ray Serve applications.

Run this to start all demos:
    python deploy_all.py
"""

import os
import ray
from ray import serve

# Get Ollama host from environment (for Docker)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost:11434")


def deploy_all():
    """Deploy all Ray Serve applications."""
    
    # Initialize Ray
    ray_address = os.environ.get("RAY_HEAD_ADDRESS")
    if ray_address:
        ray.init(address=f"ray://{ray_address}", ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)
    
    print("\n" + "="*60)
    print("Deploying Ray Serve Applications")
    print("="*60)
    
    # 1. Deploy batched model server
    from basic_serve import BatchedModelServer
    batched_app = BatchedModelServer.bind()
    serve.run(batched_app, name="batched", route_prefix="/predict")
    print("  [OK] Batched inference at /predict")
    
    # 2. Deploy NLP pipeline
    from composed_pipeline import create_pipeline
    pipeline_app = create_pipeline()
    serve.run(pipeline_app, name="nlp", route_prefix="/nlp")
    print("  [OK] NLP pipeline at /nlp")
    
    # 3. Deploy Ollama serving
    from ollama_serve import OllamaServe
    ollama_app = OllamaServe.bind(
        ollama_url=f"http://{OLLAMA_HOST}",
        model="llama2"
    )
    serve.run(ollama_app, name="ollama", route_prefix="/v1")
    print("  [OK] Ollama at /v1")
    
    print("\n" + "="*60)
    print("All deployments running!")
    print("="*60)
    print("""
Endpoints:
  - Batched inference: http://localhost:8000/predict
  - NLP pipeline:      http://localhost:8000/nlp  
  - Ollama LLM:        http://localhost:8000/v1
  - Ray Dashboard:     http://localhost:8265

Example requests:
  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "hello"}'
  curl -X POST http://localhost:8000/nlp -H "Content-Type: application/json" -d '{"text": "Ray is great!"}'
  curl -X POST http://localhost:8000/v1 -H "Content-Type: application/json" -d '{"prompt": "Hello!"}'
    """)


if __name__ == "__main__":
    deploy_all()
    
    import time
    print("\nPress Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve.shutdown()
        ray.shutdown()
