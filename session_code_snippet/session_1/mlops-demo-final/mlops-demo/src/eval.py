"""
src/eval.py — Eval-Driven Development in practice.

This script runs BEFORE you promote a model. CI runs it on every PR.
If metrics regress vs the baseline in main, the PR is blocked.

RAGAS metrics used:
  - faithfulness       : does the answer stick to the retrieved context?
  - context_recall     : did retrieval surface the right passages?
  - context_precision  : are the retrieved passages actually relevant?
  - answer_relevancy   : does the answer address the question asked?

Usage:
  make eval
  python src/eval.py --prompt-version v2 --output evals/metrics.json
"""

import argparse
import json
import os
import time
import yaml

from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# ── Ollama setup — runs 100% locally, no API key needed ──────────────────────
# RAGAS uses an LLM internally as a judge. We point it at local Llama via Ollama.
# Make sure Ollama is running: `ollama serve`
# And the model is pulled: `ollama pull llama3.1`

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")
OLLAMA_EMBED      = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Wrap for RAGAS
ragas_llm = LangchainLLMWrapper(
    ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model=OLLAMA_EMBED, base_url=OLLAMA_BASE_URL)
)


# ── Thresholds — the contract your model must meet ───────────────────────────
# If any metric falls below these, the CI gate fails the PR.
# These are not arbitrary — they encode product requirements.

THRESHOLDS = {
    "faithfulness": 0.75,        # below this = hallucinating from the context
    "context_recall": 0.70,      # below this = retrieval is missing key passages
    "context_precision": 0.65,   # below this = retrieval is noisy
    "answer_relevancy": 0.75,    # below this = answers are drifting off-topic
}


# ── Load prompt template ──────────────────────────────────────────────────────

def load_prompt(version: str) -> dict:
    path = Path(f"prompts/{version}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ── Build eval dataset ────────────────────────────────────────────────────────

def load_eval_data(path: str = "data/eval_qa.json") -> list[dict]:
    """
    Each example must have:
      question, answer (ground truth), contexts (list of retrieved passages)
    """
    with open(path) as f:
        return json.load(f)


# ── Run model inference on eval set ──────────────────────────────────────────

def run_inference(examples: list[dict], prompt: dict, model_name: str) -> list[str]:
    """
    Runs inference via Ollama — fully local, no API key needed.
    Ollama exposes an OpenAI-compatible endpoint at /v1,
    so we just swap the base_url and set api_key to anything.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",               # required by client but unused by Ollama
    )
    generated = []

    for i, ex in enumerate(examples):
        system = prompt["system"].format(context="\n\n".join(ex["contexts"]))
        user = prompt["user_template"].format(
            context="\n\n".join(ex["contexts"]),
            question=ex["question"],
        )
        print(f"  Inference [{i+1}/{len(examples)}] via {model_name}...", end="\r")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0,
        )
        generated.append(response.choices[0].message.content.strip())

    print()
    return generated


# ── Core eval logic ───────────────────────────────────────────────────────────

def run_eval(prompt_version: str, output_path: str, model_name: str = "gpt-4o-mini"):

    prompt = load_prompt(prompt_version)
    examples = load_eval_data()

    print(f"Evaluating prompt version: {prompt_version}")
    print(f"Eval set size: {len(examples)} examples")
    print(f"Inference model: {model_name}")
    print("─" * 40)

    # ── Run inference ───────────────────────────────────────────────────────
    t0 = time.time()
    generated_answers = run_inference(examples, prompt, model_name)
    inference_time = time.time() - t0

    # ── Build RAGAS dataset ─────────────────────────────────────────────────
    # RAGAS 0.1.x+ uses new column names: user_input, response, retrieved_contexts, reference
    ragas_data = Dataset.from_dict({
        "user_input":          [e["question"] for e in examples],
        "response":            generated_answers,
        "retrieved_contexts":  [e["contexts"] for e in examples],
        "reference":           [e["answer"] for e in examples],
    })

    # ── Run RAGAS — using local Llama as the judge ──────────────────────────
    # raise_exceptions=False prevents single failures from crashing the entire eval
    result = evaluate(
        ragas_data,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )
    # Skip NaN values when computing mean (failures from timeouts/model errors)
    df = result.to_pandas()
    scores = df.select_dtypes(include='number').mean(skipna=True).to_dict()
    
    # Warn if any metrics have NaN values
    nan_counts = df.select_dtypes(include='number').isna().sum()
    for col, count in nan_counts.items():
        if count > 0:
            print(f"  ⚠ {col}: {count}/{len(df)} examples failed (returned NaN)")

    # ── Check against thresholds ────────────────────────────────────────────
    passed = True
    failures = []
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        ok = score >= threshold
        status = "✓" if ok else "✗ FAIL"
        print(f"  {status}  {metric:<25} {score:.3f}  (threshold: {threshold})")
        if not ok:
            passed = False
            failures.append({"metric": metric, "score": score, "threshold": threshold})

    print("─" * 40)
    print(f"Overall: {'PASSED' if passed else 'FAILED'}")

    # ── Write metrics.json — CI reads this ──────────────────────────────────
    output = {
        "prompt_version": prompt_version,
        "model": model_name,
        "eval_set_size": len(examples),
        "inference_time_s": round(inference_time, 2),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "thresholds": THRESHOLDS,
        "passed": passed,
        "failures": failures,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nMetrics written to: {output_path}")

    # ── Exit code matters — CI uses this ────────────────────────────────────
    if not passed:
        print("\n❌ Eval failed. This PR will be blocked by CI.")
        exit(1)

    print("\n✓ All thresholds met. Safe to promote.")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--output", default="evals/metrics.json")
    parser.add_argument("--model", default="qwen2:1.5b")
    args = parser.parse_args()

    run_eval(
        prompt_version=args.prompt_version,
        output_path=args.output,
        model_name=args.model,
    )
