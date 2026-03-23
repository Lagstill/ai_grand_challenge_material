"""
src/promote.py — Promote a model from "run artifact" to the registry.

This implements the Michelangelo pattern:
  - Models are identified by UUID (W&B run ID)
  - Aliases (staging, production) resolve to the latest promoted model
  - Clients link to the alias, not the UUID — seamless rollouts
  - Full lineage: which training run → which eval → which registry version

Usage:
  make promote                      # promotes latest eval'd run to staging
  make promote ALIAS=production     # promotes to production
  python src/promote.py --metrics evals/metrics.json --alias staging
"""

import argparse
import json
import os
import sys

import wandb


# ── Constants ─────────────────────────────────────────────────────────────────

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "iit-delhi-demo")
WANDB_ENTITY  = os.getenv("WANDB_ENTITY", None)
REGISTRY_NAME = "fine-tuned-model"

# Minimum scores required to even attempt a promotion.
# These are stricter than the CI thresholds — production bar is higher.
PROMOTION_GATES = {
    "staging":    {"faithfulness": 0.75, "answer_relevancy": 0.75},
    "production": {"faithfulness": 0.82, "answer_relevancy": 0.82, "context_recall": 0.75},
}


# ── Core promotion logic ──────────────────────────────────────────────────────

def promote(metrics_path: str, alias: str):

    # ── 1. Load eval results ─────────────────────────────────────────────────
    with open(metrics_path) as f:
        metrics = json.load(f)

    if not metrics["passed"]:
        print(f"❌ Eval did not pass baseline thresholds. Cannot promote to {alias}.")
        sys.exit(1)

    # ── 2. Check promotion gate for the target environment ───────────────────
    gate = PROMOTION_GATES.get(alias, {})
    scores = metrics["scores"]
    gate_failures = []

    print(f"Checking promotion gate for: {alias}")
    for metric, min_score in gate.items():
        actual = scores.get(metric, 0.0)
        ok = actual >= min_score
        status = "✓" if ok else "✗"
        print(f"  {status}  {metric:<25} {actual:.3f}  (gate: {min_score})")
        if not ok:
            gate_failures.append(metric)

    if gate_failures:
        print(f"\n❌ Promotion to '{alias}' blocked — metrics below gate: {gate_failures}")
        sys.exit(1)

    # ── 3. Find the latest logged model artifact from W&B ────────────────────
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT)

    # Get most recent run that has a model artifact
    latest_artifact = None
    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type == "model" and artifact.name.startswith(REGISTRY_NAME):
                latest_artifact = artifact
                break
        if latest_artifact:
            break

    if not latest_artifact:
        print("❌ No model artifact found in W&B. Run 'make train' first.")
        sys.exit(1)

    print(f"\nArtifact: {latest_artifact.name}:{latest_artifact.version}")

    # ── 4. Link to registry with alias ───────────────────────────────────────
    # This is the Michelangelo pattern:
    # The alias (staging/production) is what clients resolve.
    # The underlying UUID never changes — full lineage preserved.

    collection_path = (
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/{REGISTRY_NAME}"
        if WANDB_ENTITY
        else f"{WANDB_PROJECT}/{REGISTRY_NAME}"
    )

    latest_artifact.link(collection_path, aliases=[alias, "latest"])

    print(f"\n✓ Model promoted to registry")
    print(f"  Alias:   {alias}")
    print(f"  Version: {latest_artifact.version}")
    print(f"  Run:     {latest_artifact.source_run.url}")
    print(f"\nClients loading alias '{alias}' will now get this version automatically.")
    print("No client code changes needed.")

    # ── 5. Log promotion metadata ─────────────────────────────────────────────
    # Record that this promotion happened, and with what eval scores.
    # This is your audit trail.
    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="promotion",
        name=f"promote-{alias}-{latest_artifact.version}",
        config={
            "alias": alias,
            "artifact_version": latest_artifact.version,
            "eval_scores": scores,
            "prompt_version": metrics["prompt_version"],
        },
    ) as run:
        run.log({"promoted": 1, **{f"eval/{k}": v for k, v in scores.items()}})
        print(f"\nPromotion logged: {run.url}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="evals/metrics.json",
                        help="Path to metrics.json written by eval.py")
    parser.add_argument("--alias", default="staging",
                        choices=["staging", "production"],
                        help="Registry alias to assign")
    args = parser.parse_args()

    promote(metrics_path=args.metrics, alias=args.alias)
