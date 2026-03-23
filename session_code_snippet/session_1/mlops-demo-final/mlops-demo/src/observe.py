"""
src/observe.py — Production observability with Langfuse.

W&B is for training. Langfuse is for production.
They cover different halves of the model lifecycle.

What Langfuse captures on every single inference call:
  - Exact prompt that went in (after template rendering)
  - Raw model output
  - Token counts (prompt + completion)
  - Latency (per-call and end-to-end)
  - Which prompt version was used
  - User ID and session (for debugging specific users)
  - Any custom scores you attach (e.g., your faithfulness check)

Self-hosted in 5 minutes:
  docker compose up -d langfuse
  → UI at http://localhost:3000

For gov/defense teams: runs entirely in your VPC.
No data leaves your infrastructure.

Usage:
  from src.observe import answer_question
  response = answer_question(question="...", context_passages=[...], user_id="u123")
"""

import os
import time
import yaml

from pathlib import Path
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context


# ── Init clients ──────────────────────────────────────────────────────────────
# For self-hosted Langfuse, point to your own host:
#   LANGFUSE_HOST=http://localhost:3000

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),  # default to self-hosted
)

# Ollama — OpenAI-compatible, fully local
from openai import OpenAI as _OpenAI
openai_client = _OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
    api_key="ollama",
)

OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")


# ── Load prompt ───────────────────────────────────────────────────────────────

def load_prompt(version: str) -> dict:
    with open(f"prompts/{version}.yaml") as f:
        return yaml.safe_load(f)


# ── The @observe decorator is the whole trick ─────────────────────────────────
# It wraps a function and automatically traces:
#   - function name → span name
#   - arguments → input
#   - return value → output
#   - exceptions → error
#   - duration → latency
#
# Nested @observe calls become nested spans — perfect for RAG pipelines
# where retrieval and generation are separate steps.

@observe()     # <── this single decorator = full production observability
def retrieve_context(question: str, top_k: int = 3) -> list[str]:
    """
    Your retrieval step. Replace with your actual vector store call.
    Langfuse will trace this as a separate span — you'll see retrieval
    latency separately from generation latency.
    """
    # Placeholder — swap for ChromaDB, Weaviate, Pinecone, etc.
    return [
        f"[Retrieved passage 1 relevant to: {question}]",
        f"[Retrieved passage 2 relevant to: {question}]",
        f"[Retrieved passage 3 relevant to: {question}]",
    ]


@observe(as_type="generation")  # marks this span as an LLM generation call
def generate_answer(
    question: str,
    contexts: list[str],
    prompt: dict,
    model: str = None,   # defaults to OLLAMA_MODEL env var
) -> str:
    model = model or OLLAMA_MODEL
    """
    The generation step. Langfuse captures:
      - Rendered prompt (system + user)
      - Model response
      - Token usage
      - Latency
      - Model name
    """
    system = prompt["system"].format(context="\n\n".join(contexts))
    user   = prompt["user_template"].format(
        context="\n\n".join(contexts),
        question=question,
    )

    # Log the exact inputs to Langfuse — this is your debugging lifeline
    langfuse_context.update_current_observation(
        input={"system": system, "user": user},
        model=model,
        metadata={"prompt_version": PROMPT_VERSION},
    )

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()
    usage  = response.usage

    # Log token usage — Langfuse uses this for cost tracking
    langfuse_context.update_current_observation(
        output=answer,
        usage={
            "input":  usage.prompt_tokens,
            "output": usage.completion_tokens,
        },
    )

    return answer


@observe()     # top-level span — wraps retrieval + generation as child spans
def answer_question(
    question: str,
    user_id: str = "anonymous",
    session_id: str = None,
) -> dict:
    """
    The full RAG pipeline. One call = one Langfuse trace with nested spans:

    answer_question (root span)
    ├── retrieve_context (span)
    └── generate_answer (generation span)
         ├── prompt rendered
         ├── model response
         └── token usage

    You can click any trace in the Langfuse UI and see the full tree.
    """
    # Tag this span with metadata for debugging
    langfuse_context.update_current_trace(
        metadata={
            "question_length": len(question),
            "user_id": user_id,
            "session_id": session_id,
            "prompt_version": PROMPT_VERSION,
        },
    )

    prompt   = load_prompt(PROMPT_VERSION)
    contexts = retrieve_context(question)
    answer   = generate_answer(question, contexts, prompt)

    # Attach a custom score — e.g., a lightweight faithfulness heuristic
    # In production, you'd run a small classifier here or RAGAS async
    langfuse_context.score_current_trace(
        name="response_has_answer",
        value=0.0 if "don't have enough context" in answer.lower() else 1.0,
        comment="Heuristic: did the model produce a substantive answer?",
    )

    return {
        "question": question,
        "answer":   answer,
        "contexts": contexts,
        "prompt_version": PROMPT_VERSION,
    }


# ── Demo entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running 3 demo queries — watch them appear in Langfuse...\n")

    questions = [
        "What is the main vulnerability in the authentication flow?",
        "How does the anomaly detection threshold get calibrated?",
        "What data sources are used for vessel classification?",
    ]

    for i, q in enumerate(questions):
        result = answer_question(
            question=q,
            user_id=f"demo-user-{i}",
            session_id="mlops-demo",
        )
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:120]}...")
        print()

    # Flush — important for short scripts, Langfuse sends async
    langfuse.flush()
    print(f"✓ Traces sent → {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
