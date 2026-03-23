"""
src/skew_demo.py — Training-serving skew: the silent production killer.

This is the demo that lands hardest in the session.

The setup:
  - You train a model on data preprocessed one way
  - In production, the "same" preprocessing runs slightly differently
  - Same input → different representation → model sees something it never trained on
  - Your 91% eval score becomes 74% in prod. Nobody knows why.

Run this:
  python src/skew_demo.py

What the audience sees:
  The exact same raw input goes through two code paths.
  The model receives completely different tokens.
  No error is raised. Everything "works."
  This is why you need to version your preprocessing with your model.
"""

from transformers import AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ── The input — looks identical ───────────────────────────────────────────────

raw_input = "What are the known vulnerabilities in OpenSSL v3.0?"

# Fake "context" retrieved from a vector store
retrieved_context = [
    "OpenSSL 3.0 introduced a new provider architecture.",
    "CVE-2022-0778: infinite loop in BN_mod_sqrt() for non-prime moduli.",
    "CVE-2022-3786: buffer overflow in X.509 certificate verification.",
]


# ── Version 1: preprocessing at training time ────────────────────────────────
# Written quickly during experimentation.

def preprocess_training(question: str, contexts: list[str]) -> str:
    context_str = " ".join(contexts)           # joined with single space
    return f"Context: {context_str}\nQuestion: {question}\nAnswer:"


# ── Version 2: preprocessing at serving time ────────────────────────────────
# Written 2 months later by a different teammate for the FastAPI endpoint.
# "Same thing" — but different details.

def preprocess_serving(question: str, contexts: list[str]) -> str:
    context_str = "\n".join(contexts)          # joined with NEWLINE — not space
    context_str = context_str.strip().lower()  # lowercased — training data wasn't
    return f"Context:\n{context_str}\n\nQ: {question}\nA:"  # different format entirely


# ── Show the skew ─────────────────────────────────────────────────────────────

def demo_skew():
    print("=" * 60)
    print("TRAINING-SERVING SKEW DEMO")
    print("Same raw input. Different model input.")
    print("=" * 60)

    training_input = preprocess_training(raw_input, retrieved_context)
    serving_input  = preprocess_serving(raw_input, retrieved_context)

    print("\n── RAW INPUT ────────────────────────────────────────────")
    print(f"Question: {raw_input}")
    print(f"Contexts: {len(retrieved_context)} passages")

    print("\n── TRAINING PREPROCESSING ───────────────────────────────")
    print(repr(training_input))

    print("\n── SERVING PREPROCESSING ────────────────────────────────")
    print(repr(serving_input))

    print("\n── ARE THEY THE SAME? ───────────────────────────────────")
    print(f"Equal: {training_input == serving_input}")   # False

    # ── Now show the tokenization difference ─────────────────────────────────
    # The model is a token processor. Different text = different tokens.
    # The model was never trained on the serving format.

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        train_tokens   = tokenizer.encode(training_input)
        serving_tokens = tokenizer.encode(serving_input)

        print(f"\n── TOKEN COUNT ──────────────────────────────────────────")
        print(f"Training input tokens:  {len(train_tokens)}")
        print(f"Serving input tokens:   {len(serving_tokens)}")
        print(f"Difference:             {abs(len(train_tokens) - len(serving_tokens))} tokens")

        # Show first divergence point
        min_len = min(len(train_tokens), len(serving_tokens))
        for i in range(min_len):
            if train_tokens[i] != serving_tokens[i]:
                print(f"\nFirst divergence at token position {i}:")
                print(f"  Training: {tokenizer.decode([train_tokens[i]])!r}")
                print(f"  Serving:  {tokenizer.decode([serving_tokens[i]])!r}")
                break

    except Exception:
        print("\n[Skipping tokenizer diff — model not downloaded]")
        print("But the string repr above already shows the skew.")

    print("\n── THE FIX ──────────────────────────────────────────────")
    print("""
  1. Preprocessing is a versioned artifact — same as your model weights.
  2. Training and serving IMPORT from the same function, same file.
  3. That file is pinned in requirements.txt alongside the model version.
  4. Tested in CI: run both paths on the same input, assert outputs match.

  One file. One function. No drift.

  # src/preprocess.py  ← single source of truth
  def preprocess(question, contexts):
      ...

  # train.py
  from src.preprocess import preprocess

  # serve.py
  from src.preprocess import preprocess  ← same import, same output
    """)

    print("=" * 60)
    print("This is why Uber called it 'training-serving skew'.")
    print("It's not a bug. It's a process failure.")
    print("=" * 60)


if __name__ == "__main__":
    demo_skew()
