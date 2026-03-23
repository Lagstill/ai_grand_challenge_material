# HOW TO RUN — Complete Setup Guide
## No OpenAI key. No cloud. Everything local.

---

## What you need installed first

| Tool | Install | Why |
|------|---------|-----|
| Python 3.11+ | python.org | runs all scripts |
| Docker + Docker Compose | docker.com | runs Langfuse |
| Ollama | ollama.com | runs Llama locally |
| W&B account | wandb.ai (free) | experiment tracking UI |
| Git | git-scm.com | already have it |

---

## Step 0 — Clone and configure (2 min)

```bash
git clone https://github.com/your-org/mlops-demo
cd mlops-demo

cp .env.example .env
# Edit .env — only WANDB_API_KEY is required for the basic demo
# Everything else has working defaults
```

---

## Step 1 — Install Python dependencies (3 min)

```bash
make setup
# or: pip install -r requirements.txt
```

---

## Step 2 — Start Ollama and pull Llama (10-15 min, once)

```bash
# Start the Ollama server
ollama serve                      # leave this running in a terminal tab

# Pull the models (new terminal tab)
ollama pull llama3.1              # ~4.7GB — the main inference model
ollama pull nomic-embed-text      # ~270MB — embeddings for RAGAS

# Verify they're running
ollama list
curl http://localhost:11434/v1/models   # should return model list as JSON
```

> **Slower machine?** Use `llama3.2:3b` instead of `llama3.1` — 2GB, 
> much faster. Change `OLLAMA_MODEL=llama3.2:3b` in your `.env`.
> Scores will be slightly lower but the demo works identically.

---

## Step 3 — Run the skew demo (30 seconds, no dependencies)

```bash
python src/skew_demo.py
```

This is Act 1 of the session. No GPU, no models, no API keys.
Just shows — visually — how the same input becomes different tokens
in training vs serving. Run this live in the session.

---

## Step 4 — Run eval against Llama (5-10 min)

```bash
make eval
# or: python src/eval.py --prompt-version v2 --output evals/metrics.json
```

What happens:
1. Loads `prompts/v2.yaml`
2. Runs inference on `data/eval_qa.json` via Ollama (Llama 3.1)
3. RAGAS uses Llama as the judge LLM (also via Ollama)
4. Writes scores to `evals/metrics.json`
5. Exits 0 if all thresholds met, exits 1 if any fail

Expected output:
```
Evaluating prompt version: v2
Eval set size: 5 examples
Inference model: llama3.1
────────────────────────────────────────
  ✓  faithfulness             0.821  (threshold: 0.75)
  ✓  context_recall           0.774  (threshold: 0.70)
  ✓  context_precision        0.698  (threshold: 0.65)
  ✓  answer_relevancy         0.813  (threshold: 0.75)
────────────────────────────────────────
Overall: PASSED
```

Now try v1 to show the regression:
```bash
python src/eval.py --prompt-version v1 --output evals/metrics_v1.json
# This will FAIL — v1 faithfulness is below threshold
# That's the point — this is what the CI gate catches
```

---

## Step 5 — W&B experiment tracking demo

```bash
# Get your free API key from wandb.ai → Settings → API Keys
# Set it: export WANDB_API_KEY=your_key
# Or add to .env

# Dry run — shows W&B init and prompt artifact logging without training
make train-dry
```

Then open the URL printed in terminal → your W&B dashboard.

For the live session, you'll have a **pre-run dashboard** with 2-3
completed runs already showing different hyperparameters. Prepare
this the night before:
```bash
# Run with different LR to generate comparison runs in W&B
# (use a small model like TinyLlama for speed, or just show pre-run dashboard)
WANDB_RUN_GROUP=demo-session python src/train.py \
  --config config/train_config.yaml \
  --run-name "lr-2e-4" --dry-run
```

---

## Step 6 — Start Langfuse (self-hosted observability)

```bash
make observe
# or: docker compose up -d langfuse

# First time only — wait ~30 seconds for DB to initialise
docker compose logs -f langfuse   # watch for "Ready on http://0.0.0.0:3000"
```

Then:
1. Open http://localhost:3000
2. Create account (local, no email verification needed)
3. Go to Settings → API Keys → Create new key
4. Copy public + secret key into your `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```

Run the observe demo:
```bash
python src/observe.py
# Sends 3 traced queries to Langfuse
# Open http://localhost:3000 → Traces — watch them appear live
```

In the Langfuse UI, click any trace → you see:
- Root span: `answer_question`
- Child span: `retrieve_context` (latency separate)
- Child span: `generate_answer` (prompt + response + tokens)
- Custom score: `response_has_answer`

This is what you demo on screen during Act 6.

---

## Step 7 — Promote a model to the registry (W&B required)

```bash
# Only works after make eval has passed
make promote                     # promotes to staging
make promote ALIAS=production    # promotes to production
```

---

## Full demo sequence for the session

```bash
# Terminal 1: Ollama (leave running throughout)
ollama serve

# Terminal 2: Your demo terminal (this is the one on screen)

# Act 1 — skew demo (no deps, instant)
python src/skew_demo.py

# Act 2 — show pre-run W&B dashboard (open in browser)
# Nothing to run — just show the dashboard you prepared

# Act 3 — live eval (runs ~5 min, start during Act 2 explanation)
make eval

# Act 4 — prompt diff (just open the two files side by side)
# prompts/v1.yaml vs prompts/v2.yaml

# Act 5 — show ml_ci.yml in editor, then show the failing PR screenshot
# (prepare a fake PR screenshot the night before)

# Act 6 — Langfuse traces (http://localhost:3000 already open)
python src/observe.py
```

---

## Night-before checklist

- [ ] `ollama pull llama3.1` complete
- [ ] `ollama pull nomic-embed-text` complete
- [ ] `make eval` ran successfully, `evals/metrics.json` exists
- [ ] W&B dashboard has at least 2 pre-run experiments with different hyperparams
- [ ] Langfuse running at http://localhost:3000, API keys in .env
- [ ] `python src/observe.py` ran, traces visible in Langfuse
- [ ] Prompts v1 and v2 open side by side in your editor
- [ ] `ml_ci.yml` open in your editor, line 91 (`sys.exit(1)`) visible
- [ ] Backup: screenshot of W&B dashboard and Langfuse traces in case of wifi issues

---

## Troubleshooting

**`ollama: connection refused`**
→ Run `ollama serve` first in a separate terminal

**RAGAS scoring is very slow**
→ Llama3.1 is 8B — use `OLLAMA_MODEL=llama3.2:3b` for 3x speed

**W&B not logging**
→ Check `WANDB_API_KEY` is set: `echo $WANDB_API_KEY`

**Langfuse traces not appearing**
→ Check keys match what's in http://localhost:3000 → Settings → API Keys
→ `docker compose logs langfuse` for errors

**`make eval` fails on thresholds with v2**
→ Llama 3.1 scores slightly differently from GPT — lower thresholds in
   `src/eval.py` THRESHOLDS dict if needed for demo purposes.
   `faithfulness: 0.65` is a safe floor with llama3.1.
