# MLOps / LLMOps — IIT Delhi AI Grand Challenge
### Session: Best Practices in MLOps/LLMOps

> "A team just got their LLM fine-tune working in a Colab notebook.
> Congrats. Now their teammate runs it — different results. They try a
> new hyperparameter — they forgot what the last one was. They ship a
> new model — it's worse, and they don't know why.
> This repo is the system you build so none of that ever happens."

---

## The 5 failure modes this repo fixes

| # | Failure | Fix |
|---|---------|-----|
| 1 | Can't reproduce a training run | `config/train_config.yaml` + W&B run tracking |
| 2 | Can't compare two experiments | W&B dashboard with parallel run comparison |
| 3 | No path from experiment to production | W&B Model Registry with staging/production aliases |
| 4 | Model regressions reach production | GitHub Actions eval gate blocks the PR |
| 5 | Production model is a black box | Langfuse traces every prompt/response/token |

---

## Quickstart

```bash
git clone https://github.com/your-org/mlops-llmops-demo
cd mlops-llmops-demo
cp .env.example .env     # fill in your API keys

make setup               # install dependencies
make train               # fine-tune + log to W&B
make eval                # run RAGAS evals, write metrics.json
make promote             # promote to staging if metrics pass
make observe             # start Langfuse locally
```

---

## Repo structure

```
├── Makefile                        # single interface for all commands
├── config/
│   └── train_config.yaml           # all hyperparams — versioned in git
├── prompts/
│   ├── v1.yaml                     # faithfulness: 0.61 — retired
│   └── v2.yaml                     # faithfulness: 0.84 — current production
├── src/
│   ├── train.py                    # QLoRA fine-tuning + W&B logging
│   ├── eval.py                     # RAGAS evaluation → metrics.json
│   ├── promote.py                  # W&B registry promotion with gates
│   ├── observe.py                  # Langfuse @observe instrumentation
│   └── skew_demo.py                # training-serving skew illustration
├── evals/
│   └── metrics.json                # eval output — read by CI gate
├── docker-compose.yml              # self-hosted Langfuse (runs in your VPC)
└── .github/workflows/
    └── ml_ci.yml                   # PR gate — blocks metric regressions
```

---

## The full production loop

```
Write evals (RAGAS)          ← before you write a line of training code
       ↓
Train  (W&B tracked)
       ↓
Eval   (metrics.json)
       ↓
CI gate (GitHub Actions)     ← PR is blocked if metrics regress
       ↓
Promote (W&B Registry)       ← staging alias → production alias
       ↓
Serve  (FastAPI)
       ↓
Observe (Langfuse)           ← every prompt/response/token traced
       ↓
Alert on drift               ← back to train
```

---

## Tools used

| Tool | Purpose | Self-hostable? |
|------|---------|----------------|
| W&B | Experiment tracking + model registry | Yes (W&B Server) |
| RAGAS | LLM evaluation framework | Yes |
| GitHub Actions | CI/CD eval gate | Yes (self-hosted runners) |
| Langfuse | Production LLM observability | Yes (Docker Compose) |

---

## Prompt versioning — why it matters

```bash
git diff prompts/v1.yaml prompts/v2.yaml
```

A single prompt edit can shift RAGAS faithfulness by ±0.2.
Prompts are code. Version them. Gate them. The diff above is your audit trail.

---

## The Makefile principle

Every teammate runs `make train`, `make eval`, `make promote`.
No undocumented flags. No "works on my machine."
The Makefile is the contract between teammates.
