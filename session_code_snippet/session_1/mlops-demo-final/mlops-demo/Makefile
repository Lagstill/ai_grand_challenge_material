# ---------------------------------------------------------------
# MLOps/LLMOps Demo
# Every teammate runs the same commands. No undocumented flags.
# ---------------------------------------------------------------

.PHONY: setup train eval promote serve observe clean

# ---------- environment ----------
setup:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# ---------- training ----------
train:
	python src/train.py \
		--config config/train_config.yaml \
		--run-name "$(shell date +%Y%m%d_%H%M%S)"

train-dry:
	python src/train.py \
		--config config/train_config.yaml \
		--dry-run

# ---------- evaluation ----------
eval:
	python src/eval.py \
		--prompt-version $(or $(PROMPT_VERSION), v2) \
		--output evals/metrics.json
	@echo "── Eval results ──"
	@cat evals/metrics.json | python -m json.tool

# ---------- model registry ----------
promote:
	@test -f evals/metrics.json || (echo "❌ Run 'make eval' first" && exit 1)
	python src/promote.py \
		--metrics evals/metrics.json \
		--alias $(or $(ALIAS), staging)

# ---------- serving ----------
serve:
	uvicorn src.serve:app --reload --port 8000

# ---------- production observability ----------
observe:
	docker compose up -d langfuse
	@echo "✓ Langfuse UI → http://localhost:3000"

# ---------- cleanup ----------
clean:
	rm -f evals/metrics.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean"
