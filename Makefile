.PHONY: install run scan test eval eval-live eval-freeze lint web-build web-sanity docker-build docker-run

install:
	python -m pip install -e ".[dev]"

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

scan:
	python -m src.cli_app scan $(IMAGE)

test:
	pytest

# Offline regression gate — scanner-only, no Gemini, no LangSmith, no network.
# Exits non-zero on quality regression against the frozen baseline.
eval:
	python -m src.evals.regression

# Freeze the current scanner-only results as the regression baseline.
eval-freeze:
	python -m src.evals.regression --write-baseline

# Full-pipeline evaluation (scanner + Gemini + recovery). Requires GEMINI_API_KEY.
# Observational — Gemini is nondeterministic, so this is NOT a hard gate.
eval-full:
	python -m src.evals.regression --full-pipeline

# Freeze the current full-pipeline results as an observational baseline.
eval-full-freeze:
	python -m src.evals.regression --full-pipeline --write-baseline

# Capture Gemini audit results for deterministic replay. Requires GEMINI_API_KEY.
eval-full-cache:
	python -m src.evals.regression --full-pipeline --cache-gemini

# Deterministic full-pipeline eval using cached Gemini results. No Gemini calls.
eval-full-replay:
	python -m src.evals.regression --full-pipeline --replay-gemini

# Freeze deterministic full-pipeline results (using cached Gemini).
eval-full-replay-freeze:
	python -m src.evals.regression --full-pipeline --replay-gemini --write-baseline

# Live evaluation with Gemini + LangSmith (charged, needs GEMINI_API_KEY).
eval-live:
	python -m src.evals.runner

lint:
	ruff check .

# Frontend typecheck + production build gate (no network, no Gemini).
web-build:
	cd web && npm run build

# Live end-to-end web sanity check. Requires Docker Postgres, .env, sample
# images, and web/node_modules. Uses the frozen Gemini audit cache (replay
# mode) for deterministic full-pipeline results — no live Gemini calls.
web-sanity:
	./scripts/web_sanity.sh

docker-build:
	docker build -t barcode-scanner .

docker-run:
	docker run --rm -p 8000:8000 barcode-scanner
