PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin
CONFIG ?= configs/default.yaml
FAST   := configs/fast.yaml

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install runtime and development dependencies
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

.PHONY: fmt
fmt: ## Apply formatting and import ordering
	$(BIN)/black src tests scripts app
	$(BIN)/ruff check --fix src tests scripts app

.PHONY: lint
lint: ## Run linters and formatting checks
	$(BIN)/ruff check src tests scripts app
	$(BIN)/black --check src tests scripts app

.PHONY: typecheck
typecheck: ## Run static type analysis
	$(BIN)/mypy

.PHONY: test
test: ## Run the test suite, skipping MCMC
	$(BIN)/pytest -m "not mcmc"

.PHONY: test-all
test-all: ## Run every test including MCMC sampling
	$(BIN)/pytest

.PHONY: coverage
coverage: ## Run tests with a coverage report
	$(BIN)/pytest -m "not mcmc" --cov=cip --cov-report=term-missing

.PHONY: check
check: lint typecheck test ## Run every quality gate

.PHONY: experiment
experiment: ## Analyse a randomised experiment end to end
	$(BIN)/python scripts/run_experiment.py --config $(CONFIG) --dataset nsw

.PHONY: benchmark
benchmark: ## Score every estimator against a known treatment effect
	$(BIN)/python scripts/run_benchmark.py --config $(CONFIG)

.PHONY: uplift
uplift: ## Fit uplift models and score them against the true ranking
	$(BIN)/python scripts/run_uplift.py --config $(CONFIG)

.PHONY: pipeline
pipeline: experiment benchmark uplift ## Run every analysis in order

.PHONY: dashboard
dashboard: ## Serve the dashboard on :8501
	$(BIN)/streamlit run app/dashboard.py

.PHONY: secrets-scan
secrets-scan: ## Look for credential-shaped strings in tracked files
	$(BIN)/python scripts/secrets_scan.py

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
