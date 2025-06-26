# Makefile for nano-vllm project

.PHONY: help test lint format clean bench mini-bench
help:
	@echo "Available targets:"
	@echo "  install         Install all dependencies (auto-detects Jetson/Orin)"
	@echo "  test            Run all tests with pytest"
	@echo "  test-ci         Run tests as in CI (skipping heavy tests)"
	@echo "  lint            Run ruff for linting"
	@echo "  lint-ci         Run ruff and fail on errors (CI mode)"
	@echo "  format          Run ruff format and black for code formatting"
	@echo "  format-check    Check formatting with ruff and black (no changes)"
	@echo "  clean           Remove Python cache and build artifacts"
	@echo "  bench           Run full benchmark (bench.py)"
	@echo "  mini-bench      Run mini benchmark (scripts/mini_bench.py)"
	@echo "  act             Run GitHub Actions workflow locally with act"
	@echo "  mini-bench  Run mini benchmark (scripts/mini_bench.py)"


install:
	@if [ -f /proc/device-tree/compatible ] && grep -qiE 'nvidia,tegra|nvidia,orin' /proc/device-tree/compatible; then \
		echo "[INFO] Detected NVIDIA Jetson/Orin platform. Running Jetson setup..."; \
		$(MAKE) setup-jetson; \
	else \
		echo "[INFO] Running standard development setup..."; \
		$(MAKE) setup-regular; \
	fi

# Setup for NVIDIA Jetson/Orin (installs dependencies and NVIDIA PyTorch wheel)
setup-jetson:
	uv sync
	bash scripts/install_nvidia_tegra_pytorch.sh

# Setup for development (installs all dev dependencies)
setup-regular:
	uv sync
	uv pip install torch

# Run all tests
test:
	pytest tests

# Run tests as in CI (skipping heavy tests)
test-ci:
	CI=1 pytest tests


# Lint with ruff
lint:
	ruff check nanovllm tests

# Lint for CI (fail on errors)
lint-ci:
	ruff check --exit-zero nanovllm tests


# Format with ruff only
format:
	ruff format nanovllm tests

# Check formatting only with ruff
format-check:
	ruff format --check nanovllm tests

# Clean up cache and build artifacts
clean:
	rm -rf __pycache__ nanovllm/__pycache__ tests/__pycache__ *.pyc *.pyo *.egg-info build dist

# Run full benchmark
bench:
	python bench.py

# Run mini benchmark
mini-bench:
	python scripts/mini_bench.py

# Run nano-vllm FastAPI server
run-server:
	uvicorn apps.server.server:app --host 0.0.0.0 --port 8000 --reload

# Run GitHub Actions workflow locally with act
act:
	act --workflows .github/workflows/pr.yaml --job test --container-architecture linux/arm64

diff-to-parent:
	git diff parent/main -- 'nanovllm/**/*.py'