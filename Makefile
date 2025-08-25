# Makefile for magnet_diffrax package

.PHONY: help install install-dev test test-unit test-integration test-slow test-coverage clean format lint type-check docs build upload

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install package in current environment"
	@echo "  install-dev      - Install package with development dependencies"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"  
	@echo "  test-integration - Run integration tests only"
	@echo "  test-slow       - Run slow tests"
	@echo "  test-coverage   - Run tests with coverage report"
	@echo "  clean           - Clean build artifacts"
	@echo "  format          - Format code with black and isort"
	@echo "  lint            - Run linting with flake8"
	@echo "  type-check      - Run type checking with mypy"
	@echo "  docs            - Build documentation"
	@echo "  build           - Build distribution packages"
	@echo "  upload          - Upload to PyPI (requires credentials)"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

# Testing targets
test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-slow:
	pytest -m slow

test-coverage:
	pytest --cov=magnet_diffrax --cov-report=html --cov-report=xml --cov-report=term

test-fast:
	pytest -m "not slow"

# Code quality targets
format:
	black magnet_diffrax tests
	isort magnet_diffrax tests

lint:
	flake8 magnet_diffrax tests

type-check:
	mypy magnet_diffrax

# Documentation targets
docs:
	@echo "Building documentation..."
	@echo "Note: Sphinx not configured yet. Use 'sphinx-quickstart docs' to set up."

# Build and distribution targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Development workflow targets
check: format lint type-check test-fast
	@echo "All checks passed!"

ci: lint type-check test-coverage
	@echo "CI checks completed!"

# Quick development test
dev-test:
	pytest tests/ -v --tb=short -x

# Performance testing
perf-test:
	pytest -m "not slow" --benchmark-only

# Create sample data for testing
create-samples:
	python -c "from magnet_diffrax.main import create_sample_data; create_sample_data()"
	python -c "from magnet_diffrax.coupled_main import create_sample_coupled_data; create_sample_coupled_data()"

# Run examples
run-single-example:
	python -m magnet_diffrax.main --show-plots --show-analytics

run-coupled-example:
	python -m magnet_diffrax.coupled_main --n-circuits 3 --show-plots --show-analytics

# Docker targets (if needed)
docker-build:
	docker build -t magnet-diffrax .

docker-test:
	docker run --rm magnet-diffrax pytest

# Environment setup
setup-env:
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Update dependencies
update-deps:
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -e ".[dev,docs]"
