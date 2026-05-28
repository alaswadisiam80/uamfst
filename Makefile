PY ?= python

.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "install  install package (no deps)"
	@echo "dev      install dev + backbone extras"
	@echo "lint     ruff + black + isort checks"
	@echo "type     mypy --strict"
	@echo "test     pytest"
	@echo "smoke    2-step training on the smoke config"
	@echo "docker   build the container image"
	@echo "clean    remove caches and build artefacts"

install:
	$(PY) -m pip install --no-deps -e .

dev:
	$(PY) -m pip install -e ".[dev,backbones]"

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .
	$(PY) -m isort --check-only .

type:
	$(PY) -m mypy --strict src/uamfst

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m uamfst.cellarmaster.vintage --gin_file configs/experiment/_smoke.gin

docker:
	docker build -t uamfst:latest .

clean:
	rm -rf build dist *.egg-info .mypy_cache .ruff_cache .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
