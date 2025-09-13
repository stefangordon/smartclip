# Contributing to smartclip

Thanks for your interest in contributing! This guide covers developer setup, testing, docs, benchmarks, and releasing. For user docs, see the project site.

## Developer setup

### Create environment and install

```bash
uv venv
uv pip install -e ".[dev,test,docs]"
```

Optional framework extras (install framework wheels per vendor docs first):

- PyTorch: `uv pip install -e ".[dev,test,docs,torch]"`
- TensorFlow: `uv pip install -e ".[dev,test,docs,tf]"`
- JAX: `uv pip install -e ".[dev,test,docs,jax]"`

Install pre-commit hooks:

```bash
uv run pre-commit install
```

### Common tasks

- Lint:

```bash
uv run ruff check .
uv run ruff format --check .
```

- Type check:

```bash
uv run mypy smartclip
```

- Tests (unit):

```bash
uv run pytest -q -m "not integration"
```

- Tests (parallel + coverage):

```bash
uv run pytest -q -n auto --cov=smartclip --cov-report=term-missing -m "not integration"
```

- Docs (serve locally):

```bash
uv run mkdocs serve -a 127.0.0.1:8000
```

- Docs (build):

```bash
uv run mkdocs build --strict
```

- Build distributions (wheel + sdist):

```bash
uv run python -m build
```

## Running integration tests locally

Integration tests require real framework CPU wheels. Keep runs small and deterministic.

With `uv`:

```bash
# Core (no heavy frameworks)
uv pip install -e .[test,dev]
uv run ruff check . && uv run ruff format --check . && uv run mypy smartclip
uv run pytest -q -m "not integration"

# PyTorch integrations (CPU wheels)
uv pip install --index-url https://download.pytorch.org/whl/cpu torch --extra-index-url https://pypi.org/simple --system
uv pip install -e .[test,torch] --system
uv run pytest -q -m integration tests/torch

# TensorFlow integrations (CPU wheels)
uv pip install tensorflow-cpu --only-binary :all: --system
uv pip install -e .[test] --system
uv run pytest -q -m integration tests/tf

# JAX integrations (CPU)
uv pip install "jax[cpu]" flax optax --system
uv pip install -e .[test] --system
uv run pytest -q -m integration tests/jax
```

Selecting tests:

- File: `pytest tests/torch/test_torch_real_integration.py -m integration`
- Single test: `pytest -k test_name -m integration`

## Benchmarks

CPU-only micro-benchmarks to compare algorithm stability.

Run (after installing the framework):

```bash
# PyTorch CNN
uv run python benchmarks/torch_cnn.py --steps 50 --batch-size 64 --dataset fashion_mnist --algo autoclip

# TensorFlow CNN
uv run python benchmarks/tf_cnn.py --steps 50 --batch-size 64 --dataset cifar10 --algo agc

# JAX MLP
uv run python benchmarks/jax_mlp.py --steps 50 --batch-size 256 --dataset fashion_mnist --algo zscore
```

Plotting comparisons (generates SVGs under `docs/assets/`):

```bash
uv pip install -e ".[bench]"
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install torchvision
uv run python benchmarks/plot_benchmarks.py
```

## Versioning and releases

- Version derived from Git tags via hatch-vcs
- Tag releases like:

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push --tags
```

Publishing:

- CI publishes to PyPI via Trusted Publishing (OIDC) when a GitHub Release is published

## CI overview

GitHub Actions workflows:

- `ci.yml`: lint, types, core tests, integration tests (CPU-only)
- `docs.yml`: build and deploy MkDocs site to GitHub Pages
- `release.yml`: build distributions and publish to PyPI (OIDC)

Notes:

- Framework jobs use CPU wheels only for portability
- PyTest markers:

```toml
[tool.pytest.ini_options]
markers = [
  "integration: marks tests that require real framework dependencies",
]
```

## Opening PRs

1. Fork and create a feature branch
2. Add tests and docs for changes
3. Run lint, types, and tests locally
4. Open PR with a clear description and motivation

## Code style

We use Ruff and MyPy. Favor small, readable functions, explicit types on public APIs, and early returns. Keep imports fast. Avoid catching exceptions without handling.
