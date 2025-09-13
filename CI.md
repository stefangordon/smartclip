# Continuous Integration (CI)

This repository uses GitHub Actions to provide CI for code quality, tests, docs, and releases. The CI is designed to:

- Keep the core package pure-Python and quick to test
- Run real integration tests per framework on CPU-only wheels
- Enforce style and typing
- Build and publish releases using PyPI Trusted Publishing (OIDC)

The three workflows are:

- `.github/workflows/ci.yml` — lint, type-check, and test (core + frameworks)
- `.github/workflows/docs.yml` — build and deploy MkDocs documentation to GitHub Pages
- `.github/workflows/release.yml` — build distributions and publish to PyPI via OIDC

---

## Test layout and markers

Tests are split into two categories:

- Unit tests (default): fast, hermetic tests that do not require heavy frameworks. These run everywhere.
- Integration tests: require real framework wheels (PyTorch, TensorFlow, JAX). Marked with `@pytest.mark.integration`.

The pytest marker is registered in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
  "integration: marks tests that require real framework dependencies",
]
```

CI selects tests via markers:

- Core job: `-m "not integration"`
- Framework jobs: `-m integration` (scoped to each framework's test directory)

This keeps the core job lightweight while still exercising real integrations where appropriate.

---

## Workflow: ci.yml

Matrix across Python versions 3.9–3.13 (TensorFlow job currently excludes 3.13 until wheels are available). Each job uses `uv` for fast installs and consistent execution.

Jobs:

1. core
   - Installs: `uv pip install -e .[test,dev] --system`
   - Lint: `uv run ruff check .` and `uv run ruff format --check .`
   - Types: `uv run mypy smartclip`
   - Tests: `uv run pytest -q -m "not integration" --maxfail=1 --disable-warnings --cov=smartclip --cov-report=xml --junitxml=reports/junit-core.xml`
   - Artifacts: JUnit XML + `coverage.xml`

2. torch
   - Installs Torch CPU wheels from the PyTorch CPU index, then `uv pip install -e .[test,torch] --system`
   - Tests: `uv run pytest -q -m integration tests/torch --junitxml=reports/junit-torch.xml`

3. tf
   - Installs `tensorflow-cpu` wheels, then `uv pip install -e .[test] --system`
   - Tests: `uv run pytest -q -m integration tests/tf --junitxml=reports/junit-tf.xml`

4. jax
   - Installs `jax[cpu]`, `flax`, `optax`, then `uv pip install -e .[test] --system`
   - Tests: `uv run pytest -q -m integration tests/jax --junitxml=reports/junit-jax.xml`

Notes:

- All framework jobs are CPU-only to ensure portability and predictable runtimes.
- Stubs and shim tests (which fake framework modules) live in each framework directory but are not marked `integration`, so they run in the core job.
- Artifacts are uploaded per job as JUnit XML for CI surfaces.

---

## Running tests locally

You can run tests with either `uv` (recommended) or vanilla `pip`.

### With uv

```bash
# Install uv (see https://github.com/astral-sh/uv)
uv venv
. .venv/bin/activate  # or .venv\Scripts\activate on Windows

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

### With pip

```bash
python -m venv .venv
. .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Core
pip install -e .[test,dev]
ruff check . && ruff format --check . && mypy smartclip
pytest -q -m "not integration"

# PyTorch integrations
pip install --index-url https://download.pytorch.org/whl/cpu torch --extra-index-url https://pypi.org/simple
pip install -e .[test,torch]
pytest -q -m integration tests/torch

# TensorFlow integrations
pip install tensorflow-cpu --only-binary :all:
pip install -e .[test]
pytest -q -m integration tests/tf

# JAX integrations
pip install "jax[cpu]" flax optax
pip install -e .[test]
pytest -q -m integration tests/jax
```

#### Selecting specific tests

- Run only one file: `pytest tests/torch/test_torch_real_integration.py -m integration`
- Run a single test: `pytest -k test_name -m integration`
- Exclude slow tests (if any): `pytest -m "not slow"`

---

## Adding new integration tests

- Put tests under the appropriate directory: `tests/torch`, `tests/tf`, `tests/jax`.
- Decorate with `@pytest.mark.integration`.
- Keep tests CPU-only and fast (< ~2 minutes per job total).
- Prefer tiny models and batches, deterministic seeds, and minimal dependencies.

Example:

```python
import pytest
pytestmark = pytest.mark.integration

def test_feature_x_works_with_backend():
    ...
```

---

## Docs workflow (docs.yml)

- Trigger: pushes to `main` or manual dispatch
- Steps: install `-e .[docs]`, `mkdocs build --strict`, upload artifact, deploy to GitHub Pages
- Local build: `pip install -e .[docs] && mkdocs serve`

---

## Release workflow (release.yml)

- Trigger: GitHub Release `published` (or manual dispatch)
- Builds sdist and wheel with `python -m build`
- Publishes to PyPI via `pypa/gh-action-pypi-publish@release/v1` (OIDC / Trusted Publishing)
- No API tokens are stored in repo secrets; ensure the PyPI project is configured with the repo as a Trusted Publisher

Release steps:

1. Update version (if not using VCS versioning) and changelog
2. Create a GitHub Release with tag `vX.Y.Z`
3. Workflow builds and publishes automatically

---

## Troubleshooting

- "Module X not found" in integration tests: ensure you installed the right CPU wheels as shown above.
- TensorFlow on Python 3.13: the CI excludes it until official CPU wheels are published.
- Timeouts/slow runs: use `-k` to target specific tests and keep batch sizes small.
- Windows: activate venv with `.venv\Scripts\activate` and prefer CPU wheels.

---

## Quick reference

- Core tests only: `pytest -q -m "not integration"`
- All integration tests: `pytest -q -m integration`
- Only PyTorch integrations: `pytest -q -m integration tests/torch`
- Lint + types: `ruff check . && ruff format --check . && mypy smartclip`

If you run into CI issues, open an issue with the job URL and failing step output.
