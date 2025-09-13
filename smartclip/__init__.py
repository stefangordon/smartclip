"""smartclip: Adaptive gradient clipping algorithms for deep learning frameworks.

This package provides a framework-agnostic core with optional thin integrations for
PyTorch, TensorFlow/Keras, and JAX/Flax. Public APIs are typed and designed for
fast import times and production use.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .core.types import OnMetricsCallback

try:  # Prefer a real version from installed metadata; fall back during dev.
    __version__ = version("smartclip")
except PackageNotFoundError:  # pragma: no cover - only hit in editable installs without build
    __version__ = "0.0.0"

# Expose core algorithms at the top level
from ._lazy import get_backend_or_raise
from .core import AGC, AutoClip, ZScoreClip

__all__ = [
    "__version__",
    # Algorithms
    "AutoClip",
    "AGC",
    "ZScoreClip",
    # Top-level backend-agnostic helpers
    "apply",
    "step",
    "clip_context",
]


def apply(
    model: Any,
    clipper: AutoClip | AGC | ZScoreClip,
    on_metrics: OnMetricsCallback | None = None,
) -> Any:
    """Apply adaptive clipping to model parameters.

    Delegates to the active backend determined from the model instance.
    """

    backend = get_backend_or_raise(model)
    return backend.apply(model, clipper, on_metrics)


def step(
    model: Any,
    optimizer: Any,
    clipper: AutoClip | AGC | ZScoreClip,
    on_metrics: OnMetricsCallback | None = None,
) -> Any:
    """Clip gradients on the model and then call optimizer.step()."""

    backend = get_backend_or_raise(model)
    return backend.step(model, optimizer, clipper, on_metrics)


def clip_context(
    model: Any,
    optimizer: Any | None = None,
    clipper: AutoClip | AGC | ZScoreClip | None = None,
    on_metrics: OnMetricsCallback | None = None,
) -> Any:
    """Context manager that clips before each optimizer step for the active backend.

    Defaults to AutoClip() when clipper is None.
    """

    backend = get_backend_or_raise(model)
    return backend.clip_context(model, optimizer, clipper, on_metrics)
