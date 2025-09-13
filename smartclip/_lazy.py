from __future__ import annotations

"""Lazy backend resolution utilities.

Backends are imported only on demand to keep cold import time minimal and to
avoid importing optional deep learning frameworks unless actually used.
"""

from importlib import import_module
from types import ModuleType


class MissingBackend(RuntimeError):
    """Raised when a suitable backend cannot be found or imported."""


def _is_torch_model(model: object) -> bool:
    """Best-effort detection for PyTorch models.

    Prefer an isinstance check against torch.nn.Module when torch is available,
    which correctly handles user-defined modules (e.g., classes defined in
    application code). Fallback to module-name heuristics otherwise.
    """
    try:  # Fast path when torch is present
        torch_mod = import_module("torch")
        nn_mod = getattr(torch_mod, "nn", None)
        module_cls = getattr(nn_mod, "Module", None) if nn_mod is not None else None
        if module_cls is not None:
            return isinstance(model, module_cls)
        # If torch imported but structure unexpected, fall back to heuristic
        mod = type(model).__module__
        return mod.startswith("torch.") or mod.split(".")[0] == "torch"
    except Exception:
        # Heuristic fallback avoids importing torch eagerly
        mod = type(model).__module__
        return mod.startswith("torch.") or mod.split(".")[0] == "torch"


def _is_tf_model(model: object) -> bool:
    mod = type(model).__module__
    return mod.startswith("tensorflow") or mod.startswith("keras")


def _is_jax_model(model: object) -> bool:
    mod = type(model).__module__
    # Heuristics for Flax/linen, Equinox, Haiku modules
    return (
        mod.startswith("flax.")
        or mod.startswith("equinox")
        or mod.startswith("haiku")
        or mod.split(".")[0] == "jax"
    )


def get_backend_or_raise(model: object) -> ModuleType:
    """Return the backend module for the given model instance.

    The returned object is a Python module that exposes the public surface:
    - apply(model, clipper)
    - step(model, optimizer, clipper)
    - clip_context(model, optimizer=None, clipper=None)
    """

    if _is_torch_model(model):
        try:
            return import_module("smartclip.backends.torch")
        except Exception as exc:  # pragma: no cover - defensive
            raise MissingBackend(
                "PyTorch backend not available. Install with: pip install 'smartclip[torch]' and ensure torch is installed (see https://pytorch.org/get-started/)."
            ) from exc

    if _is_tf_model(model):
        try:
            return import_module("smartclip.backends.tf")
        except Exception as exc:  # pragma: no cover - defensive
            raise MissingBackend(
                "TensorFlow backend not available. Install with: pip install 'smartclip[tf]' and ensure tensorflow/tf-keras is installed."
            ) from exc

    if _is_jax_model(model):
        try:
            return import_module("smartclip.backends.jax")
        except Exception as exc:  # pragma: no cover - defensive
            raise MissingBackend(
                "JAX backend not available. Install with: pip install 'smartclip[jax]' and ensure jax/flax/optax are installed."
            ) from exc

    raise MissingBackend(
        "Unrecognized model type. Expected a PyTorch, TensorFlow, or JAX model instance."
    )


__all__ = [
    "MissingBackend",
    "get_backend_or_raise",
]
