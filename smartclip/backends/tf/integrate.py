from __future__ import annotations

"""TensorFlow Keras integrations: Callback and optimizer patch helper.

These utilities are optional and only require TensorFlow when actually used.
They work by wrapping ``optimizer.apply_gradients`` to clip gradients using the
backend's ``apply_grads`` function before the optimizer update.
"""

from typing import Any, Callable, Iterable, Tuple


def _get_tf_callback_base() -> Any:
    try:
        import tensorflow as tf  # type: ignore

        return tf.keras.callbacks.Callback
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "TensorFlow is not available. Please install tensorflow or tensorflow-cpu."
        ) from exc


def _resolve_model(model_or_ref: Any) -> Any:
    # Check if it's a reference/accessor (weakref, lambda, etc.) vs an actual model
    # Keras models are callable but have specific attributes we can check
    if hasattr(model_or_ref, 'trainable_variables'):
        # It's already a model
        return model_or_ref
    elif callable(model_or_ref):
        # It's a reference that needs to be called to get the model
        return model_or_ref()
    else:
        # Just return as-is
        return model_or_ref


def patch_optimizer_apply_gradients(
    optimizer: Any, model_or_ref: Any, clipper: Any
) -> Callable[[], None]:
    """Monkey-patch ``optimizer.apply_gradients`` to clip grads first.

    Returns a callable that restores the original method.
    """

    import smartclip.backends.tf as sc_tf  # lazy import

    orig_apply = optimizer.apply_gradients

    def wrapped_apply_gradients(
        grads_and_vars: Iterable[Tuple[Any, Any]], *args: Any, **kwargs: Any
    ) -> Any:
        pairs = list(grads_and_vars)
        grads = [g for g, _ in pairs]
        vars_ = [v for _, v in pairs]
        _ = _resolve_model(model_or_ref)  # ensure model is resolvable, but not needed directly here
        clipped = sc_tf._apply_grads_to_vars(grads, vars_, clipper)
        # Avoid Python 3.10+ only zip(strict=True); maintain 3.9 compatibility
        new_pairs = [(cg, v) for cg, v in zip(clipped, vars_)]
        return orig_apply(new_pairs, *args, **kwargs)

    optimizer.apply_gradients = wrapped_apply_gradients

    def restore() -> None:
        optimizer.apply_gradients = orig_apply

    return restore


class SmartClipCallback(_get_tf_callback_base()):  # type: ignore[misc]
    """Keras Callback that clips gradients before optimizer updates in ``fit``.

    Parameters
    ----------
    model_ref: object or zero-arg callable returning the model
        Allows passing a weakref or late-bound accessor to the model instance.
    optimizer: tf.keras.optimizers.Optimizer
        Optimizer instance used by the model.
    clipper: smartclip AutoClip | AGC | ZScoreClip
        Clipper algorithm instance.
    """

    def __init__(self, model_ref: Any, optimizer: Any, clipper: Any) -> None:
        super().__init__()
        self.model_ref = model_ref
        self.optimizer = optimizer
        self.clipper = clipper
        self._restore_fn: Callable[[], None] | None = None

    def on_train_begin(self, logs: Any | None = None) -> None:
        self._restore_fn = patch_optimizer_apply_gradients(
            self.optimizer, self.model_ref, self.clipper
        )

    def on_train_end(self, logs: Any | None = None) -> None:
        if self._restore_fn is not None:
            self._restore_fn()
            self._restore_fn = None


__all__ = [
    "SmartClipCallback",
    "patch_optimizer_apply_gradients",
]
