from __future__ import annotations

"""TensorFlow backend public surface.

Expose functions compatible with the top-level API. This backend provides a minimal
implementation for custom training loops. Keras `Model.fit` users should prefer
calling `apply` inside a custom train_step.
"""

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, List, Sequence, Tuple

from ...core import AGC, AutoClip, ZScoreClip
from ...core.types import MetricsRecord, OnMetricsCallback


def _tf() -> Any:
    # Local import to avoid importing TensorFlow unless this backend is actively used
    import tensorflow as tf  # type: ignore

    return tf


def _iter_trainable_vars(model: Any) -> List[Any]:
    vars_: List[Any] = []
    for v in getattr(model, "trainable_variables", []):
        if v is not None:
            vars_.append(v)
    return vars_


def apply(
    model: Any, clipper: AutoClip | AGC | ZScoreClip, on_metrics: OnMetricsCallback | None = None
) -> Any:
    """Apply clipping to gradients stored on the model via tape.gradient workflows.

    TensorFlow does not store gradients on variables by default. This function is
    a light wrapper which expects you to provide gradients separately using
    `apply_grads`.
    """

    raise RuntimeError(
        "tf backend requires explicit gradients. Use smartclip.backends.tf.apply_grads(grads, model, clipper)."
    )


def _apply_grads_to_vars(
    grads: Sequence[Any],
    variables: Sequence[Any],
    clipper: AutoClip | AGC | ZScoreClip,
    on_metrics: OnMetricsCallback | None = None,
) -> List[Any]:
    tf = _tf()
    clipped: list[Any] = []
    if isinstance(clipper, AGC):
        for g, v in zip(grads, variables):
            if g is None:
                clipped.append(None)
                continue
            if clipper.should_exclude_param(v):
                clipped.append(g)
                continue
            g_norm = tf.linalg.global_norm([g])
            w_norm = tf.linalg.global_norm([v])
            clipper.observe(float(g_norm.numpy()), float(w_norm.numpy()))
            if clipper.can_clip():
                T = clipper.target_norm(float(w_norm.numpy()))
                denom = float(g_norm.numpy()) + clipper.eps
                s = 1.0 if denom <= T else max(T / denom, 0.0)
                clipped.append(tf.cast(s, g.dtype) * g)
            else:
                s = 1.0
                clipped.append(g)
            if on_metrics is not None:
                rec: MetricsRecord = {
                    "algo": "agc",
                    "key": ("param", str(id(v))),
                    "scope": clipper.scope,
                    "grad_norm": float(g_norm.numpy()),
                    "weight_norm": float(w_norm.numpy()),
                    "target": float(clipper.target_norm(float(w_norm.numpy()))),
                    "scale": float(s),
                    "clipped": bool(s < 1.0),
                }
                try:
                    on_metrics(rec)
                except Exception:
                    pass
        return clipped

    # AutoClip and ZScore use gradient norm thresholding
    for g, v in zip(grads, variables):
        if g is None:
            clipped.append(None)
            continue
        g_norm = tf.linalg.global_norm([g])
        key = ("param", str(id(v)))
        if isinstance(clipper, AutoClip):
            clipper.observe(float(g_norm.numpy()), key=key)
        else:  # ZScoreClip
            clipper.observe(float(g_norm.numpy()), key=key)
        if clipper.can_clip():
            T = clipper.threshold(key)
            denom = float(g_norm.numpy()) + clipper.eps
            s = 1.0 if denom <= T else max(T / denom, 0.0)
            clipped.append(tf.cast(s, g.dtype) * g)
        else:
            s = 1.0
            T = clipper.threshold(key)
            clipped.append(g)
        if on_metrics is not None:
            rec2: MetricsRecord = {
                "algo": "autoclip" if isinstance(clipper, AutoClip) else "zscore",
                "key": key,
                "scope": clipper.scope,
                "grad_norm": float(g_norm.numpy()),
                "threshold": float(T),
                "scale": float(s),
                "clipped": bool(s < 1.0),
            }
            try:
                on_metrics(rec2)
            except Exception:
                pass
    return clipped


def apply_grads(
    grads: Sequence[Any],
    model: Any,
    clipper: AutoClip | AGC | ZScoreClip,
    on_metrics: OnMetricsCallback | None = None,
) -> List[Any]:
    tf = _tf()
    variables = _iter_trainable_vars(model)
    if clipper.scope == "global":
        # Compute global gradient norm across all provided grads
        non_none = [g for g in grads if g is not None]
        if non_none:
            g_norm = tf.linalg.global_norm(non_none)

            # For AGC, we also need global weight norm
            if isinstance(clipper, AGC):
                # Compute global weight norm across all variables
                # Handle both v.value() method and v.value property across TF versions
                w_vals = []
                for v in variables:
                    val = getattr(v, 'value', v)
                    w_vals.append(val() if callable(val) else val)
                w_norm = tf.linalg.global_norm(w_vals)
                clipper.observe(float(g_norm.numpy()), float(w_norm.numpy()), key=("global",))
                if clipper.can_clip():
                    s = clipper.scale(float(g_norm.numpy()), float(w_norm.numpy()))
                else:
                    s = 1.0
                if on_metrics is not None:
                    rec: MetricsRecord = {
                        "algo": "agc",
                        "key": ("global",),
                        "scope": clipper.scope,
                        "grad_norm": float(g_norm.numpy()),
                        "weight_norm": float(w_norm.numpy()),
                        "target": float(clipper.target_norm(float(w_norm.numpy()))),
                        "scale": float(s),
                        "clipped": bool(s < 1.0),
                    }
                    try:
                        on_metrics(rec)
                    except Exception:
                        pass
                if clipper.can_clip():
                    clipped = [tf.cast(s, g.dtype) * g if g is not None else None for g in grads]
                    return clipped
            else:
                clipper.observe(float(g_norm.numpy()), key=("global",))
                T = clipper.threshold(("global",))
                if clipper.can_clip():
                    denom = float(g_norm.numpy()) + clipper.eps
                    s = 1.0 if denom <= T else max(T / denom, 0.0)
                else:
                    s = 1.0
                if on_metrics is not None:
                    rec2: MetricsRecord = {
                        "algo": "autoclip" if isinstance(clipper, AutoClip) else "zscore",
                        "key": ("global",),
                        "scope": clipper.scope,
                        "grad_norm": float(g_norm.numpy()),
                        "threshold": float(T),
                        "scale": float(s),
                        "clipped": bool(s < 1.0),
                    }
                    try:
                        on_metrics(rec2)
                    except Exception:
                        pass
                if clipper.can_clip():
                    clipped = [tf.cast(s, g.dtype) * g if g is not None else None for g in grads]
                    return clipped
    # per_param and per_layer handled by per-variable thresholding
    return _apply_grads_to_vars(grads, variables, clipper, on_metrics)


def step(model: Any, optimizer: Any, clipper: AutoClip | AGC | ZScoreClip) -> None:
    raise RuntimeError(
        "tf backend does not patch optimizer.step(). Use apply_grads() and optimizer.apply_gradients()."
    )


@contextmanager
def clip_context(
    model: Any,
    optimizer: Any | None = None,
    clipper: AutoClip | AGC | ZScoreClip | None = None,
    on_metrics: OnMetricsCallback | None = None,
) -> Iterator[None]:
    tf = _tf()
    if optimizer is None:
        raise ValueError("optimizer must be provided for TensorFlow clip_context")
    if clipper is None:
        clipper = AutoClip()

    orig_apply = optimizer.apply_gradients

    def wrapped_apply_gradients(grads_and_vars: Iterable[Tuple[Any, Any]], *args, **kwargs):  # type: ignore[no-untyped-def]
        # Materialize sequence in case it's a generator
        pairs = list(grads_and_vars)
        grads: list[Any] = [g for g, _ in pairs]
        vars_: list[Any] = [v for _, v in pairs]
        if clipper.scope == "global":
            non_none = [g for g in grads if g is not None]
            if non_none:
                g_norm = tf.linalg.global_norm(non_none)

                # For AGC, we also need global weight norm
                if isinstance(clipper, AGC):
                    # Compute global weight norm across all variables
                    # Handle both v.value() method and v.value property across TF versions
                    w_vals = []
                    for v in vars_:
                        val = getattr(v, 'value', v)
                        w_vals.append(val() if callable(val) else val)
                    w_norm = tf.linalg.global_norm(w_vals)
                    clipper.observe(float(g_norm.numpy()), float(w_norm.numpy()), key=("global",))
                    if clipper.can_clip():
                        s = clipper.scale(float(g_norm.numpy()), float(w_norm.numpy()))
                    else:
                        s = 1.0
                    if on_metrics is not None:
                        rec: MetricsRecord = {
                            "algo": "agc",
                            "key": ("global",),
                            "scope": clipper.scope,
                            "grad_norm": float(g_norm.numpy()),
                            "weight_norm": float(w_norm.numpy()),
                            "target": float(clipper.target_norm(float(w_norm.numpy()))),
                            "scale": float(s),
                            "clipped": bool(s < 1.0),
                        }
                        try:
                            on_metrics(rec)
                        except Exception:
                            pass
                    if clipper.can_clip():
                        clipped = [
                            tf.cast(s, g.dtype) * g if g is not None else None for g in grads
                        ]
                        new_pairs = [(cg, v) for cg, v in zip(clipped, vars_)]
                        return orig_apply(new_pairs, *args, **kwargs)
                else:
                    clipper.observe(float(g_norm.numpy()), key=("global",))
                    T = clipper.threshold(("global",))
                    if clipper.can_clip():
                        denom = float(g_norm.numpy()) + clipper.eps
                        s = 1.0 if denom <= T else max(T / denom, 0.0)
                    else:
                        s = 1.0
                    if on_metrics is not None:
                        rec2: MetricsRecord = {
                            "algo": "autoclip" if isinstance(clipper, AutoClip) else "zscore",
                            "key": ("global",),
                            "scope": clipper.scope,
                            "grad_norm": float(g_norm.numpy()),
                            "threshold": float(T),
                            "scale": float(s),
                            "clipped": bool(s < 1.0),
                        }
                        try:
                            on_metrics(rec2)
                        except Exception:
                            pass
                    if clipper.can_clip():
                        clipped = [
                            tf.cast(s, g.dtype) * g if g is not None else None for g in grads
                        ]
                        new_pairs = [(cg, v) for cg, v in zip(clipped, vars_)]
                        return orig_apply(new_pairs, *args, **kwargs)
        clipped = _apply_grads_to_vars(grads, vars_, clipper, on_metrics)
        new_pairs = [(cg, v) for cg, v in zip(clipped, vars_)]
        return orig_apply(new_pairs, *args, **kwargs)

    optimizer.apply_gradients = wrapped_apply_gradients
    try:
        yield None
    finally:
        optimizer.apply_gradients = orig_apply


__all__ = [
    "apply",
    "apply_grads",
    "step",
    "clip_context",
]
