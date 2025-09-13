from __future__ import annotations

"""JAX backend public surface.

Provides helpers to clip gradients in JAX/Flax training loops. Users typically
call `apply_grads(grads, params, clipper)` and then feed the result to their
Optax optimizer update.
"""

from contextlib import contextmanager
from typing import Any, Iterator

from ...core import AGC, AutoClip, ZScoreClip
from ...core.types import MetricsRecord, OnMetricsCallback


def _jax() -> tuple[Any, Any]:
    # Local import to avoid importing JAX unless this backend is actively used
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    return jax, jnp


def _tree_l2_norm(tree: Any) -> Any:
    jax, jnp = _jax()
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(tree)))


def apply(model: Any, clipper: AutoClip | AGC | ZScoreClip, on_metrics: OnMetricsCallback | None = None) -> Any:
    # JAX backend requires explicit grads; see apply_grads.
    raise RuntimeError(
        "jax backend requires explicit gradients. Use smartclip.backends.jax.apply_grads(grads, params, clipper)."
    )


def apply_grads(grads: Any, params: Any, clipper: AutoClip | AGC | ZScoreClip, on_metrics: OnMetricsCallback | None = None) -> Any:
    # Global scope: compute a single scale and apply to all leaves
    jax, jnp = _jax()
    if clipper.scope == "global":
        g_leaves = [x for x in jax.tree_util.tree_leaves(grads) if x is not None]
        if g_leaves:
            # Compute global L2 norm across all leaves
            total = sum(jnp.sum(jnp.square(x)) for x in g_leaves)
            g_norm = jnp.sqrt(total)

            # For AGC, we also need global weight norm
            if isinstance(clipper, AGC):
                # Compute global weight norm across all parameter leaves
                p_leaves = [x for x in jax.tree_util.tree_leaves(params) if x is not None]
                p_total = sum(jnp.sum(jnp.square(x)) for x in p_leaves)
                w_norm = jnp.sqrt(p_total)
                clipper.observe(float(g_norm), float(w_norm), key=("global",))
                if clipper.can_clip():
                    s = clipper.scale(float(g_norm), float(w_norm))
                else:
                    s = 1.0
                if on_metrics is not None:
                    rec: MetricsRecord = {
                        "algo": "agc",
                        "key": ("global",),
                        "scope": clipper.scope,
                        "grad_norm": float(g_norm),
                        "weight_norm": float(w_norm),
                        "target": float(clipper.target_norm(float(w_norm))),
                        "scale": float(s),
                        "clipped": bool(s < 1.0),
                    }
                    try:
                        on_metrics(rec)
                    except Exception:
                        pass
                if clipper.can_clip():
                    return jax.tree_map(
                        lambda g: None if g is None else jnp.asarray(s, dtype=g.dtype) * g, grads
                    )
            else:
                clipper.observe(float(g_norm), key=("global",))
                T = clipper.threshold(("global",))
                if clipper.can_clip():
                    denom = float(g_norm) + clipper.eps
                    s = 1.0 if denom <= T else max(T / denom, 0.0)
                else:
                    s = 1.0
                if on_metrics is not None:
                    rec2: MetricsRecord = {
                        "algo": "autoclip" if isinstance(clipper, AutoClip) else "zscore",
                        "key": ("global",),
                        "scope": clipper.scope,
                        "grad_norm": float(g_norm),
                        "threshold": float(T),
                        "scale": float(s),
                        "clipped": bool(s < 1.0),
                    }
                    try:
                        on_metrics(rec2)
                    except Exception:
                        pass
                if clipper.can_clip():
                    return jax.tree_map(
                        lambda g: None if g is None else jnp.asarray(s, dtype=g.dtype) * g, grads
                    )
        return grads

    # AGC: per-leaf using weight norms
    if isinstance(clipper, AGC):

        def scale_leaf(g: Any, p: Any) -> Any:
            if g is None:
                return None
            g_norm = jnp.linalg.norm(g)
            w_norm = jnp.linalg.norm(p)
            clipper.observe(float(g_norm), float(w_norm))
            if clipper.can_clip():
                T = clipper.target_norm(float(w_norm))
                denom = float(g_norm) + clipper.eps
                s = 1.0 if denom <= T else max(T / denom, 0.0)
            else:
                s = 1.0
                T = clipper.target_norm(float(w_norm))
            if on_metrics is not None:
                rec: MetricsRecord = {
                    "algo": "agc",
                    "key": ("param", "*"),
                    "scope": clipper.scope,
                    "grad_norm": float(g_norm),
                    "weight_norm": float(w_norm),
                    "target": float(T),
                    "scale": float(s),
                    "clipped": bool(s < 1.0),
                }
                try:
                    on_metrics(rec)
                except Exception:
                    pass
            return jnp.asarray(s, dtype=g.dtype) * g if clipper.can_clip() else g

        return jax.tree_map(scale_leaf, grads, params)

    # AutoClip/ZScore: per-leaf thresholds with deterministic keys
    leaves, treedef = jax.tree_util.tree_flatten(grads)
    keys = [("param", str(i)) for i in range(len(leaves))]

    def scale_leaf_threshold(g: Any, key: Any) -> Any:
        if g is None:
            return None
        g_norm = jnp.linalg.norm(g)
        clipper.observe(float(g_norm), key=key)
        T = clipper.threshold(key)
        if clipper.can_clip():
            denom = float(g_norm) + clipper.eps
            s = 1.0 if denom <= T else max(T / denom, 0.0)
        else:
            s = 1.0
        if on_metrics is not None:
            rec: MetricsRecord = {
                "algo": "autoclip" if isinstance(clipper, AutoClip) else "zscore",
                "key": key,
                "scope": clipper.scope,
                "grad_norm": float(g_norm),
                "threshold": float(T),
                "scale": float(s),
                "clipped": bool(s < 1.0),
            }
            try:
                on_metrics(rec)
            except Exception:
                pass
        return jnp.asarray(s, dtype=g.dtype) * g if clipper.can_clip() else g

    scaled_leaves = [scale_leaf_threshold(g, k) for g, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, scaled_leaves)


def step(model: Any, optimizer: Any, clipper: AutoClip | AGC | ZScoreClip) -> None:
    raise RuntimeError(
        "jax backend does not patch optimizer.step(). Use apply_grads() with your optimizer update."
    )


@contextmanager
def clip_context(
    model: Any,
    optimizer: Any | None = None,
    clipper: AutoClip | AGC | ZScoreClip | None = None,
    on_metrics: OnMetricsCallback | None = None,
) -> Iterator[None]:
    # Provide a lightweight wrapper for Optax-like optimizers (tx.update(grads, opt_state, params))
    if optimizer is None:
        raise ValueError("optimizer must be provided for JAX clip_context")
    if clipper is None:
        clipper = AutoClip()

    if not hasattr(optimizer, "update"):
        raise RuntimeError("optimizer must expose an update(grads, opt_state, params) method")

    orig_update = optimizer.update

    def wrapped_update(grads, opt_state, params):  # type: ignore[no-untyped-def]
        clipped = apply_grads(grads, params, clipper, on_metrics)
        return orig_update(clipped, opt_state, params)

    optimizer.update = wrapped_update
    try:
        yield None
    finally:
        optimizer.update = orig_update


__all__ = [
    "apply",
    "apply_grads",
    "step",
    "clip_context",
]
