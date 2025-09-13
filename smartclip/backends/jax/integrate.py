from __future__ import annotations

"""JAX/Flax/Optax helpers for integrating smartclip into training loops.

This module provides simple utilities to apply clipping to a gradient pytree and
then pass the result to an Optax-like ``tx.update`` step, or to wrap an
``update`` function for convenience.
"""

from typing import Any, Callable, Tuple


def apply_then_update(
    tx: Any, opt_state: Any, params: Any, grads: Any, clipper: Any
) -> Tuple[Any, Any]:
    """Apply clipping to ``grads`` and run ``tx.update``.

    Returns ``(updates, new_opt_state)``.
    """

    import smartclip.backends.jax as sc_jax  # lazy import

    clipped = sc_jax.apply_grads(grads, params, clipper)
    return tx.update(clipped, opt_state, params)  # type: ignore[no-any-return]


def wrap_tx_update(tx: Any, params_ref: Callable[[], Any], clipper: Any) -> Any:
    """Return a shallow wrapper around ``tx`` that clips grads before update.

    ``params_ref`` should be a zero-arg callable returning the latest params,
    useful in stateful training loops where params live on a TrainState.
    """

    class _Wrapped:
        def __init__(self, _tx: Any) -> None:
            self._tx = _tx

        def update(self, grads: Any, opt_state: Any, params: Any | None = None):  # type: ignore[no-untyped-def]
            import smartclip.backends.jax as sc_jax  # lazy import

            p = params if params is not None else params_ref()
            clipped = sc_jax.apply_grads(grads, p, clipper)
            return self._tx.update(clipped, opt_state, p)

    return _Wrapped(tx)


__all__ = [
    "apply_then_update",
    "wrap_tx_update",
]
