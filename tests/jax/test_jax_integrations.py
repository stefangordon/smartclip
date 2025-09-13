from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any


def _install_fake_jax(monkeypatch):
    jax = ModuleType("jax")
    jnp = ModuleType("jax.numpy")

    # Minimal functions used in backend module definitions
    def _identity(x):  # type: ignore[no-untyped-def]
        return x

    jnp.sqrt = _identity  # type: ignore[attr-defined]
    jnp.sum = _identity  # type: ignore[attr-defined]
    jnp.square = _identity  # type: ignore[attr-defined]

    linalg = ModuleType("jax.numpy.linalg")

    def _norm(x):  # type: ignore[no-untyped-def]
        return 0.0

    linalg.norm = _norm  # type: ignore[attr-defined]
    jnp.linalg = linalg  # type: ignore[attr-defined]

    tree_util = ModuleType("jax.tree_util")

    def tree_leaves(tree):  # type: ignore[no-untyped-def]
        return []

    def tree_flatten(tree):  # type: ignore[no-untyped-def]
        return [], object()

    def tree_unflatten(treedef, leaves):  # type: ignore[no-untyped-def]
        return leaves

    tree_util.tree_leaves = tree_leaves  # type: ignore[attr-defined]
    tree_util.tree_flatten = tree_flatten  # type: ignore[attr-defined]
    tree_util.tree_unflatten = tree_unflatten  # type: ignore[attr-defined]

    # Register modules
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)
    monkeypatch.setitem(sys.modules, "jax.numpy.linalg", linalg)
    monkeypatch.setitem(sys.modules, "jax.tree_util", tree_util)


def test_apply_then_update_calls_backend(monkeypatch):
    _install_fake_jax(monkeypatch)
    mod = importlib.import_module("smartclip.backends.jax.integrate")

    calls: dict[str, Any] = {"apply_grads": None, "update": None}

    # Provide fake backend apply_grads
    import smartclip.backends.jax as sc_jax

    def fake_apply_grads(grads, params, clipper):  # type: ignore[no-untyped-def]
        calls["apply_grads"] = (grads, params, clipper)
        return {"g": 42}

    monkeypatch.setattr(sc_jax, "apply_grads", fake_apply_grads, raising=True)

    class _Tx:
        def update(self, clipped, opt_state, params):  # type: ignore[no-untyped-def]
            calls["update"] = (clipped, opt_state, params)
            return ("updates", "new_state")

    tx = _Tx()
    out = mod.apply_then_update(tx, opt_state="S", params="P", grads="G", clipper="C")
    assert out == ("updates", "new_state")
    assert calls["apply_grads"] == ("G", "P", "C")
    assert calls["update"] == ({"g": 42}, "S", "P")


def test_wrap_tx_update_clips_before_update(monkeypatch):
    _install_fake_jax(monkeypatch)
    mod = importlib.import_module("smartclip.backends.jax.integrate")

    import smartclip.backends.jax as sc_jax

    def fake_apply_grads(grads, params, clipper):  # type: ignore[no-untyped-def]
        return {"g": (grads, params, clipper)}

    monkeypatch.setattr(sc_jax, "apply_grads", fake_apply_grads, raising=True)

    class _Tx:
        def update(self, clipped, opt_state, params):  # type: ignore[no-untyped-def]
            return (clipped, opt_state, params)

    params_store = {"p": 0}

    def params_ref():  # type: ignore[no-untyped-def]
        return params_store

    tx = _Tx()
    wrapped = mod.wrap_tx_update(tx, params_ref, clipper="C")
    out = wrapped.update(grads="G", opt_state="S", params=None)
    assert out == (({"g": ("G", params_store, "C")}), "S", params_store)
