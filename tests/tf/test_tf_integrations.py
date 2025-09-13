from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any


def _install_fake_tf(monkeypatch):
    # Minimal TF surface for our usage
    tf = ModuleType("tensorflow")

    class _CallbackBase:  # noqa: D401 - stub
        """Stub Keras Callback base"""

        pass

    class _CallbacksMod(ModuleType):
        pass

    class _KerasMod(ModuleType):
        pass

    class _OptimizersMod(ModuleType):
        pass

    class _Opt:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def apply_gradients(self, pairs, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(list(pairs))

    keras = _KerasMod("keras")
    callbacks = _CallbacksMod("callbacks")
    optimizers = _OptimizersMod("optimizers")
    callbacks.Callback = _CallbackBase  # type: ignore[attr-defined]
    keras.callbacks = callbacks  # type: ignore[attr-defined]
    keras.optimizers = optimizers  # type: ignore[attr-defined]
    tf.keras = keras  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    return tf, _Opt


def test_keras_callback_patches_and_restores(monkeypatch):
    tf, Opt = _install_fake_tf(monkeypatch)
    mod = importlib.import_module("smartclip.backends.tf.integrate")

    # Build fake model and optimizer
    class _Model:
        def __init__(self) -> None:
            self.trainable_variables = ["w1", "w2"]

    opt = Opt()
    model = _Model()

    # Spy on backend function used by the callback
    import smartclip.backends.tf as sc_tf

    def fake_apply_grads_to_vars(grads, vars_, clipper):  # type: ignore[no-untyped-def]
        # simply pass-through for test, but record that we were called by returning tagged grads
        return [g for g in grads]

    monkeypatch.setattr(sc_tf, "_apply_grads_to_vars", fake_apply_grads_to_vars, raising=True)

    clipper = object()
    cb = mod.SmartClipCallback(model, opt, clipper)

    # Before training begins, no patch
    orig = opt.apply_gradients
    cb.on_train_begin()
    assert opt.apply_gradients is not orig

    # Call patched method and ensure it forwards
    opt.apply_gradients([(1, "w1"), (2, "w2")])
    assert opt.calls and opt.calls[0] == [(1, "w1"), (2, "w2")]

    # Restore on end and ensure wrapper is removed by making backend raise if called
    cb.on_train_end()

    def raise_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("_apply_grads_to_vars should not be called after restore")

    monkeypatch.setattr(sc_tf, "_apply_grads_to_vars", raise_if_called, raising=True)
    # Should use original optimizer method and not call backend helper
    opt.apply_gradients([(3, "w1"), (4, "w2")])
    assert opt.calls[-1] == [(3, "w1"), (4, "w2")]
