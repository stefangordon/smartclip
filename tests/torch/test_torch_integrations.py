from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any


def _install_fake_torch(monkeypatch):
    torch = ModuleType("torch")

    class _NoGrad:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return None

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

    linalg = ModuleType("torch.linalg")

    def vector_norm(x):  # type: ignore[no-untyped-def]
        class _T:
            def item(self):  # type: ignore[no-untyped-def]
                return 0.0

        return _T()

    linalg.vector_norm = vector_norm  # type: ignore[attr-defined]
    torch.linalg = linalg  # type: ignore[attr-defined]

    def no_grad():  # type: ignore[no-untyped-def]
        return _NoGrad()

    torch.no_grad = no_grad  # type: ignore[attr-defined]

    nn = ModuleType("torch.nn")

    class Parameter:  # noqa: D401 - stub
        """Stub Parameter"""

        def __init__(self):  # type: ignore[no-untyped-def]
            self.grad = None
            self.requires_grad = True

        def detach(self):  # type: ignore[no-untyped-def]
            return self

    nn.Parameter = Parameter  # type: ignore[attr-defined]
    torch.nn = nn  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.linalg", linalg)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)


def _install_fake_lightning(monkeypatch):
    # Create minimal module tree: lightning.pytorch.callbacks.Callback
    lightning = ModuleType("lightning")
    pytorch = ModuleType("lightning.pytorch")
    callbacks = ModuleType("lightning.pytorch.callbacks")

    class Callback:  # noqa: D401 - minimal stub
        """Stub Callback base"""

        pass

    callbacks.Callback = Callback  # type: ignore[attr-defined]
    pytorch.callbacks = callbacks  # type: ignore[attr-defined]
    lightning.pytorch = pytorch  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "lightning", lightning)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", pytorch)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.callbacks", callbacks)


def _install_fake_transformers(monkeypatch):
    transformers = ModuleType("transformers")

    class TrainerCallback:  # noqa: D401 - minimal stub
        """Stub TrainerCallback base"""

        pass

    transformers.TrainerCallback = TrainerCallback  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_lightning_callback_invokes_apply(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_lightning(monkeypatch)
    _install_fake_transformers(monkeypatch)

    # Import after fakes are installed so class bases resolve to stubs
    mod = importlib.import_module("smartclip.backends.torch.integrate")

    calls: list[tuple[Any, Any]] = []

    # Patch smartclip.apply to record invocations
    import smartclip as sc

    def fake_apply(model, clipper):  # type: ignore[no-untyped-def]
        calls.append((model, clipper))
        return model

    monkeypatch.setattr(sc, "apply", fake_apply, raising=True)

    clipper = object()
    model = object()
    optimizer = object()
    cb = mod.SmartClipCallback(clipper)

    # Simulate PL hook
    cb.on_before_optimizer_step(trainer=object(), pl_module=model, optimizer=optimizer)

    assert calls == [(model, clipper)]


def test_hf_trainer_callback_invokes_apply(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_lightning(monkeypatch)
    _install_fake_transformers(monkeypatch)
    mod = importlib.import_module("smartclip.backends.torch.integrate")

    calls: list[tuple[Any, Any]] = []
    import smartclip as sc

    def fake_apply(model, clipper):  # type: ignore[no-untyped-def]
        calls.append((model, clipper))
        return model

    monkeypatch.setattr(sc, "apply", fake_apply, raising=True)

    clipper = object()
    model = object()
    cb = mod.SmartClipTrainerCallback(clipper)

    # Simulate Transformers hook with model in kwargs
    cb.on_optimizer_step(args=None, state=None, control=None, model=model)

    assert calls == [(model, clipper)]
