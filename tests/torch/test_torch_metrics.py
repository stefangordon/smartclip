from __future__ import annotations

import sys
from types import ModuleType
from typing import List


def _install_fake_torch(monkeypatch):
    torch = ModuleType("torch")

    class _NoGrad:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return None

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

    linalg = ModuleType("torch.linalg")

    class _T:
        def item(self):  # type: ignore[no-untyped-def]
            return 0.0

    def vector_norm(x):  # type: ignore[no-untyped-def]
        return _T()

    linalg.vector_norm = vector_norm  # type: ignore[attr-defined]
    torch.linalg = linalg  # type: ignore[attr-defined]

    def no_grad():  # type: ignore[no-untyped-def]
        return _NoGrad()

    torch.no_grad = no_grad  # type: ignore[attr-defined]

    nn = ModuleType("torch.nn")

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.linalg", linalg)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)


class _Grad:
    def detach(self):  # type: ignore[no-untyped-def]
        return self


class _Param:
    def __init__(self) -> None:
        self.grad = _Grad()
        self.requires_grad = True

    def detach(self):  # type: ignore[no-untyped-def]
        return self


class _Model:
    def __init__(self, n: int = 2) -> None:
        self._params: List[_Param] = [_Param() for _ in range(n)]

    def parameters(self):  # type: ignore[no-untyped-def]
        return self._params

    def named_modules(self):  # type: ignore[no-untyped-def]
        return []


# Make _Model appear to be from torch module for backend detection
_Model.__module__ = "torch.nn"


def test_torch_apply_emits_metrics(monkeypatch):
    _install_fake_torch(monkeypatch)

    import smartclip as sc

    model = _Model(n=3)
    clipper = sc.AutoClip(
        mode="percentile",
        percentile=90.0,
        history="ema",
        warmup_steps=0,
        min_history=0,
        scope="per_param",
    )

    records: List[dict] = []

    def on_metrics(rec: dict) -> None:
        records.append(rec)

    sc.apply(model, clipper, on_metrics=on_metrics)

    # Should emit one record per parameter (per_param scope)
    assert len(records) == 3
    # Schema smoke checks
    for r in records:
        assert "algo" in r
        assert "key" in r and isinstance(r["key"], tuple)
        assert "grad_norm" in r
        assert "scale" in r
