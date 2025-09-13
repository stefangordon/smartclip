from __future__ import annotations

"""PyTorch backend public surface.

This module exposes `apply`, `step`, and `clip_context` for use by the top-level API.
"""

import importlib
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

from ...core import AGC, AutoClip, ZScoreClip
from ...core.grouping import Group
from ...core.types import MetricsRecord, OnMetricsCallback, Scope


def _torch() -> Any:
    # Local import to avoid importing torch unless this backend is actively used
    mod = importlib.import_module("torch")
    return mod


def _iter_params(model: Any) -> list[Any]:
    params: list[Any] = []
    for p in model.parameters():
        if p is not None and p.requires_grad:
            params.append(p)
    return params


def _param_norm(p: Any) -> float:
    # L2 norm of parameter data (weights)
    torch = _torch()
    with torch.no_grad():
        return float(torch.linalg.vector_norm(p.detach()).item())


def _grad_norm(p: Any) -> float:
    g = p.grad
    if g is None:
        return 0.0
    torch = _torch()
    with torch.no_grad():
        return float(torch.linalg.vector_norm(g.detach()).item())


def _scale_grad_(p: Any, scale: float) -> None:
    if p.grad is None:
        return
    if scale == 1.0:
        return
    torch = _torch()
    with torch.no_grad():
        p.grad.mul_(float(scale))


def _group_params(model: Any, scope: Scope) -> List[Group]:
    # Default grouping:
    # - global: single group
    # - per_param: one per parameter
    # - per_layer: group by owning module name when available; fallback to global
    params = _iter_params(model)
    if scope == "global":
        return [Group(key=("global",), params=list(params))]
    if scope == "per_param":
        return [Group(key=("param", str(i)), params=[p]) for i, p in enumerate(params)]

    # per_layer: best-effort map from parameter to its module name by inspecting named_modules and parameters
    layer_map: Dict[Tuple[int, int], str] = {}
    # Build a mapping from parameter object id to module name
    for module_name, module in model.named_modules():
        for idx, p in enumerate(module.parameters(recurse=False)):
            layer_map[(id(module), id(p))] = module_name or "root"

    groups: Dict[str, List[Any]] = {}
    for p in params:
        owner_name = None
        # Try to find owner by scanning modules again for exact param object
        for module_name, module in model.named_modules():
            found = False
            for q in module.parameters(recurse=False):
                if q is p:
                    owner_name = module_name or "root"
                    found = True
                    break
            if found:
                break
        if owner_name is None:
            owner_name = "layer:unknown"
        groups.setdefault(owner_name, []).append(p)

    return [Group(key=("layer", name), params=plist) for name, plist in groups.items()]


def apply(
    model: Any,
    clipper: AutoClip | AGC | ZScoreClip,
    on_metrics: OnMetricsCallback | None = None,
) -> Any:
    """Apply adaptive clipping to gradients on a PyTorch model in-place.

    This function is no-op when gradients are missing.
    """

    torch = _torch()
    groups = _group_params(model, clipper.scope)
    # AGC uses weight norms as well; process per-group
    if isinstance(clipper, AGC):
        for group in groups:
            for p in group.params:
                if clipper.should_exclude_param(p):
                    continue
                g_norm = _grad_norm(p)
                w_norm = _param_norm(p)
                clipper.observe(g_norm, w_norm)
                if clipper.can_clip():
                    s = clipper.scale(g_norm, w_norm)
                    _scale_grad_(p, s)
                else:
                    s = 1.0
                if on_metrics is not None:
                    rec: MetricsRecord = {
                        "algo": "agc",
                        "key": group.key,
                        "scope": clipper.scope,
                        "grad_norm": float(g_norm),
                        "weight_norm": float(w_norm),
                        "target": float(clipper.target_norm(w_norm)),
                        "scale": float(s),
                        "clipped": bool(s < 1.0),
                    }
                    try:
                        on_metrics(rec)
                    except Exception:
                        pass
        return model

    # AutoClip and ZScore: compute per-group gradient norms and scale accordingly
    # First pass: compute norms and record observations
    group_norms: List[Tuple[Tuple[str, ...], float, List[Any]]] = []
    for group in groups:
        if clipper.scope == "global":
            # Use combined norm for the group
            with torch.no_grad():
                total = 0.0
                for p in group.params:
                    if p.grad is None:
                        continue
                    g_flat = p.grad.detach()
                    total += float(torch.dot(g_flat.reshape(-1), g_flat.reshape(-1)).item())
                g_norm = float(total) ** 0.5
        else:
            # For layer/param, treat each param independently as a group of one when per_param
            if group.key[0] == "param":
                p = group.params[0]
                g_norm = _grad_norm(p)
            else:
                # per_layer: sum norms over parameters in the layer
                with torch.no_grad():
                    total = 0.0
                    for p in group.params:
                        if p.grad is None:
                            continue
                        g_flat = p.grad.detach()
                        total += float(torch.dot(g_flat.reshape(-1), g_flat.reshape(-1)).item())
                    g_norm = float(total) ** 0.5
        group_norms.append((group.key, g_norm, group.params))
        if isinstance(clipper, AutoClip):
            clipper.observe(g_norm, key=group.key)
        else:
            clipper.observe(g_norm, key=group.key)

    # Second pass: scale each group's parameters by the group's threshold
    for key, g_norm, params in group_norms:
        if not clipper.can_clip():
            s = 1.0
            T = clipper.threshold(key)  # may be eps early on
        else:
            T = clipper.threshold(key)
            denom = g_norm + clipper.eps
            s = 1.0 if denom <= T else max(T / denom, 0.0)
        if clipper.scope == "per_param":
            # params has length 1
            _scale_grad_(params[0], s)
        else:
            for p in params:
                _scale_grad_(p, s)
        if on_metrics is not None:
            rec2: MetricsRecord = {
                "algo": "autoclip" if isinstance(clipper, AutoClip) else "zscore",
                "key": key,
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
    return model


def step(
    model: Any,
    optimizer: Any,
    clipper: AutoClip | AGC | ZScoreClip,
    on_metrics: OnMetricsCallback | None = None,
) -> Any:
    apply(model, clipper, on_metrics)
    return optimizer.step()


@contextmanager
def clip_context(
    model: Any,
    optimizer: Any | None = None,
    clipper: AutoClip | AGC | ZScoreClip | None = None,
    on_metrics: OnMetricsCallback | None = None,
) -> Iterator[None]:
    """Context manager that clips before each optimizer.step().

    Implementation strategy: monkey-patch the provided optimizer's step() with a
    wrapper that calls apply() first. This is local to the context.
    """

    if optimizer is None:
        raise ValueError("optimizer must be provided for PyTorch clip_context")
    if clipper is None:
        clipper = AutoClip()

    orig_step = optimizer.step

    def wrapped_step(*args, **kwargs):  # type: ignore[no-untyped-def]
        apply(model, clipper, on_metrics)  # clip in-place
        return orig_step(*args, **kwargs)

    optimizer.step = wrapped_step
    try:
        yield None
    finally:
        optimizer.step = orig_step


__all__ = [
    "apply",
    "step",
    "clip_context",
]
