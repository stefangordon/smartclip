from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

from .clipper_base import ClipperBase
from .types import ParamLike, Scope
from .utils import ensure_positive, is_finite_number, safe_ratio

Key = Tuple[str, ...]


class AGC(ClipperBase):
    """Adaptive Gradient Clipping (NFNets-style).

    Scales gradients based on the ratio between gradient norm and parameter (weight)
    norm per group. Given gradient norm ``g`` and weight norm ``w``, the target
    maximum gradient norm is ``T = clipping * (w + eps)`` and the applied scale is::

        scale = min(1.0, T / (g + eps))

    When ``exclude_bias_bn=True`` a simple framework-agnostic heuristic is used to
    skip parameters with dimensionality ``<= 1`` (bias vectors and affine scale
    parameters such as BatchNorm/LN gammas).
    """

    def __init__(
        self,
        *,
        clipping: float = 0.01,
        exclude_bias_bn: bool = True,
        scope: Scope = "per_layer",
        eps: float = 1e-8,
        warmup_steps: int = 0,
        min_history: int = 0,
        guard_nans: bool = True,
    ) -> None:
        super().__init__(
            scope=scope,
            warmup_steps=warmup_steps,
            min_history=min_history,
            eps=eps,
            guard_nans=guard_nans,
        )
        if clipping <= 0.0:
            raise ValueError("clipping must be positive")
        self._clipping = float(clipping)
        self._exclude_bias_bn = bool(exclude_bias_bn)

    # ---------- Properties ----------
    @property
    def clipping(self) -> float:  # pragma: no cover - trivial accessor
        return self._clipping

    @property
    def exclude_bias_bn(self) -> bool:  # pragma: no cover - trivial accessor
        return self._exclude_bias_bn

    # ---------- Algorithm helpers ----------
    def target_norm(self, weight_norm: float) -> float:
        """Return the allowed gradient norm for a given weight norm.

        Computes ``clipping * (weight_norm + eps)`` and lower-bounds the result by ``eps``.
        """

        if not is_finite_number(float(weight_norm)) and self.guard_nans:
            return ensure_positive(0.0, self.eps)
        t = self._clipping * (float(weight_norm) + self.eps)
        return ensure_positive(t, self.eps)

    def scale(self, grad_norm: float, weight_norm: float) -> float:
        """Compute scale factor in [0, 1] for given gradient and weight norms.

        ``scale = min(1, target_norm(weight_norm) / (grad_norm + eps))``
        Non-finite inputs return 1.0 (no scaling) when ``guard_nans`` is True.
        """

        if (
            not is_finite_number(float(grad_norm)) or not is_finite_number(float(weight_norm))
        ) and self.guard_nans:
            return 1.0
        target = self.target_norm(weight_norm)
        ratio = safe_ratio(target, float(grad_norm), self.eps)
        return 1.0 if ratio >= 1.0 else max(ratio, 0.0)

    def should_exclude_param(self, param: ParamLike) -> bool:
        """Return True if a parameter should be excluded from clipping.

        Heuristic: when ``exclude_bias_bn`` is enabled, exclude parameters whose data
        has dimensionality ``<= 1`` (bias vectors and affine scales). If shape cannot
        be determined, do not exclude.
        """

        if not self._exclude_bias_bn:
            return False
        try:
            shape = getattr(param.data, "shape", None)
            if shape is None:
                return False
            try:
                ndim = len(shape)
            except Exception:
                return False
            return ndim <= 1
        except Exception:
            return False

    # ---------- Observation / history gating ----------
    def observe(self, grad_norm: float, weight_norm: float, key: Optional[Key] = None) -> None:
        """Record one AGC observation for warmup/min-history gating.

        Backends should call this once per group measurement prior to applying scaling.
        Non-finite values are ignored when ``guard_nans`` is True.
        """

        if self.guard_nans:
            if not is_finite_number(float(grad_norm)) or not is_finite_number(float(weight_norm)):
                return
        # We do not store history for AGC beyond gating; record a single observation.
        self.record_observation(1)

    # ---------- Serialization ----------
    def state_dict(self) -> Mapping[str, object]:
        base = dict(super().state_dict())
        algo: Dict[str, object] = {
            "clipping": self._clipping,
            "exclude_bias_bn": self._exclude_bias_bn,
        }
        base["agc"] = algo
        return base

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        super().load_state_dict(state)
        algo_obj = state.get("agc")
        if not isinstance(algo_obj, Mapping):
            return  # tolerate missing algorithm state
        c = algo_obj.get("clipping")
        ex = algo_obj.get("exclude_bias_bn")
        if isinstance(c, (float, int)) and float(c) > 0.0:
            self._clipping = float(c)
        if isinstance(ex, bool):
            self._exclude_bias_bn = ex


__all__ = ["AGC"]
