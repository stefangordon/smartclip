from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

from .clipper_base import ClipperBase
from .stats import EmaMoments
from .types import Scope
from .utils import ensure_positive, is_finite_number

Key = Tuple[str, ...]


class ZScoreClip(ClipperBase):
    """Z-score based adaptive clipping using EMA mean/variance.

    Tracks exponentially-weighted moving averages of the observed gradient norm
    (``m``) and squared norm (``m2``) per grouping key. The standard deviation is
    computed as ``sqrt(max(0, m2 - m^2))``. For a new observation with norm ``g``,
    the Z-score is ``z = (g - m) / (std + eps)`` and clipping is recommended when
    ``z > zmax``. Backends typically implement clipping by scaling gradients by
    ``min(1, T / (g + eps))`` where the threshold ``T = m + zmax * std``.
    """

    def __init__(
        self,
        *,
        zmax: float = 3.0,
        ema_decay: float = 0.99,
        scope: Scope = "global",
        warmup_steps: int = 100,
        min_history: int = 50,
        eps: float = 1e-8,
        guard_nans: bool = True,
    ) -> None:
        super().__init__(
            scope=scope,
            warmup_steps=warmup_steps,
            min_history=min_history,
            eps=eps,
            guard_nans=guard_nans,
        )
        if zmax <= 0.0:
            raise ValueError("zmax must be positive")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        self._zmax = float(zmax)
        self._ema_decay = float(ema_decay)
        self._moments_by_key: Dict[Key, EmaMoments] = {}

    # ---------- Observation / updates ----------
    def observe(self, value: float, key: Optional[Key] = None) -> None:
        """Observe a gradient norm for a grouping key.

        Backends should call this once per measured norm (global/layer/param)
        before applying clipping. Values that are non-finite are ignored when
        ``guard_nans`` is True.

        Args:
            value: Gradient norm (L2 norm of gradients for this group).
            key: Grouping key tuple. Examples:
                 - ("global",) for global scope
                 - ("layer", "conv1") for per-layer scope
                 - ("param", "0") for per-parameter scope
                 Defaults to ("global",) if None.
        """

        if not is_finite_number(float(value)):
            if self.guard_nans:
                return
        v = ensure_positive(float(value), self.eps)
        k = key if key is not None else ("global",)

        mom = self._moments_by_key.setdefault(k, EmaMoments(decay=self._ema_decay))
        mom.update(v)

        # Count this observation towards history/warmup gating
        self.record_observation(1)

    # ---------- Threshold queries ----------
    def threshold(self, key: Optional[Key] = None) -> float:
        """Return current z-score threshold ``m + zmax * std`` for a key.

        This does not enforce warmup/min-history gates. Callers should check
        ``can_clip()`` to decide whether clipping should be applied.

        Args:
            key: Grouping key tuple (e.g., ("global",), ("layer", "conv1")).
                 Defaults to ("global",) if None.

        Returns:
            Current threshold, lower-bounded by eps.
        """

        k = key if key is not None else ("global",)
        mom = self._moments_by_key.get(k)
        if mom is None:
            return ensure_positive(0.0, self.eps)
        m = mom.mean()
        s = mom.std()
        if m is None or s is None:
            return ensure_positive(0.0, self.eps)
        t = float(m) + self._zmax * float(s)
        return ensure_positive(t, self.eps)

    def threshold_any(self) -> float:
        """Convenience method for callers that do not track keys.

        Returns the global threshold if available, otherwise the threshold
        for the single key if exactly one exists, otherwise eps.
        """
        if ("global",) in self._moments_by_key:
            return self.threshold()
        if len(self._moments_by_key) == 1:
            only_key = next(iter(self._moments_by_key.keys()))
            return self.threshold(only_key)
        return ensure_positive(0.0, self.eps)

    # ---------- Statistics ----------
    def stats(self, key: Optional[Key] = None) -> tuple[float, float]:
        """Return the current ``(mean, std)`` estimates for a key.

        If the key has not been observed, or estimates are uninitialized,
        returns ``(0.0, 0.0)``.
        """

        k = key if key is not None else ("global",)
        mom = self._moments_by_key.get(k)
        if mom is None:
            return 0.0, 0.0
        m = mom.mean()
        s = mom.std()
        return (float(m) if m is not None else 0.0, float(s) if s is not None else 0.0)

    # ---------- Serialization ----------
    def state_dict(self) -> Mapping[str, object]:
        base = dict(super().state_dict())
        algo: Dict[str, object] = {
            "zmax": self._zmax,
            "ema_decay": self._ema_decay,
            "keys": [list(k) for k in self._moments_by_key.keys()],
            "mean": [m.mean() for m in self._moments_by_key.values()],
            "var": [m.variance() for m in self._moments_by_key.values()],
        }
        base["zscore"] = algo
        return base

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        super().load_state_dict(state)
        algo_obj = state.get("zscore")
        if not isinstance(algo_obj, Mapping):
            return  # tolerate missing algorithm state
        z = algo_obj.get("zmax")
        d = algo_obj.get("ema_decay")
        if isinstance(z, (float, int)) and float(z) > 0.0:
            self._zmax = float(z)
        if isinstance(d, (float, int)) and 0.0 < float(d) < 1.0:
            self._ema_decay = float(d)

        self._moments_by_key.clear()
        keys = algo_obj.get("keys")
        means = algo_obj.get("mean")
        vars_ = algo_obj.get("var")
        if isinstance(keys, list) and isinstance(means, list) and isinstance(vars_, list):
            for k_list, m_val, v_val in zip(keys, means, vars_):
                if not isinstance(k_list, list):
                    continue
                key_tuple: Key = tuple(str(s) for s in k_list)
                init_mean = float(m_val) if isinstance(m_val, (float, int)) else None
                init_var = float(v_val) if isinstance(v_val, (float, int)) else 0.0
                self._moments_by_key[key_tuple] = EmaMoments(
                    decay=self._ema_decay,
                    initial_mean=init_mean,
                    initial_var=init_var,
                )


__all__ = ["ZScoreClip"]
