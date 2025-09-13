from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

from .clipper_base import ClipperBase
from .stats import EmaQuantile, P2Quantile, RollingWindow, WelfordVariance
from .types import Scope
from .utils import ensure_positive, is_finite_number

Key = Tuple[str, ...]


class AutoClip(ClipperBase):
    """Adaptive clipping of gradients.

    Modes:
    - "auto" (default): hyperparameter-free threshold using P² median (p=0.5)
      and Welford variance: T = median + 3 * std.
    - "percentile": target percentile of recent gradient norms using either
      EMA quantile estimator (``history="ema"``) or rolling window (``history="window"``).
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        percentile: float = 95.0,
        history: str = "ema",
        ema_decay: float = 0.99,
        window_size: int = 1024,
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
        if mode not in ("auto", "percentile"):
            raise ValueError("mode must be 'auto' or 'percentile'")
        if not (0.0 < percentile <= 100.0):
            raise ValueError("percentile must be in (0, 100]")
        if history not in ("ema", "window"):
            raise ValueError("history must be 'ema' or 'window'")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._mode = mode
        self._percentile = float(percentile)
        self._history_mode = history
        self._ema_decay = float(ema_decay)
        self._window_size = int(window_size)

        # Per-key estimators
        self._ema_by_key: Dict[Key, EmaQuantile] = {}
        self._win_by_key: Dict[Key, RollingWindow] = {}
        self._p2_by_key: Dict[Key, P2Quantile] = {}
        self._welford_by_key: Dict[Key, WelfordVariance] = {}

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

        if self._mode == "auto":
            med = self._p2_by_key.setdefault(k, P2Quantile(p=0.5))
            med.update(v)
            w = self._welford_by_key.setdefault(k, WelfordVariance())
            w.update(v)
        else:  # percentile mode
            if self._history_mode == "ema":
                # Convert percentile to quantile in (0, 1)
                p = self._percentile / 100.0
                alpha = 1.0 - self._ema_decay
                est = self._ema_by_key.setdefault(k, EmaQuantile(p=p, alpha=alpha))
                est.update(v)
            else:  # window
                win = self._win_by_key.setdefault(k, RollingWindow(size=self._window_size))
                win.append(v)

        # Count this observation towards history/warmup gating
        self.record_observation(1)

    # ---------- Threshold queries ----------
    def threshold(self, key: Optional[Key] = None) -> float:
        """Return current threshold for a key (default: global).

        This does not enforce warmup/min-history gates. Callers should check
        ``can_clip()`` to decide whether clipping should be applied.

        Args:
            key: Grouping key tuple (e.g., ("global",), ("layer", "conv1")).
                 Defaults to ("global",) if None.

        Returns:
            Current percentile threshold, lower-bounded by eps.
        """

        k = key if key is not None else ("global",)
        t = 0.0
        if self._mode == "auto":
            med = self._p2_by_key.get(k)
            w = self._welford_by_key.get(k)
            median = med.value if med is not None else None
            std = w.std if w is not None else None
            if median is not None and std is not None:
                t = float(median + 3.0 * std)
            elif median is not None:
                # Have median but no std yet (< 2 observations for Welford)
                # Use a more conservative threshold than eps but less than aggressive clipping
                t = float(median * 2.0)
        else:
            if self._history_mode == "ema":
                est = self._ema_by_key.get(k)
                if est is not None and est.value is not None:
                    t = float(est.value)
            else:
                win = self._win_by_key.get(k)
                if win is not None and len(win) > 0:
                    t = float(win.percentile(self._percentile))
        return ensure_positive(t, self.eps)

    def threshold_any(self) -> float:
        """Convenience method for callers that do not track keys.

        Returns the global threshold if available, otherwise the threshold
        for the single key if exactly one exists, otherwise eps.
        """
        if self._mode == "auto":
            # Check for global key first
            if ("global",) in self._p2_by_key:
                return self.threshold()
            # Fallback to single key if exactly one exists
            if len(self._p2_by_key) == 1:
                only_key = next(iter(self._p2_by_key.keys()))
                return self.threshold(only_key)
            return ensure_positive(0.0, self.eps)

        if ("global",) in self._ema_by_key or ("global",) in self._win_by_key:
            return self.threshold()
        # Probe single-key case
        if self._history_mode == "ema":
            if len(self._ema_by_key) == 1:
                only_key = next(iter(self._ema_by_key.keys()))
                return self.threshold(only_key)
        else:  # window
            if len(self._win_by_key) == 1:
                only_key = next(iter(self._win_by_key.keys()))
                return self.threshold(only_key)
        return ensure_positive(0.0, self.eps)

    # ---------- Serialization ----------
    def state_dict(self) -> Mapping[str, object]:
        base = dict(super().state_dict())
        algo: Dict[str, object] = {
            "mode": self._mode,
            "percentile": self._percentile,
            "history": self._history_mode,
            "ema_decay": self._ema_decay,
            "window_size": self._window_size,
        }
        if self._mode == "auto":
            algo["auto"] = {
                "keys": [list(k) for k in self._p2_by_key.keys()],
                "p2": [list(e.get_state()) for e in self._p2_by_key.values()],
                "welford": [
                    list(self._welford_by_key[k].get_state()) for k in self._p2_by_key.keys()
                ],
            }
        elif self._history_mode == "ema":
            algo["ema"] = {
                "keys": [list(k) for k in self._ema_by_key.keys()],
                "q": [e.value if e.value is not None else None for e in self._ema_by_key.values()],
                "count": [e.count for e in self._ema_by_key.values()],
            }
        else:
            algo["window"] = {
                "keys": [list(k) for k in self._win_by_key.keys()],
                "values": [w.to_list() for w in self._win_by_key.values()],
            }
        base["autoclip"] = algo
        return base

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        super().load_state_dict(state)
        algo_obj = state.get("autoclip")
        if not isinstance(algo_obj, Mapping):
            return  # tolerate missing algorithm state
        m = algo_obj.get("mode")
        p = algo_obj.get("percentile")
        h = algo_obj.get("history")
        ed = algo_obj.get("ema_decay")
        ws = algo_obj.get("window_size")
        if isinstance(m, str) and m in ("auto", "percentile"):
            self._mode = m
        if isinstance(p, (float, int)) and 0.0 < float(p) <= 100.0:
            self._percentile = float(p)
        if isinstance(h, str) and h in ("ema", "window"):
            self._history_mode = h
        if isinstance(ed, (float, int)) and 0.0 < float(ed) < 1.0:
            self._ema_decay = float(ed)
        if isinstance(ws, int) and ws > 0:
            self._window_size = int(ws)

        self._ema_by_key.clear()
        self._win_by_key.clear()
        self._p2_by_key.clear()
        self._welford_by_key.clear()
        if self._mode == "auto":
            auto_obj = algo_obj.get("auto")
            if isinstance(auto_obj, Mapping):
                keys = auto_obj.get("keys")
                p2_list = auto_obj.get("p2")
                w_list = auto_obj.get("welford")
                if (
                    isinstance(keys, list)
                    and isinstance(p2_list, list)
                    and isinstance(w_list, list)
                ):
                    for k_list, p2_state, w_state in zip(keys, p2_list, w_list):
                        if (
                            not isinstance(k_list, list)
                            or not isinstance(p2_state, list)
                            or not isinstance(w_state, list)
                        ):
                            continue
                        key_tuple: Key = tuple(str(s) for s in k_list)
                        p2 = P2Quantile(p=0.5)
                        try:
                            # p2_state: [count, q(list), npos(list), np(list), dn(list)]
                            p2_count = int(p2_state[0])
                            q_values = [float(v) for v in p2_state[1]]
                            npos_list = [int(v) for v in p2_state[2]]
                            np_des_list = [float(v) for v in p2_state[3]]
                            dn_list = [float(v) for v in p2_state[4]]
                            p2.set_state(p2_count, q_values, npos_list, np_des_list, dn_list)
                        except Exception:
                            pass
                        wf = WelfordVariance()
                        try:
                            # w_state: [count, mean, m2]
                            wf.set_state(int(w_state[0]), float(w_state[1]), float(w_state[2]))
                        except Exception:
                            pass
                        self._p2_by_key[key_tuple] = p2
                        self._welford_by_key[key_tuple] = wf
        elif self._history_mode == "ema":
            ema_obj = algo_obj.get("ema")
            if isinstance(ema_obj, Mapping):
                keys = ema_obj.get("keys")
                q_list_obj = ema_obj.get("q")
                count_list_obj = ema_obj.get("count")
                if (
                    isinstance(keys, list)
                    and isinstance(q_list_obj, list)
                    and isinstance(count_list_obj, list)
                ):
                    alpha = 1.0 - self._ema_decay
                    for k_list, q_val, c_val in zip(keys, q_list_obj, count_list_obj):
                        if not isinstance(k_list, list):
                            continue
                        key_tuple = tuple(str(s) for s in k_list)
                        est = EmaQuantile(p=self._percentile / 100.0, alpha=alpha)
                        q_state = float(q_val) if isinstance(q_val, (float, int)) else None
                        c_state = int(c_val) if isinstance(c_val, int) else 0
                        est.set_state(q_state, c_state)
                        self._ema_by_key[key_tuple] = est
        else:
            win_obj = algo_obj.get("window")
            if isinstance(win_obj, Mapping):
                keys = win_obj.get("keys")
                values = win_obj.get("values")
                if isinstance(keys, list) and isinstance(values, list):
                    for k_list, vals in zip(keys, values):
                        if not isinstance(k_list, list) or not isinstance(vals, list):
                            continue
                        key_tuple = tuple(str(s) for s in k_list)
                        win = RollingWindow(size=self._window_size)
                        for v in vals:
                            try:
                                win.append(float(v))
                            except Exception:
                                continue
                        self._win_by_key[key_tuple] = win


__all__ = ["AutoClip"]
