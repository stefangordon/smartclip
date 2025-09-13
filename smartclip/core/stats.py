from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

from .utils import percentile


class EmaAccumulator:
    """Exponentially-weighted moving average accumulator for a single value.

    Uses the convention: new = decay * old + (1 - decay) * x
    """

    def __init__(self, decay: float, initial_value: Optional[float] = None) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")
        self._decay = float(decay)
        self._value: Optional[float] = float(initial_value) if initial_value is not None else None

    @property
    def decay(self) -> float:  # pragma: no cover - trivial accessor
        return self._decay

    @property
    def value(self) -> Optional[float]:  # current EMA value or None if uninitialized
        return self._value

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = float(x)
        else:
            d = self._decay
            self._value = d * self._value + (1.0 - d) * float(x)
        return self._value


class EmaQuantile:
    """Online exponential quantile estimator.

    Tracks a target quantile ``p`` in (0, 1) from a stream using a stochastic
    approximation update rule with step size ``alpha`` (typically ``1 - decay``).

    Update rule (quantile regression style):

        q <- q + alpha * (x - q) * (p - 1[x < q])

    This uses the magnitude of the residual (x - q) to adapt faster to large
    deviations while remaining stable for small ones. The estimate is
    initialized to the first observed value.
    """

    def __init__(self, p: float, alpha: float) -> None:
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1)")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self._p = float(p)
        self._alpha = float(alpha)
        self._q: Optional[float] = None
        self._count: int = 0

    @property
    def value(self) -> Optional[float]:  # current quantile estimate
        return self._q

    @property
    def count(self) -> int:  # number of updates applied
        return self._count

    def update(self, x: float) -> float:
        if self._q is None:
            self._q = float(x)
        else:
            q = self._q
            p = self._p
            alpha = self._alpha
            indicator = 1.0 if float(x) < q else 0.0
            self._q = q + alpha * (float(x) - q) * (p - indicator)
        self._count += 1
        return self._q

    def set_state(self, q: Optional[float], count: int) -> None:
        """Set internal estimate and count for state restoration."""
        self._q = float(q) if q is not None else None
        self._count = int(count)


class EmaMoments:
    """EMA of first and second moments to derive mean and variance.

    Maintains EMA of x and x^2 to compute variance as E[x^2] - (E[x])^2.
    """

    def __init__(
        self, decay: float, initial_mean: Optional[float] = None, initial_var: float = 0.0
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")
        self._decay = float(decay)
        self._ema_x = EmaAccumulator(decay, initial_mean)
        init_e2 = (initial_mean**2 + initial_var) if initial_mean is not None else None
        self._ema_x2 = EmaAccumulator(decay, init_e2)

    @property
    def decay(self) -> float:  # pragma: no cover - trivial accessor
        return self._decay

    def update(self, x: float) -> tuple[float, float]:
        m = self._ema_x.update(float(x))
        m2 = self._ema_x2.update(float(x) * float(x))
        var = max(0.0, m2 - m * m)
        return m, var

    def mean(self) -> Optional[float]:
        return self._ema_x.value

    def variance(self) -> Optional[float]:
        if self._ema_x.value is None or self._ema_x2.value is None:
            return None
        m = self._ema_x.value
        m2 = self._ema_x2.value
        assert m is not None and m2 is not None
        return max(0.0, m2 - m * m)

    def std(self) -> Optional[float]:
        v = self.variance()
        if v is None:
            return None
        # Local import to avoid importing math globally when not needed
        from math import sqrt

        return sqrt(v)


class RollingWindow:
    """Fixed-size rolling window of floats with percentile helper."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self._data: Deque[float] = deque(maxlen=int(size))

    def append(self, x: float) -> None:
        self._data.append(float(x))

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def to_list(self) -> List[float]:
        return list(self._data)

    def percentile(self, pct: float) -> float:
        return percentile(self._data, pct)


class WelfordVariance:
    """Online mean/variance estimator using Welford's algorithm.

    Tracks count, mean, and M2 (sum of squared deviations). Provides variance and std.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

    def update(self, x: float) -> Tuple[int, float, float]:
        self._count += 1
        delta = float(x) - self._mean
        self._mean += delta / self._count
        delta2 = float(x) - self._mean
        self._m2 += delta * delta2
        return self._count, self._mean, self._m2

    @property
    def count(self) -> int:  # pragma: no cover - trivial accessor
        return self._count

    @property
    def mean(self) -> Optional[float]:  # pragma: no cover - trivial accessor
        return None if self._count == 0 else self._mean

    @property
    def variance(self) -> Optional[float]:
        if self._count < 2:
            return None
        return max(0.0, self._m2 / (self._count - 1))

    @property
    def std(self) -> Optional[float]:
        var = self.variance
        if var is None:
            return None
        from math import sqrt

        return sqrt(var)

    def set_state(self, count: int, mean: float, m2: float) -> None:
        self._count = int(count)
        self._mean = float(mean)
        self._m2 = float(m2)

    def get_state(self) -> Tuple[int, float, float]:  # (count, mean, m2)
        return self._count, self._mean, self._m2


class P2Quantile:
    """P² (P-square) algorithm for streaming quantile estimation.

    Tracks five markers to approximate a target quantile ``p`` without storing the window.
    Reference: R. Jain and I. Chlamtac, "The P2 algorithm for dynamic calculation
    of quantiles and histograms without storing observations," CACM, 1985.
    """

    def __init__(self, p: float) -> None:
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1)")
        self._p = float(p)
        self._n: int = 0
        # Marker heights q[0..4] and positions n[0..4], desired positions np[0..4], and increments dn[0..4]
        self._q: List[float] = []
        self._npos: List[int] = []
        self._np: List[float] = []
        self._dn: List[float] = []

    @property
    def value(self) -> Optional[float]:
        return None if self._n < 5 else self._q[2]

    @property
    def count(self) -> int:  # pragma: no cover - trivial accessor
        return self._n

    def _parabolic(self, i: int, d: int) -> float:
        # Parabolic prediction of marker i with displacement d in position
        q = self._q
        n = self._npos
        denom1 = n[i + 1] - n[i]
        denom2 = n[i] - n[i - 1]
        # Guard against division by zero (should not happen in normal P² operation, but be defensive)
        if abs(denom1) < 1e-12 or abs(denom2) < 1e-12:
            return self._linear(i, d)  # Fallback to linear interpolation
        return q[i] + d * (
            (n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / denom1
            + (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / denom2
        )

    def _linear(self, i: int, d: int) -> float:
        # Linear interpolation towards neighbor depending on direction d
        denom = self._npos[i + d] - self._npos[i]
        if abs(denom) < 1e-12:
            return self._q[i]  # No interpolation if positions are identical
        return self._q[i] + d * (self._q[i + d] - self._q[i]) / denom

    def update(self, x: float) -> float:
        x = float(x)
        self._n += 1

        # Initialization phase: store first 5 observations sorted
        if self._n <= 5:
            self._q.append(x)
            if self._n == 5:
                self._q.sort()
                self._npos = [0, 1, 2, 3, 4]
                p = self._p
                self._np = [0.0, 2.0 * p, 4.0 * p, 2.0 + 2.0 * p, 4.0]
                self._dn = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]
            return self._q[2] if len(self._q) >= 3 else x

        # Locate cell k
        q = self._q
        if x < q[0]:
            q[0] = x
            k = 0
        elif x < q[1]:
            k = 0
        elif x < q[2]:
            k = 1
        elif x < q[3]:
            k = 2
        elif x <= q[4]:
            k = 3
        else:
            q[4] = x
            k = 3

        # Increment positions of markers k+1..4
        for i in range(k + 1, 5):
            self._npos[i] += 1

        # Desired marker positions
        for i in range(5):
            self._np[i] += self._dn[i]

        # Adjust heights of markers 1..3
        for i in range(1, 4):
            d = self._np[i] - self._npos[i]
            if (d >= 1.0 and self._npos[i + 1] - self._npos[i] > 1) or (
                d <= -1.0 and self._npos[i - 1] - self._npos[i] < -1
            ):
                s = 1 if d >= 1.0 else -1
                q_new = self._parabolic(i, s)
                if q[i - 1] < q_new < q[i + 1]:
                    q[i] = q_new
                else:
                    q[i] = self._linear(i, s)
                self._npos[i] += s

        return self._q[2]

    def set_state(
        self,
        count: int,
        q: List[float],
        npos: List[int],
        np_desired: List[float],
        dn: List[float],
    ) -> None:
        self._n = int(count)
        self._q = list(float(v) for v in q)
        self._npos = list(int(v) for v in npos)
        self._np = list(float(v) for v in np_desired)
        self._dn = list(float(v) for v in dn)

    def get_state(self) -> Tuple[int, List[float], List[int], List[float], List[float]]:
        return self._n, list(self._q), list(self._npos), list(self._np), list(self._dn)


__all__ = [
    "EmaAccumulator",
    "EmaQuantile",
    "EmaMoments",
    "RollingWindow",
    "P2Quantile",
    "WelfordVariance",
]
