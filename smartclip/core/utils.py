from __future__ import annotations

import math
from typing import Iterable, Sequence


def is_finite_number(value: float) -> bool:
    """Return True if value is a finite real number (not NaN/Inf)."""

    return math.isfinite(value)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value into the inclusive range [min_value, max_value]."""

    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def l2_norm(values: Iterable[float]) -> float:
    """Compute the L2 norm for an iterable of floats.

    Pure-Python helper useful in tests and simple fallbacks. Framework backends
    should provide optimized tensor-native reductions.
    """

    total = 0.0
    for x in values:
        total += float(x) * float(x)
    return math.sqrt(total)


def safe_ratio(numerator: float, denominator: float, eps: float) -> float:
    """Compute numerator / (denominator + eps) with basic stability."""

    return float(numerator) / (float(denominator) + float(eps))


def ensure_positive(value: float, eps: float) -> float:
    """Ensure a strictly positive value by lower-bounding with eps."""

    return value if value > 0.0 else eps


def percentile(values: Sequence[float], pct: float) -> float:
    """Compute a percentile (0..100) over a sequence of floats.

    This is a simple, dependency-free implementation suitable for small windows.
    For large windows or high-throughput use, algorithms specialized for streaming
    quantiles should be used by higher-level code.
    """

    if not values:
        raise ValueError("percentile() requires at least one value")
    if not (0.0 <= pct <= 100.0):
        raise ValueError("pct must be in [0, 100]")

    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    # Convert to 0..1 quantile and compute linear interpolation between neighbors
    q = pct / 100.0
    pos = q * (len(sorted_vals) - 1)
    left_index = int(math.floor(pos))
    right_index = int(math.ceil(pos))

    left = sorted_vals[left_index]
    right = sorted_vals[right_index]
    if left_index == right_index:
        return left
    frac = pos - left_index
    return (1.0 - frac) * left + frac * right


__all__ = [
    "is_finite_number",
    "clamp",
    "l2_norm",
    "safe_ratio",
    "ensure_positive",
    "percentile",
]
