from __future__ import annotations

from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    runtime_checkable,
)

# Public scope literal used by all clippers
Scope = Literal["global", "per_layer", "per_param"]


@runtime_checkable
class TensorLike(Protocol):
    """A minimal protocol representing a tensor/array from any framework.

    Intentionally small to avoid importing optional frameworks at type-check time.
    """

    shape: Any


@runtime_checkable
class ParamLike(Protocol):
    """A minimal protocol representing a trainable parameter.

    The parameter stores its data (tensor/array) and an optional gradient.
    """

    data: Any
    grad: Optional[Any]


# Helpful aliases for readability in annotations. These are typing constructs at
# runtime and carry no import cost.
Params = Iterable[ParamLike]
GradList = Sequence[Any]


# Metrics collector types
AlgoName = Literal["autoclip", "zscore", "agc"]


class MetricsRecord(TypedDict, total=False):
    """Structured metrics emitted by backends during clipping.

    Fields are optional to allow a single schema across algorithms:
    - AutoClip/ZScore: populate ``threshold``; AGC: populate ``target`` and ``weight_norm``.
    """

    algo: AlgoName
    key: Tuple[str, ...]
    scope: Scope
    grad_norm: float
    weight_norm: float
    threshold: float
    target: float
    scale: float
    clipped: bool


OnMetricsCallback = Callable[[MetricsRecord], None]


__all__ = [
    "Scope",
    "TensorLike",
    "ParamLike",
    "Params",
    "GradList",
    "AlgoName",
    "MetricsRecord",
    "OnMetricsCallback",
]
