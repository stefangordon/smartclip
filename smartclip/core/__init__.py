from __future__ import annotations

"""Core scaffolding for smartclip.

This module exposes foundational types and base classes used by all algorithms.
Algorithms themselves are provided as lightweight stubs for now and will be
implemented in subsequent tasks.
"""

from .agc import AGC
from .autoclip import AutoClip
from .clipper_base import ClipperBase
from .types import (
    ParamLike,
    Scope,
    TensorLike,
)
from .zscore import ZScoreClip

__all__ = [
    # Types
    "Scope",
    "TensorLike",
    "ParamLike",
    # Base
    "ClipperBase",
    # Algorithms (stubs)
    "AutoClip",
    "AGC",
    "ZScoreClip",
]
