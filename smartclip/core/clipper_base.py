from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypedDict, cast

from .types import Scope


@dataclass
class _HistoryGate:
    """Tracks warmup and history counts to gate clipping operations.

    Clipping is disabled while either warmup_steps is not exhausted or the
    observed history count is below min_history.
    """

    warmup_steps: int
    min_history: int
    steps: int = 0
    observations: int = 0

    def observe(self, count: int = 1) -> None:
        self.observations += int(count)
        self.steps += 1

    def can_clip(self) -> bool:
        return self.steps >= self.warmup_steps and self.observations >= self.min_history


class _GateState(TypedDict):
    warmup_steps: int
    min_history: int
    steps: int
    observations: int


class ClipperState(TypedDict):
    scope: Scope
    eps: float
    guard_nans: bool
    gate: _GateState


class ClipperBase:
    """Base class for adaptive gradient clippers.

    This class manages configuration, numeric stability constants, and minimal
    state serialization. Subclasses implement algorithm-specific logic.
    """

    def __init__(
        self,
        *,
        scope: Scope = "global",
        warmup_steps: int = 100,
        min_history: int = 50,
        eps: float = 1e-8,
        guard_nans: bool = True,
    ) -> None:
        if scope not in ("global", "per_layer", "per_param"):
            raise ValueError("scope must be one of: 'global', 'per_layer', 'per_param'")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if min_history < 0:
            raise ValueError("min_history must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self._scope: Scope = scope
        self._eps: float = float(eps)
        self._guard_nans: bool = bool(guard_nans)
        self._gate = _HistoryGate(warmup_steps=int(warmup_steps), min_history=int(min_history))

    # ---------- Properties ----------
    @property
    def scope(self) -> Scope:  # pragma: no cover - trivial accessor
        return self._scope

    @property
    def eps(self) -> float:  # pragma: no cover - trivial accessor
        return self._eps

    @property
    def guard_nans(self) -> bool:  # pragma: no cover - trivial accessor
        return self._guard_nans

    # ---------- History / warmup gating ----------
    def record_observation(self, count: int = 1) -> None:
        self._gate.observe(count)

    def can_clip(self) -> bool:
        return self._gate.can_clip()

    # ---------- Serialization ----------
    def state_dict(self) -> Mapping[str, object]:
        return {
            "scope": self._scope,
            "eps": self._eps,
            "guard_nans": self._guard_nans,
            "gate": {
                "warmup_steps": self._gate.warmup_steps,
                "min_history": self._gate.min_history,
                "steps": self._gate.steps,
                "observations": self._gate.observations,
            },
        }

    def load_state_dict(self, state: Mapping[str, object] | ClipperState) -> None:
        try:
            scope_obj = state.get("scope")
            if not isinstance(scope_obj, str) or scope_obj not in (
                "global",
                "per_layer",
                "per_param",
            ):
                raise TypeError("state['scope'] must be one of: 'global', 'per_layer', 'per_param'")
            self._scope = cast(Scope, scope_obj)

            eps_obj = state.get("eps")
            if not isinstance(eps_obj, (float, int)):
                raise TypeError("state['eps'] must be a number")
            self._eps = float(eps_obj)

            guard_obj = state.get("guard_nans")
            if not isinstance(guard_obj, bool):
                raise TypeError("state['guard_nans'] must be a bool")
            self._guard_nans = guard_obj

            gate_obj = state.get("gate")
            if not isinstance(gate_obj, Mapping):
                raise TypeError("state['gate'] must be a mapping")

            def _req_int(m: Mapping[str, object], key: str) -> int:
                val: object | None = m.get(key)
                if not isinstance(val, int):
                    raise TypeError(f"state['gate']['{key}'] must be an int")
                return int(val)

            self._gate = _HistoryGate(
                warmup_steps=_req_int(gate_obj, "warmup_steps"),
                min_history=_req_int(gate_obj, "min_history"),
                steps=_req_int(gate_obj, "steps"),
                observations=_req_int(gate_obj, "observations"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid state_dict for ClipperBase") from exc


__all__ = [
    "ClipperBase",
]
