from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .types import ParamLike, Params, Scope


@dataclass(frozen=True)
class Group:
    """A group of parameters associated with a grouping key.

    For framework-agnostic core, we only store the parameters. Backends can map
    modules to parameters and provide richer metadata.
    """

    key: Tuple[str, ...]
    params: List[ParamLike]


def group_parameters(params: Params, scope: Scope) -> List[Group]:
    """Group parameters by the requested scope.

    - global: one group with all parameters
    - per_param: one group per parameter (key contains a stable index)
    - per_layer: default fallback is global because layer ownership is
      framework-specific; core leaves this to backends. Callers may re-group.
    """

    param_list = list(params)

    if scope == "global":
        return [Group(key=("global",), params=param_list)]

    if scope == "per_param":
        return [Group(key=("param", str(i)), params=[p]) for i, p in enumerate(param_list)]

    # per_layer cannot be derived without framework objects; return a single group.
    return [Group(key=("layer", "all"), params=param_list)]


__all__ = [
    "Group",
    "group_parameters",
]
