from __future__ import annotations

"""PyTorch integrations: Lightning Callback and HuggingFace Trainer hook.

These utilities live behind the optional "torch" extra. Imports of heavy
dependencies are done lazily inside functions/methods so importing this module
does not require PyTorch Lightning or Transformers unless used.
"""

from typing import Any


def _get_pl_callback_base() -> Any:
    """Return the Lightning Callback base class (v2 preferred), lazily.

    Supports both the modern ``lightning.pytorch`` path (PL >= 2) and the legacy
    ``pytorch_lightning`` import path for broader compatibility.
    """

    try:
        from lightning.pytorch.callbacks import Callback  # type: ignore

        return Callback
    except Exception:
        try:
            from pytorch_lightning.callbacks import Callback  # type: ignore

            return Callback
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PyTorch Lightning is not available. Install with: pip install 'smartclip[torch]'"
            ) from exc


class SmartClipCallback(_get_pl_callback_base()):  # type: ignore[misc]
    """PyTorch Lightning callback that clips gradients before each optimizer step.

    Usage:
        >>> import smartclip as sc
        >>> from smartclip.backends.torch.integrate import SmartClipCallback
        >>> clipper = sc.AutoClip(percentile=95.0)
        >>> callback = SmartClipCallback(clipper)
        >>> # pass to Trainer(callbacks=[callback])
    """

    def __init__(self, clipper: "Any") -> None:
        super().__init__()
        self.clipper = clipper

    # Signature varies slightly across PL versions; accept both styles.
    def on_before_optimizer_step(
        self, trainer: Any, pl_module: Any, optimizer: Any, *args: Any, **kwargs: Any
    ) -> None:
        import smartclip as sc

        sc.apply(pl_module, self.clipper)


def _get_hf_trainer_callback_base() -> Any:
    try:
        from transformers import TrainerCallback  # type: ignore

        return TrainerCallback
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Transformers is not available. Install with: pip install 'smartclip[torch]'"
        ) from exc


class SmartClipTrainerCallback(_get_hf_trainer_callback_base()):  # type: ignore[misc]
    """HuggingFace Transformers callback that clips before optimizer steps.

    Attach to ``Trainer(callbacks=[...])``. Requires the model instance in the
    callback kwargs (Transformers provides ``model``).
    """

    def __init__(self, clipper: "Any") -> None:
        super().__init__()
        self.clipper = clipper

    def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        import smartclip as sc

        model = kwargs.get("model")
        if model is not None:
            sc.apply(model, self.clipper)


__all__ = [
    "SmartClipCallback",
    "SmartClipTrainerCallback",
]
