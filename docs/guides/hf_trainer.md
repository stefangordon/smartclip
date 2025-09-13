# Hugging Face Trainer Guide

Requires the `torch` extra.

## Trainer callback

```python
import smartclip as sc
from smartclip.backends.torch.integrate import SmartClipTrainerCallback

# AutoClip auto mode (default)
callback = SmartClipTrainerCallback(sc.AutoClip())
trainer = Trainer(callbacks=[callback], ...)
trainer.train()
```

To use specific algorithms:

```python
# AGC
callback = SmartClipTrainerCallback(sc.AGC(clipping=0.01, scope="per_layer"))

# Z-Score
callback = SmartClipTrainerCallback(sc.ZScoreClip(zmax=3.0))
```

The callback invokes `sc.apply(model, clipper)` during optimizer steps.
