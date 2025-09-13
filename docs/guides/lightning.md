# Lightning Guide

Requires the `torch` extra.

## Callback

```python
import smartclip as sc
from smartclip.backends.torch.integrate import SmartClipCallback

# AutoClip auto mode (default)
callback = SmartClipCallback(sc.AutoClip())
trainer = Trainer(callbacks=[callback], max_epochs=5)
trainer.fit(model, dataloader)
```

To use specific algorithms:

```python
# AGC
callback = SmartClipCallback(sc.AGC(clipping=0.01, scope="per_layer"))

# Z-Score
callback = SmartClipCallback(sc.ZScoreClip(zmax=3.0))
```

The callback clips gradients before each optimizer step via the PyTorch backend.
