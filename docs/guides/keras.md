# Keras Guide

Use the provided callback to clip gradients during `Model.fit`.

## Callback

### AutoClip (auto mode, recommended)

```python
import smartclip as sc
from smartclip.backends.tf.integrate import SmartClipCallback

cb = SmartClipCallback(model_ref=lambda: model, optimizer=opt, clipper=sc.AutoClip())
model.fit(ds, epochs=5, callbacks=[cb])
```

### AGC

```python
cb = SmartClipCallback(model_ref=lambda: model, optimizer=opt, clipper=sc.AGC(clipping=0.01))
model.fit(ds, epochs=5, callbacks=[cb])
```

### Z-Score

```python
cb = SmartClipCallback(model_ref=lambda: model, optimizer=opt, clipper=sc.ZScoreClip(zmax=3.0))
model.fit(ds, epochs=5, callbacks=[cb])
```

Alternatively, implement a custom `train_step` and call the TF backend `apply_grads`.
