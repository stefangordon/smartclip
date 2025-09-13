# TensorFlow Guide

TensorFlow uses explicit gradients. Clip gradients then call `optimizer.apply_gradients`.

## Custom training step

### AutoClip (auto mode)

```python
import tensorflow as tf
import smartclip as sc
from smartclip.backends import tf as sc_tf

model = MyModel()
opt = tf.keras.optimizers.Adam(3e-4)
clipper = sc.AutoClip()

def on_metrics(rec: dict) -> None:
    # Example: log to W&B/TensorBoard
    pass

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        )
    grads = tape.gradient(loss, model.trainable_variables)
    clipped = sc_tf.apply_grads(grads, model, clipper, on_metrics=on_metrics)
    opt.apply_gradients(zip(clipped, model.trainable_variables))
    return loss
```

### AGC

```python
clipper = sc.AGC(clipping=0.01)
grads = tape.gradient(loss, model.trainable_variables)
clipped = sc_tf.apply_grads(grads, model, clipper)
opt.apply_gradients(zip(clipped, model.trainable_variables))
```

### Z-Score

```python
clipper = sc.ZScoreClip(zmax=3.0)
grads = tape.gradient(loss, model.trainable_variables)
clipped = sc_tf.apply_grads(grads, model, clipper)
opt.apply_gradients(zip(clipped, model.trainable_variables))
```

## Model.fit with Keras callback

```python
import smartclip as sc
from smartclip.backends.tf.integrate import SmartClipCallback

# AutoClip auto mode by default
cb = SmartClipCallback(model_ref=lambda: model, optimizer=opt, clipper=sc.AutoClip())
model.fit(ds, epochs=5, callbacks=[cb])
```

Notes:

- Global scope computes one scale across all variables; per-parameter is supported by default.
- Prefer `apply_grads` for custom loops, or the Keras callback for `Model.fit`.
