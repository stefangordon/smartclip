# Telemetry Guide

Track thresholds, gradient norms, and the applied scale for debugging and monitoring.
You can either query thresholds directly or attach a metrics collector to stream
per-group records during clipping.

```python
import smartclip as sc

clipper = sc.AutoClip(percentile=95.0)

def log_step(step: int) -> None:
    if isinstance(clipper, sc.AutoClip):
        T = clipper.threshold_any()
        print({"step": step, "autoclip_threshold": float(T)})
```

Integrate with your logger of choice (TensorBoard/W&B) by emitting these values at intervals.

## Metrics collector (recommended)

Attach an `on_metrics(record)` callback to capture per-step metrics with no extra
norm recomputation. Each `record` may include:

- `algo`: `"autoclip" | "zscore" | "agc"`
- `key`: grouping key tuple, e.g., `("global",)`, `("layer","conv1")`, `("param","7")`
- `scope`: `"global" | "per_layer" | "per_param"`
- `grad_norm`: L2 norm for the group
- `threshold` (AutoClip/ZScore) or `target` and `weight_norm` (AGC)
- `scale`: applied scale factor in [0, 1]
- `clipped`: whether `scale < 1`

### PyTorch + W&B example

```python
import wandb, smartclip as sc

wandb.init(project="smartclip-demo")
clipper = sc.AutoClip()  # or sc.ZScoreClip(), sc.AGC(...)

def on_metrics(rec: dict) -> None:
    # Flatten key tuple for nicer chart names
    key = "/".join(rec.get("key", ("global",)))
    data = {
        "smartclip/grad_norm": rec.get("grad_norm"),
        "smartclip/scale": rec.get("scale"),
        "smartclip/clipped": float(bool(rec.get("clipped", False))),
    }
    if rec.get("algo") in ("autoclip", "zscore"):
        data["smartclip/threshold"] = rec.get("threshold")
    else:
        data["smartclip/target"] = rec.get("target")
        data["smartclip/weight_norm"] = rec.get("weight_norm")
    wandb.log({f"{k}[{key}]": v for k, v in data.items() if v is not None})

with sc.clip_context(model, optimizer=opt, clipper=clipper, on_metrics=on_metrics):
    for x, y in loader:
        opt.zero_grad(set_to_none=True)
        loss = model(x).loss_fn(y)
        loss.backward()
        opt.step()
```

### TensorFlow custom loop

```python
from smartclip.backends import tf as sc_tf

clipper = sc.AutoClip()

def on_metrics(rec: dict) -> None:
    # send to TensorBoard or W&B
    pass

with tf.GradientTape() as tape:
    logits = model(x, training=True)
    loss = loss_fn(y, logits)
grads = tape.gradient(loss, model.trainable_variables)
grads = sc_tf.apply_grads(grads, model, clipper, on_metrics=on_metrics)
opt.apply_gradients(zip(grads, model.trainable_variables))
```

### JAX/Flax with Optax

```python
from smartclip.backends import jax as sc_jax

def on_metrics(rec: dict) -> None:
    pass

clipped = sc_jax.apply_grads(grads, params, clipper, on_metrics=on_metrics)
updates, new_state = tx.update(clipped, opt_state, params)
```

## Direct threshold logging (simple)

You can still query thresholds directly when you only need a couple of metrics:

```python
T = clipper.threshold_any()  # global threshold
```
