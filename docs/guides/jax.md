# JAX Guide

Clip gradients explicitly with the backend, or use the clip context to wrap your Optax update.

## Clip context (Optax update wrapped)

```python
import jax
import optax
import smartclip as sc

tx = optax.adam(3e-4)

with sc.clip_context(model, optimizer=tx):  # AutoClip auto mode by default
    grads = jax.grad(loss_fn)(params, batch)
    updates, opt_state = tx.update(grads, opt_state, params)  # clipped automatically
    params = optax.apply_updates(params, updates)
```

## Training step with Optax (explicit apply_grads)

### AutoClip (auto mode)

```python
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import smartclip as sc
from smartclip.backends import jax as sc_jax

clipper = sc.AutoClip()

def loss_fn(params, batch):
    logits = model.apply(params, batch["x"], train=True)
    return jnp.mean(cross_entropy(logits, batch["y"]))

@jax.jit
def train_step(state: TrainState, batch):
    grads = jax.grad(loss_fn)(state.params, batch)
    def on_metrics(rec: dict) -> None:
        pass  # log to W&B/TensorBoard if desired
    clipped = sc_jax.apply_grads(grads, state.params, clipper, on_metrics=on_metrics)
    updates, new_opt_state = state.tx.update(clipped, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
```

### AGC

```python
clipper = sc.AGC(clipping=0.01)
grads = jax.grad(loss_fn)(params, batch)
clipped = sc_jax.apply_grads(grads, params, clipper, on_metrics=lambda rec: None)
updates, opt_state = tx.update(clipped, opt_state, params)
params = optax.apply_updates(params, updates)
```

### Z-Score

```python
clipper = sc.ZScoreClip(zmax=3.0)
grads = jax.grad(loss_fn)(params, batch)
clipped = sc_jax.apply_grads(grads, params, clipper, on_metrics=lambda rec: None)
updates, opt_state = tx.update(clipped, opt_state, params)
params = optax.apply_updates(params, updates)
```

Notes:

- Global and per-leaf clipping are supported; AGC uses weight norms per leaf.
- Clip context wraps the optimizer's `update` for convenience; alternatively use `apply_grads` explicitly.
