# smartclip

[![PyPI version](https://img.shields.io/pypi/v/smartclip.svg)](https://pypi.org/project/smartclip/)
[![CI](https://github.com/stefangordon/smartclip/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/stefangordon/smartclip/actions/workflows/ci.yml)
[![Docs Build](https://github.com/stefangordon/smartclip/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/stefangordon/smartclip/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-blue)](https://stefangordon.github.io/smartclip)
[![Python Versions](https://img.shields.io/pypi/pyversions/smartclip.svg)](https://pypi.org/project/smartclip/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Adaptive gradient clipping for PyTorch, TensorFlow, and JAX.

SmartClip keeps training stable with adaptive, per-step clipping you can enable in one line of code.

See the full [documentation](https://stefangordon.github.io/smartclip/) for details of the algorithms, framework usage examples, and logging metrics.

## Supported Algorithms

- AutoClip — Seetharaman et al., 2020 (MLSP). Adaptive percentile-based clipping of gradient norms.
  - [AutoClip: Adaptive Gradient Clipping for Source Separation Networks (arXiv:2007.14469)](https://arxiv.org/abs/2007.14469)
- Adaptive Gradient Clipping (AGC, NFNets-style) — Brock, De, Smith, 2021. Threshold scales with parameter norm.
  - [High-Performance Large-Scale Image Recognition Without Normalization (arXiv:2102.06171)](https://arxiv.org/abs/2102.06171)
- Z-Score clipping (EMA mean/std) — standard z-score thresholding using streaming mean/variance

  - `zmax` controls how aggressive clipping is: threshold is `mean + zmax * std` over recent norms. Higher `zmax` clips less (more tolerant), lower clips more (more aggressive). Start at `zmax=3.0`; try `2.0–2.5` if you see instability from spikes, or `3.5–4.0` if training seems over‑clipped.

## Install

```bash
pip install smartclip
```

Optional extras provide helpers for specific frameworks (install framework wheels first per vendor docs):

```bash
pip install "smartclip[torch]"    # PyTorch + Lightning/Transformers helpers
pip install "smartclip[tf]"       # TensorFlow/Keras helpers
pip install "smartclip[jax]"      # JAX/Flax/Optax helpers
```

## Quickstart

### PyTorch

```python
import torch
import smartclip as sc

model = MyModel().to("cpu")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

with sc.clip_context(model, opt):  # Defaults to AutoClip
    for x, y in loader:
        opt.zero_grad(set_to_none=True)
        loss = model(x).loss_fn(y)
        loss.backward()
        opt.step()  # clipped automatically
```

### TensorFlow/Keras

```python
import tensorflow as tf
import smartclip as sc

model = MyModel()
opt = tf.keras.optimizers.Adam(3e-4)

with sc.clip_context(model, opt, clipper=sc.ZScoreClip(zmax=3.0)):  # Use the zscore algorithm
    model.fit(ds, epochs=5)
```

### JAX/Optax

```python
import jax
import optax
from flax import linen as nn
import smartclip as sc

model = MyModel()  # Flax Module
tx = optax.adam(3e-4)

with sc.clip_context(model, tx):  # wraps tx.update
    grads = jax.grad(loss_fn)(params, batch)
    updates, opt_state = tx.update(grads, opt_state, params)  # clipped automatically
    params = optax.apply_updates(params, updates)
```

See documentation for full guides for TensorFlow, JAX, Lightning, Keras, and HF Trainer.


## Contributing

We welcome issues and pull requests. See `contribute.md` for developer setup, testing, docs, and release workflows.

## License

MIT
