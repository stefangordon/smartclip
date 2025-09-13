# smartclip

Adaptive gradient clipping for PyTorch, TensorFlow, and JAX. Pure-Python core with thin, lazy backends. Install only the extras you need.

## Installation

- Base (core only):

```bash
pip install smartclip
```

- With integrations (install framework wheels per vendor docs first):

```bash
pip install "smartclip[torch]"    # Lightning + Transformers helpers
pip install "smartclip[tf]"       # TensorFlow helpers
pip install "smartclip[jax]"      # Flax/Optax helpers
```

## PyTorch quickstart

```python
import torch
import smartclip as sc

model = MyModel().to("cpu")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

with sc.clip_context(model, opt):  # AutoClip auto mode by default
    for x, y in loader:
        opt.zero_grad(set_to_none=True)
        loss = model(x).loss_fn(y)
        loss.backward()
        opt.step()  # clipped automatically
```

See Guides for TensorFlow, JAX, Lightning, Keras, and HF Trainer.
