# PyTorch Guide

## Clip context (recommended)

```python
import torch
import smartclip as sc

model = MyModel().to("cpu")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

def on_metrics(rec: dict) -> None:
    # Example: log to W&B/TensorBoard
    pass

with sc.clip_context(model, optimizer=opt, on_metrics=on_metrics):  # AutoClip auto mode by default
    for x, y in loader:
        opt.zero_grad(set_to_none=True)
        loss = model(x).loss_fn(y)
        loss.backward()
        opt.step()  # clipped automatically
```

To use a specific algorithm in the context:

```python
# AutoClip (parameterized percentile mode)
with sc.clip_context(
    model,
    optimizer=opt,
    clipper=sc.AutoClip(mode="percentile", percentile=95.0, history="ema", ema_decay=0.99, scope="per_layer"),
):
    ...

# AGC (NFNets-style)
with sc.clip_context(model, optimizer=opt, clipper=sc.AGC(clipping=0.01, scope="per_layer")):
    ...

# Z-Score clipping
with sc.clip_context(model, optimizer=opt, clipper=sc.ZScoreClip(zmax=3.0)):
    ...
```

## Vanilla training loop (explicit step)

### AutoClip (auto mode)

```python
import torch
import smartclip as sc

model = MyModel().to("cpu")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
clipper = sc.AutoClip()  # hyperparameter-free auto mode

for x, y in loader:
    opt.zero_grad(set_to_none=True)
    loss = model(x).loss_fn(y)
    loss.backward()
    sc.step(model, opt, clipper)
```

### AGC

```python
clipper = sc.AGC(clipping=0.01, scope="per_layer")
for x, y in loader:
    opt.zero_grad(set_to_none=True)
    loss = model(x).loss_fn(y)
    loss.backward()
    sc.step(model, opt, clipper)
```

### Z-Score

```python
clipper = sc.ZScoreClip(zmax=3.0)
for x, y in loader:
    opt.zero_grad(set_to_none=True)
    loss = model(x).loss_fn(y)
    loss.backward()
    sc.step(model, opt, clipper)
```

Notes:

- `scope`: `"global"`, `"per_layer"`, `"per_param"`.
- AGC excludes biases/BN by default (`exclude_bias_bn=True`).
