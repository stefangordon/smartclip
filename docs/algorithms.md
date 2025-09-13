# Algorithms

## Scopes

- global: one threshold for the whole model.
- per_layer: threshold per module (group of parameters).
- per_param: threshold per parameter tensor.

All methods scale gradients by `min(1, T / (||g|| + eps))`, where `T` is the threshold.

## AutoClip

- Default mode `mode="auto"` (hyperparameter-free):
  - Streaming median via P² estimator and variance via Welford's algorithm.
  - Threshold: `T = median + 3 * std`.
  - No decay/percentile/window knobs; warmup/history gates still apply.
- Based on AutoClip by Seetharaman et al. (MLSP 2020) [arXiv:2007.14469](https://arxiv.org/abs/2007.14469).
- Custom percentile mode `mode="percentile"`:
  - Choose history as either EMA quantile estimator (`history="ema"`) or rolling window (`history="window"`).
  - Parameters: `percentile` (default 95.0), `ema_decay` (0.99), `window_size` (1024).
- Warmup/history gates: `warmup_steps=100`, `min_history=50`.

When to use / strengths:

- Auto mode: best default when you don't want to tune; robust across tasks and datasets, particularly when gradient norms are heavy‑tailed or non‑stationary. Conservative clipping reduces the chance of over‑clipping early in training.
- Percentile mode: choose an explicit tail risk (e.g., p95/p99). Prefer `history="ema"` for low‑memory, fast adaptation on non‑stationary streams; prefer `history="window"` for stronger outlier robustness on stationary regimes.

Examples:

```python
# Hyperparameter-free auto mode (recommended default)
clipper = AutoClip()

# Custom percentile mode with EMA quantile
clipper = AutoClip(mode="percentile", percentile=95.0, history="ema", ema_decay=0.99)

# Custom percentile mode with rolling window
clipper = AutoClip(mode="percentile", percentile=95.0, history="window", window_size=1024)
```

Recommended default: `AutoClip()` (auto mode).

## AGC (NFNets-style)

- Threshold depends on weight norm: `T = clipping * (||w|| + eps)`.
- Scale: `min(1, T / (||g|| + eps))` per group.
- Parameters: `clipping=0.01` (default), `exclude_bias_bn=True`, scope `per_layer` by default.
- Exclusions: skip parameters with dimensionality ≤ 1 when `exclude_bias_bn` is enabled.

Recommended default: `AGC(clipping=0.01, exclude_bias_bn=True, scope="per_layer")`.

When to use / strengths:

- Ideal when gradient magnitudes naturally scale with parameter magnitudes (e.g., NFNets and many CNNs). No history or warmup is required, so behavior is stable from step 1 and deterministic across processes.
- Per‑layer scope pairs well with the ratio‑based rule, offering scale‑invariant clipping that preserves signal for well‑scaled layers while taming outliers.
- Use `exclude_bias_bn=True` to avoid over‑regularizing biases and affine scale parameters.

## Z-Score clipping (EMA mean/std)

- Tracks EMA of mean and variance of norms per scope.
- Threshold: `T = m + zmax * std`, clip when `||g|| > T`.
- Parameters: `zmax=3.0`, `ema_decay=0.99`.

Recommended default: `ZScoreClip(zmax=3.0, ema_decay=0.99)`.

When to use / strengths:

- Good when gradient norms are roughly unimodal and you primarily want to suppress rare spikes; `zmax` gives an intuitive, unitless control on aggressiveness.
- EMA adapts smoothly to drifting scales, making it suitable for long runs and curriculum schedules without manual retuning.

### zmax: what it means and how to set it

- `zmax` is the tolerance in standard deviations above the EMA mean. The clip threshold is `m + zmax * std`, so `zmax = 3.0` means “allow up to ~3 standard deviations above the recent average before clipping.”
- Higher `zmax` → fewer clips (more tolerant). Lower `zmax` → more clips (more aggressive).
- Practical starting points:
  - Start with `zmax=3.0` (default) for most tasks.
  - If you see frequent large spikes or instability, try `zmax=2.0–2.5`.
  - If training seems over‑clipped (signal too damped) or gradients are heavy‑tailed, try `zmax=3.5–4.0`.
- Simple tuning loop: if clipping triggers on a large fraction of steps for stable runs, increase `zmax`; if spikes routinely exceed the threshold and hurt stability, decrease `zmax`.
- Note on `ema_decay`: larger values (e.g., `0.99`) adapt more slowly but are smoother; if your gradient scale shifts quickly, consider a slightly smaller decay (e.g., `0.95–0.98`) so the mean/std track faster.

## Safety and stability

- Warmup/min-history gates prevent premature clipping.
- NaN/Inf guards drop non-finite observations when enabled (`guard_nans=True`).
- Thresholds are lower-bounded by `eps`.
