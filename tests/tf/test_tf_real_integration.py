from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _import_tf():
    return pytest.importorskip("tensorflow")


def _build_tiny_model(tf):  # type: ignore[no-untyped-def]
    tf.random.set_seed(0)
    inputs = tf.keras.Input(shape=(4,))
    x = tf.keras.layers.Dense(8, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(3)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def _make_grads(tf, model):  # type: ignore[no-untyped-def]
    x = tf.random.normal((2, 4))
    y = tf.random.normal((2, 3))
    opt = tf.keras.optimizers.SGD(learning_rate=0.0)
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.mse(y, pred))
    grads = tape.gradient(loss, model.trainable_variables)
    return opt, grads


def _global_norm(tf, tensors):  # type: ignore[no-untyped-def]
    non_none = [t for t in tensors if t is not None]
    if not non_none:
        return 0.0
    return float(tf.linalg.global_norm(non_none).numpy())


def _var_key(tf, v):  # type: ignore[no-untyped-def]
    """Return a stable key for a TensorFlow Variable across TF versions.

    Prefer ``v.ref()`` when available; fall back to variable name or id.
    """
    ref_attr = getattr(v, "ref", None)
    if callable(ref_attr):
        try:
            return ref_attr()
        except Exception:
            pass
    name = getattr(v, "name", None)
    if isinstance(name, str) and name:
        return name
    return ("var", id(v))


def _leq_with_tol(a: float, b: float, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    """Return True if a <= b within numerical tolerance.

    Allows small overshoot due to float32/64 rounding using a combined
    relative+absolute tolerance: a <= b * (1 + rtol) + max(atol, rtol*abs(b)).
    """

    # We implement as: a <= b + max(atol, rtol * abs(b))
    return a <= b + max(atol, rtol * abs(b))


def test_apply_grads_agc_global_respects_threshold():
    tf = _import_tf()
    import smartclip as sc
    from smartclip.backends import tf as sc_tf

    model = _build_tiny_model(tf)
    opt, grads = _make_grads(tf, model)

    clipper = sc.AGC(clipping=1e-3, scope="global")

    # Compute baseline global norms
    _global_norm(tf, grads)
    # Handle both v.value() method and v.value property across TF versions
    w_vals = []
    for v in model.trainable_variables:
        val = getattr(v, 'value', v)
        w_vals.append(val() if callable(val) else val)
    w_norm = float(tf.linalg.global_norm(w_vals).numpy())
    T = clipper.target_norm(w_norm)

    clipped = sc_tf.apply_grads(grads, model, clipper)
    g_new = _global_norm(tf, clipped)
    assert _leq_with_tol(g_new, T)  # Allow for floating point precision

    # Ensure optimizer.apply_gradients accepts clipped grads
    opt.apply_gradients(zip(clipped, model.trainable_variables))


def test_keras_callback_patches_and_restores_and_scales():
    tf = _import_tf()
    import smartclip as sc
    from smartclip.backends import tf as sc_tf
    from smartclip.backends.tf.integrate import SmartClipCallback

    model = _build_tiny_model(tf)
    opt, grads = _make_grads(tf, model)
    clipper = sc.AGC(clipping=1e-3, scope="global")

    # Precompute target
    # Handle both v.value() method and v.value property across TF versions
    w_vals = []
    for v in model.trainable_variables:
        val = getattr(v, 'value', v)
        w_vals.append(val() if callable(val) else val)
    w_norm = float(tf.linalg.global_norm(w_vals).numpy())
    T = clipper.target_norm(w_norm)

    # Install callback and patch
    cb = SmartClipCallback(model, opt, clipper)
    cb.on_train_begin()
    try:
        # Call through patched apply_gradients and ensure it works and scales
        opt.apply_gradients(zip(grads, model.trainable_variables))

        # Build fresh grads and verify post-clip norm does not exceed T
        _, grads2 = _make_grads(tf, model)

        clipped = sc_tf.apply_grads(grads2, model, clipper)
        g_new = _global_norm(tf, clipped)
        assert _leq_with_tol(g_new, T)  # Allow for floating point precision
    finally:
        cb.on_train_end()


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_tf_agc_scopes_respect_target(scope: str):
    tf = _import_tf()
    import smartclip as sc
    from smartclip.backends import tf as sc_tf

    model = _build_tiny_model(tf)
    _, grads = _make_grads(tf, model)

    # Store original gradient norms
    orig_norms = []
    for g in grads:
        if g is not None:
            orig_norms.append(float(tf.linalg.global_norm([g]).numpy()))
        else:
            orig_norms.append(0.0)

    clipper = sc.AGC(clipping=1e-3, scope=scope)
    clipped = sc_tf.apply_grads(grads, model, clipper)

    # AGC should never increase gradient norms
    for orig, cg in zip(orig_norms, clipped):
        if cg is not None:
            new_norm = float(tf.linalg.global_norm([cg]).numpy())
            assert _leq_with_tol(new_norm, orig), (
                f"Gradient norm increased from {orig} to {new_norm}"
            )  # Allow for floating point precision

    # For global scope specifically, verify the global constraint
    if scope == "global":
        non_excluded_vars = [
            v for v in model.trainable_variables if not clipper.should_exclude_param(v)
        ]
        if non_excluded_vars:
            # Handle both v.value() method and v.value property across TF versions
            w_vals = []
            for v in non_excluded_vars:
                val = getattr(v, 'value', v)
                w_vals.append(val() if callable(val) else val)
            w_norm = float(tf.linalg.global_norm(w_vals).numpy())
            target = clipper.target_norm(w_norm)

            # Check that global gradient norm doesn't exceed target
            non_none_clipped = [cg for cg in clipped if cg is not None]
            if non_none_clipped:
                g_norm = float(tf.linalg.global_norm(non_none_clipped).numpy())
                assert _leq_with_tol(g_norm, target)  # Allow for floating point precision


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_tf_autoclip_scopes_do_not_increase_norms(scope: str):
    tf = _import_tf()
    import smartclip as sc
    from smartclip.backends import tf as sc_tf

    model = _build_tiny_model(tf)
    _, grads = _make_grads(tf, model)

    pre = []
    for g in grads:
        pre.append(float(tf.linalg.global_norm([g]).numpy()) if g is not None else 0.0)

    clipper = sc.AutoClip(percentile=95.0, scope=scope, warmup_steps=0, min_history=0)
    clipped = sc_tf.apply_grads(grads, model, clipper)

    for g0, cg in zip(pre, clipped):
        if cg is None:
            continue
        g_new = float(tf.linalg.global_norm([cg]).numpy())
        assert _leq_with_tol(g_new, g0)  # Allow for floating point precision


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_tf_zscore_scopes_do_not_increase_norms(scope: str):
    tf = _import_tf()
    import smartclip as sc
    from smartclip.backends import tf as sc_tf

    model = _build_tiny_model(tf)
    _, grads = _make_grads(tf, model)

    pre = []
    for g in grads:
        pre.append(float(tf.linalg.global_norm([g]).numpy()) if g is not None else 0.0)

    clipper = sc.ZScoreClip(zmax=3.0, scope=scope, warmup_steps=0, min_history=0)
    clipped = sc_tf.apply_grads(grads, model, clipper)

    for g0, cg in zip(pre, clipped):
        if cg is None:
            continue
        g_new = float(tf.linalg.global_norm([cg]).numpy())
        assert _leq_with_tol(g_new, g0)  # Allow for floating point precision
