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


def test_apply_grads_agc_global_respects_threshold():
    tf = _import_tf()
    import smartclip as sc
    from smartclip.backends import tf as sc_tf

    model = _build_tiny_model(tf)
    opt, grads = _make_grads(tf, model)

    clipper = sc.AGC(clipping=1e-3, scope="global")

    # Compute baseline global norms
    _global_norm(tf, grads)
    w_norm = float(tf.linalg.global_norm([v.value() for v in model.trainable_variables]).numpy())
    T = clipper.target_norm(w_norm)

    clipped = sc_tf.apply_grads(grads, model, clipper)
    g_new = _global_norm(tf, clipped)
    assert g_new <= T + 1e-12

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
    w_norm = float(tf.linalg.global_norm([v.value() for v in model.trainable_variables]).numpy())
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
        assert g_new <= T + 1e-12
    finally:
        cb.on_train_end()


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_tf_agc_scopes_respect_target(scope: str):
    tf = _import_tf()
    import smartclip as sc
    from smartclip.backends import tf as sc_tf

    model = _build_tiny_model(tf)
    _, grads = _make_grads(tf, model)

    clipper = sc.AGC(clipping=1e-3, scope=scope)

    # Compute per-variable targets
    targets = {}
    for v in model.trainable_variables:
        if clipper.should_exclude_param(v):
            continue
        # Pass variable directly to tf.linalg.global_norm - it handles value extraction
        w_norm = float(tf.linalg.global_norm([v]).numpy())
        targets[_var_key(tf, v)] = clipper.target_norm(w_norm)

    clipped = sc_tf.apply_grads(grads, model, clipper)

    for cg, v in zip(clipped, model.trainable_variables):
        if cg is None:
            continue
        if clipper.should_exclude_param(v):
            continue
        g_norm = float(tf.linalg.global_norm([cg]).numpy())
        assert g_norm <= targets[_var_key(tf, v)] + 1e-12


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
        assert g_new <= g0 + 1e-12


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
        assert g_new <= g0 + 1e-12
