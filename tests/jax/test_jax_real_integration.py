from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _import_jax():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    return jax, jnp


def test_apply_grads_global_respects_threshold():
    jax, jnp = _import_jax()
    import smartclip as sc
    from smartclip.backends import jax as sc_jax

    # Tiny params and grads
    params = {"w1": jnp.ones((2, 2)), "w2": jnp.ones((3,))}
    grads = {"w1": jnp.full((2, 2), 10.0), "w2": jnp.full((3,), 10.0)}

    clipper = sc.AGC(clipping=1e-3, scope="global")

    # Compute global norms
    def _l2(tree):  # type: ignore[no-untyped-def]
        leaves = jax.tree_util.tree_leaves(tree)
        return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))

    _l2(grads)
    w_norm = _l2(params)
    T = clipper.target_norm(w_norm)

    clipped = sc_jax.apply_grads(grads, params, clipper)
    g_new = _l2(clipped)
    assert g_new <= T + 1e-12


def test_wrap_tx_update_smoke():
    jax, jnp = _import_jax()
    import optax  # type: ignore

    import smartclip as sc
    from smartclip.backends.jax.integrate import wrap_tx_update

    params = {"w": jnp.ones((2,))}
    grads = {"w": jnp.full((2,), 5.0)}
    tx = optax.sgd(learning_rate=0.0)
    opt_state = tx.init(params)
    clipper = sc.AGC(clipping=1e-3)

    def params_ref():  # type: ignore[no-untyped-def]
        return params

    wrapped = wrap_tx_update(tx, params_ref, clipper)
    _ = wrapped.update(grads, opt_state, params)


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_jax_agc_scopes_respect_target(scope: str):
    jax, jnp = _import_jax()
    import smartclip as sc
    from smartclip.backends import jax as sc_jax

    params = {"w1": jnp.ones((2, 2)), "w2": jnp.ones((3,))}
    grads = {"w1": jnp.full((2, 2), 10.0), "w2": jnp.full((3,), 10.0)}

    clipper = sc.AGC(clipping=1e-3, scope=scope)

    # Compute per-leaf targets
    def _leaf_targets():  # type: ignore[no-untyped-def]
        t = []
        for p in jax.tree_util.tree_leaves(params):
            w_norm = float(jnp.linalg.norm(p))
            t.append(clipper.target_norm(w_norm))
        return t

    targets = _leaf_targets()
    clipped = sc_jax.apply_grads(grads, params, clipper)

    # Compare leaves by order
    g_leaves = jax.tree_util.tree_leaves(clipped)
    for g, T in zip(g_leaves, targets):
        if g is None:
            continue
        g_norm = float(jnp.linalg.norm(g))
        assert g_norm <= T + 1e-12


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_jax_autoclip_scopes_do_not_increase_norms(scope: str):
    jax, jnp = _import_jax()
    import smartclip as sc
    from smartclip.backends import jax as sc_jax

    params = {"w1": jnp.ones((2, 2)), "w2": jnp.ones((3,))}
    grads = {"w1": jnp.full((2, 2), 5.0), "w2": jnp.full((3,), 5.0)}

    # Pre norms per leaf
    pre = [float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(grads)]

    clipper = sc.AutoClip(percentile=95.0, scope=scope, warmup_steps=0, min_history=0)
    clipped = sc_jax.apply_grads(grads, params, clipper)

    post = [float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(clipped)]
    for g0, g1 in zip(pre, post):
        assert g1 <= g0 + 1e-12


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_jax_zscore_scopes_do_not_increase_norms(scope: str):
    jax, jnp = _import_jax()
    import smartclip as sc
    from smartclip.backends import jax as sc_jax

    params = {"w1": jnp.ones((2, 2)), "w2": jnp.ones((3,))}
    grads = {"w1": jnp.full((2, 2), 5.0), "w2": jnp.full((3,), 5.0)}

    pre = [float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(grads)]

    clipper = sc.ZScoreClip(zmax=3.0, scope=scope, warmup_steps=0, min_history=0)
    clipped = sc_jax.apply_grads(grads, params, clipper)

    post = [float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(clipped)]
    for g0, g1 in zip(pre, post):
        assert g1 <= g0 + 1e-12
