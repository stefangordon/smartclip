from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _import_torch():
    return pytest.importorskip("torch")


def _build_tiny_model(torch):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    )
    return model


def _compute_param_norm(torch, p):  # type: ignore[no-untyped-def]
    with torch.no_grad():
        return float(torch.linalg.vector_norm(p.detach()).item())


def _compute_grad_norm(torch, p):  # type: ignore[no-untyped-def]
    g = p.grad
    if g is None:
        return 0.0
    with torch.no_grad():
        return float(torch.linalg.vector_norm(g.detach()).item())


def _make_grads(torch, model):  # type: ignore[no-untyped-def]
    x = torch.randn(2, 4)
    y = torch.randn(2, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    opt.zero_grad(set_to_none=True)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    return opt


def test_sc_step_agc_respects_thresholds():
    torch = _import_torch()
    import smartclip as sc

    model = _build_tiny_model(torch)
    opt = _make_grads(torch, model)

    clipper = sc.AGC(clipping=1e-3, exclude_bias_bn=True)

    # Precompute targets per-parameter
    targets = {}
    for p in model.parameters():
        if p.grad is None or clipper.should_exclude_param(p):
            continue
        _compute_grad_norm(torch, p)
        w = _compute_param_norm(torch, p)
        targets[p] = clipper.target_norm(w)

    sc.step(model, opt, clipper)

    for p, T in targets.items():
        g_new = _compute_grad_norm(torch, p)
        assert g_new <= T + 1e-12


def test_autoclip_context_smoke():
    torch = _import_torch()
    import smartclip as sc

    model = _build_tiny_model(torch)
    opt = _make_grads(torch, model)
    clipper = sc.AGC(clipping=1e-3)

    with sc.clip_context(model, optimizer=opt, clipper=clipper):
        # Calling step should not raise and should perform clipping under the hood
        opt.step()


def test_lightning_callback_invokes_apply_and_scales():
    torch = _import_torch()
    _ = pytest.importorskip("lightning")
    import smartclip as sc
    from smartclip.backends.torch.integrate import SmartClipCallback

    model = _build_tiny_model(torch)
    _ = _make_grads(torch, model)
    clipper = sc.AGC(clipping=1e-3)

    # Record targets pre-callback
    targets = {}
    for p in model.parameters():
        if p.grad is None or clipper.should_exclude_param(p):
            continue
        _compute_grad_norm(torch, p)
        w = _compute_param_norm(torch, p)
        targets[p] = clipper.target_norm(w)

    cb = SmartClipCallback(clipper)
    cb.on_before_optimizer_step(trainer=object(), pl_module=model, optimizer=object())

    for p, T in targets.items():
        g_new = _compute_grad_norm(torch, p)
        assert g_new <= T + 1e-12


def test_hf_trainer_callback_invokes_apply_and_scales():
    torch = _import_torch()
    pytest.importorskip("transformers")
    import smartclip as sc
    from smartclip.backends.torch.integrate import SmartClipTrainerCallback

    model = _build_tiny_model(torch)
    _ = _make_grads(torch, model)
    clipper = sc.AGC(clipping=1e-3)

    # Record targets
    targets = {}
    for p in model.parameters():
        if p.grad is None or clipper.should_exclude_param(p):
            continue
        _compute_grad_norm(torch, p)
        w = _compute_param_norm(torch, p)
        targets[p] = clipper.target_norm(w)

    cb = SmartClipTrainerCallback(clipper)
    cb.on_optimizer_step(args=None, state=None, control=None, model=model)

    for p, T in targets.items():
        g_new = _compute_grad_norm(torch, p)
        assert g_new <= T + 1e-12


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_agc_scopes_enforce_target_per_param(scope: str):
    torch = _import_torch()
    import smartclip as sc

    model = _build_tiny_model(torch)
    _ = _make_grads(torch, model)
    clipper = sc.AGC(clipping=1e-3, scope=scope)

    # Compute targets and verify after apply grads are <= target for each param
    targets = {}
    for p in model.parameters():
        if p.grad is None or clipper.should_exclude_param(p):
            continue
        w = _compute_param_norm(torch, p)
        targets[p] = clipper.target_norm(w)

    # Apply clipping
    import smartclip as sc

    sc.apply(model, clipper)

    for p, T in targets.items():
        g_new = _compute_grad_norm(torch, p)
        assert g_new <= T + 1e-12


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_autoclip_scopes_do_not_increase_param_norms(scope: str):
    torch = _import_torch()
    import smartclip as sc

    model = _build_tiny_model(torch)
    _ = _make_grads(torch, model)

    # Record pre norms
    pre = {}
    for p in model.parameters():
        if p.grad is None:
            continue
        pre[p] = _compute_grad_norm(torch, p)

    clipper = sc.AutoClip(percentile=95.0, scope=scope, warmup_steps=0, min_history=0)
    sc.apply(model, clipper)

    for p, g0 in pre.items():
        g_new = _compute_grad_norm(torch, p)
        assert g_new <= g0 + 1e-12


@pytest.mark.parametrize("scope", ["global", "per_layer", "per_param"])  # type: ignore[list-item]
def test_zscore_scopes_do_not_increase_param_norms(scope: str):
    torch = _import_torch()
    import smartclip as sc

    model = _build_tiny_model(torch)
    _ = _make_grads(torch, model)

    pre = {}
    for p in model.parameters():
        if p.grad is None:
            continue
        pre[p] = _compute_grad_norm(torch, p)

    clipper = sc.ZScoreClip(zmax=3.0, scope=scope, warmup_steps=0, min_history=0)
    sc.apply(model, clipper)

    for p, g0 in pre.items():
        g_new = _compute_grad_norm(torch, p)
        assert g_new <= g0 + 1e-12
