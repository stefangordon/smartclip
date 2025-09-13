from __future__ import annotations

import math

from smartclip.core import AGC


class _StubTensor:
    def __init__(self, shape):
        self.shape = shape


class _StubParam:
    def __init__(self, shape):
        self.data = _StubTensor(shape)
        self.grad = None


def test_target_norm_basic() -> None:
    agc = AGC(clipping=0.01, eps=1e-8)
    t = agc.target_norm(10.0)
    assert math.isfinite(t)
    assert math.isclose(t, 0.01 * (10.0 + agc.eps), rel_tol=0, abs_tol=1e-12)


def test_scale_behavior_caps_and_zero() -> None:
    agc = AGC(clipping=0.01, eps=1e-8)
    # Large gradient -> scale < 1
    s_small = agc.scale(grad_norm=2.0, weight_norm=1.0)
    expected = (0.01 * (1.0 + agc.eps)) / (2.0 + agc.eps)
    assert 0.0 <= s_small < 1.0
    assert math.isclose(s_small, expected, rel_tol=0, abs_tol=1e-12)

    # Tiny gradient -> scale capped at 1.0
    s_one = agc.scale(grad_norm=1e-12, weight_norm=1.0)
    assert s_one == 1.0


def test_nan_guard_behavior() -> None:
    agc = AGC(clipping=0.01, eps=1e-8, guard_nans=True)
    # Non-finite weight -> target_norm returns >= eps
    t = agc.target_norm(float("nan"))
    assert math.isfinite(t) and t >= agc.eps

    # Non-finite inputs -> scale returns 1.0 (no scaling)
    assert agc.scale(float("nan"), 1.0) == 1.0
    assert agc.scale(1.0, float("inf")) == 1.0


def test_exclude_bias_bn_heuristic() -> None:
    agc = AGC(clipping=0.01, exclude_bias_bn=True)
    # Scalar or vector-like (<= 1D) should be excluded
    assert agc.should_exclude_param(_StubParam(())) is True  # 0-D
    assert agc.should_exclude_param(_StubParam((16,))) is True  # 1-D
    # Higher dims should not be excluded
    assert agc.should_exclude_param(_StubParam((3, 3))) is False

    agc2 = AGC(clipping=0.01, exclude_bias_bn=False)
    assert agc2.should_exclude_param(_StubParam((16,))) is False


def test_state_roundtrip() -> None:
    src = AGC(clipping=0.02, exclude_bias_bn=False, scope="per_layer", eps=1e-6)
    state = src.state_dict()
    dst = AGC()  # defaults
    dst.load_state_dict(state)
    assert math.isclose(dst.clipping, 0.02, rel_tol=0, abs_tol=1e-12)
    assert dst.exclude_bias_bn is False
    assert dst.scope == "per_layer"
    assert math.isclose(dst.eps, 1e-6, rel_tol=0, abs_tol=0)


def test_gating_with_observe() -> None:
    agc = AGC(clipping=0.01, warmup_steps=2, min_history=3)
    assert agc.can_clip() is False
    # Three observations -> steps=3, obs=3; should satisfy warmup and min_history
    agc.observe(grad_norm=1.0, weight_norm=1.0)
    assert agc.can_clip() is False
    agc.observe(grad_norm=1.0, weight_norm=1.0)
    assert agc.can_clip() is False
    agc.observe(grad_norm=1.0, weight_norm=1.0)
    assert agc.can_clip() is True
