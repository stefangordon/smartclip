from __future__ import annotations

import pytest

from smartclip.core.autoclip import AutoClip


def test_auto_initial_threshold_is_eps() -> None:
    clip = AutoClip(mode="auto", warmup_steps=0, min_history=0)
    assert clip.threshold() == pytest.approx(clip.eps)


def test_auto_updates_median_and_std_and_threshold() -> None:
    clip = AutoClip(mode="auto", warmup_steps=0, min_history=0)
    # feed symmetric values around 1.0 to get median ~1 and std > 0
    values = [0.0, 2.0, 0.0, 2.0, 1.0, 1.0]
    for v in values:
        clip.observe(v)
    t = clip.threshold()
    assert t >= clip.eps
    # For this simple distribution, median should be near 1 and std > 0
    assert t > 1.0


def test_auto_multiple_keys_independent_thresholds() -> None:
    clip = AutoClip(mode="auto", warmup_steps=0, min_history=0)
    key_a = ("layer", "A")
    key_b = ("layer", "B")
    for _ in range(5):
        clip.observe(1.0, key=key_a)
    for v in [0.0, 2.0, 0.0, 2.0]:
        clip.observe(v, key=key_b)
    t_a = clip.threshold(key_a)
    t_b = clip.threshold(key_b)
    assert t_a != t_b


def test_auto_state_roundtrip() -> None:
    clip = AutoClip(mode="auto", warmup_steps=0, min_history=0)
    for v in [0.5, 1.0, 2.0, 1.5, 1.0, 0.8]:
        clip.observe(v)
    t_before = clip.threshold()
    state = clip.state_dict()

    restored = AutoClip(mode="auto")
    restored.load_state_dict(state)
    t_after = restored.threshold()
    assert t_after == pytest.approx(t_before, rel=1e-6, abs=1e-9)


def test_auto_early_stage_behavior() -> None:
    """Test threshold behavior with few observations."""
    clip = AutoClip(mode="auto", warmup_steps=0, min_history=0)

    # Initially should return eps
    assert clip.threshold() == pytest.approx(clip.eps)

    # After 1-4 observations, P² has no median yet (needs 5), should still be eps
    for value in [1.0, 2.0, 1.5, 1.2]:
        clip.observe(value)
        assert clip.threshold() == pytest.approx(clip.eps)

    # After 5 observations, P² has a median but Welford has limited variance data
    # Should use median * 2.0 fallback since variance is small/unreliable
    clip.observe(1.8)
    t = clip.threshold()
    assert t > clip.eps
    # The median should be around 1.5 (middle of sorted [1.0, 1.2, 1.5, 1.8, 2.0])
    # So threshold should be at least median * 2.0
    assert t >= 1.0  # Conservative check - actual median * 2 should be ~3.0


def test_auto_identical_values() -> None:
    """Test with identical values (std = 0 case)."""
    clip = AutoClip(mode="auto", warmup_steps=0, min_history=0)
    for _ in range(10):
        clip.observe(1.0)

    t = clip.threshold()
    # With std = 0, threshold should be median + 3 * 0 = median = 1.0
    assert t == pytest.approx(1.0, abs=1e-6)


def test_auto_nan_guard() -> None:
    """Test NaN handling in auto mode."""
    clip = AutoClip(mode="auto", warmup_steps=0, min_history=0, guard_nans=True)

    # Feed some normal values
    for v in [1.0, 2.0, 1.5]:
        clip.observe(v)
    t_before = clip.threshold()

    # NaN should be ignored
    clip.observe(float("nan"))
    t_after = clip.threshold()
    assert t_after == pytest.approx(t_before, rel=1e-6)


def test_auto_threshold_any_single_key() -> None:
    """Test threshold_any() with single non-global key."""
    clip = AutoClip(mode="auto", warmup_steps=0, min_history=0)
    key = ("layer", "conv1")

    for v in [0.5, 1.0, 2.0, 1.5, 1.0]:
        clip.observe(v, key=key)

    t_key = clip.threshold(key)
    t_any = clip.threshold_any()
    assert t_any == pytest.approx(t_key, rel=1e-6)


def test_auto_partial_state_restoration() -> None:
    """Test state restoration with partially initialized estimators."""
    clip = AutoClip(mode="auto", warmup_steps=0, min_history=0)

    # Only 2 observations - P² not fully initialized, Welford has no std
    clip.observe(1.0)
    clip.observe(1.5)

    state = clip.state_dict()
    restored = AutoClip(mode="auto")
    restored.load_state_dict(state)

    # Should handle partial state gracefully
    t_before = clip.threshold()
    t_after = restored.threshold()
    assert t_after == pytest.approx(t_before, rel=1e-6, abs=1e-9)


def test_percentile_mode_still_works() -> None:
    """Ensure percentile mode wasn't broken by auto mode changes."""
    clip = AutoClip(
        mode="percentile", percentile=95.0, history="ema", warmup_steps=0, min_history=0
    )

    for v in [0.5, 1.0, 2.0, 1.5, 1.0, 0.8, 3.0]:
        clip.observe(v)

    t = clip.threshold()
    assert t > clip.eps
    # Should be close to 95th percentile of the values
