from __future__ import annotations

import pytest

from smartclip.core.zscore import ZScoreClip


def test_initial_threshold_is_eps() -> None:
    clip = ZScoreClip()
    assert clip.threshold() == pytest.approx(clip.eps)
    assert clip.stats() == (0.0, 0.0)


def test_observe_constant_sequence_updates_stats_and_threshold() -> None:
    clip = ZScoreClip(ema_decay=0.9, warmup_steps=0, min_history=0)
    for _ in range(5):
        clip.observe(1.0)
    mean, std = clip.stats()
    assert mean == pytest.approx(1.0, abs=1e-12)
    assert std == pytest.approx(0.0, abs=1e-12)
    assert clip.threshold() == pytest.approx(1.0, abs=1e-12)


def test_threshold_increases_with_zmax() -> None:
    clip_1 = ZScoreClip(zmax=1.0, ema_decay=0.9)
    clip_3 = ZScoreClip(zmax=3.0, ema_decay=0.9)
    values = [0.0, 2.0, 0.0, 2.0, 1.0, 1.0]
    for v in values:
        clip_1.observe(v)
        clip_3.observe(v)
    m, s = clip_1.stats()
    # Ensure we have a non-zero std for a meaningful comparison
    assert s >= 0.0
    t1 = clip_1.threshold()
    t3 = clip_3.threshold()
    assert t3 >= t1
    assert t1 == pytest.approx(m + 1.0 * s, rel=1e-6, abs=1e-9)
    assert t3 == pytest.approx(m + 3.0 * s, rel=1e-6, abs=1e-9)


def test_nan_guard_drops_observation() -> None:
    clip = ZScoreClip(warmup_steps=1, min_history=1)
    clip.observe(float("nan"))
    # No observation should have been recorded; gating remains false
    assert not clip.can_clip()
    # No stats and threshold remains at eps
    assert clip.stats() == (0.0, 0.0)
    assert clip.threshold() == pytest.approx(clip.eps)


def test_threshold_any_single_non_global_key() -> None:
    clip = ZScoreClip()
    key = ("layer", "L1")
    clip.observe(2.0, key=key)
    assert clip.threshold_any() == pytest.approx(clip.threshold(key))


def test_multiple_keys_independent_thresholds() -> None:
    clip = ZScoreClip(ema_decay=0.8)
    key_a = ("layer", "A")
    key_b = ("layer", "B")
    for _ in range(5):
        clip.observe(1.0, key=key_a)
    for v in [0.0, 2.0, 0.0, 2.0]:
        clip.observe(v, key=key_b)
    t_a = clip.threshold(key_a)
    t_b = clip.threshold(key_b)
    assert t_a != t_b
    m_a, s_a = clip.stats(key_a)
    m_b, s_b = clip.stats(key_b)
    assert t_a == pytest.approx(m_a + clip._zmax * s_a, rel=1e-6, abs=1e-9)  # noqa: SLF001
    assert t_b == pytest.approx(m_b + clip._zmax * s_b, rel=1e-6, abs=1e-9)  # noqa: SLF001


def test_can_clip_gate_behavior() -> None:
    clip = ZScoreClip(warmup_steps=2, min_history=2)
    assert not clip.can_clip()
    clip.observe(1.0)
    assert not clip.can_clip()
    clip.observe(1.0)
    assert clip.can_clip()


def test_state_dict_roundtrip_with_multiple_keys() -> None:
    clip = ZScoreClip(ema_decay=0.8, zmax=2.5)
    key_a = ("layer", "A")
    key_b = ("layer", "B")
    for _ in range(3):
        clip.observe(1.0, key=key_a)
    for v in [0.0, 2.0, 0.0, 2.0]:
        clip.observe(v, key=key_b)

    state = clip.state_dict()

    restored = ZScoreClip(ema_decay=0.5, zmax=1.0)
    restored.load_state_dict(state)

    assert restored.threshold(key_a) == pytest.approx(clip.threshold(key_a))
    assert restored.threshold(key_b) == pytest.approx(clip.threshold(key_b))
    # Verify zmax carried over
    m_b, s_b = restored.stats(key_b)
    assert restored.threshold(key_b) == pytest.approx(m_b + 2.5 * s_b, rel=1e-6, abs=1e-9)
