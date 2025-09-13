from __future__ import annotations

import pytest

from smartclip.core.stats import P2Quantile, WelfordVariance


def test_p2_quantile_basic() -> None:
    """Test P² quantile estimator basic functionality."""
    p2 = P2Quantile(p=0.5)  # median

    # Should return None until 5 observations
    assert p2.value is None

    # Feed 5 values
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        p2.update(v)

    # Should have a median now
    median = p2.value
    assert median is not None
    assert 2.5 <= median <= 3.5  # Should be close to true median (3.0)


def test_p2_quantile_identical_values() -> None:
    """Test P² with identical values (edge case for division)."""
    p2 = P2Quantile(p=0.5)

    # Feed 10 identical values
    for _ in range(10):
        p2.update(1.0)

    # Should converge to the constant value
    assert p2.value == pytest.approx(1.0, abs=1e-6)


def test_p2_quantile_state_roundtrip() -> None:
    """Test P² state serialization/deserialization."""
    p2 = P2Quantile(p=0.5)

    # Initialize with some values
    for v in [0.5, 1.0, 2.0, 1.5, 1.0, 0.8]:
        p2.update(v)

    median_before = p2.value
    state = p2.get_state()

    # Create new instance and restore state
    p2_restored = P2Quantile(p=0.5)
    p2_restored.set_state(*state)

    assert p2_restored.value == pytest.approx(median_before, rel=1e-6, abs=1e-9)


def test_welford_variance_basic() -> None:
    """Test Welford variance estimator basic functionality."""
    welford = WelfordVariance()

    # Initially should have no mean/variance
    assert welford.mean is None
    assert welford.variance is None
    assert welford.std is None

    # After 1 observation
    welford.update(1.0)
    assert welford.mean == pytest.approx(1.0)
    assert welford.variance is None  # Need >= 2 for variance

    # After 2 observations
    welford.update(3.0)
    assert welford.mean == pytest.approx(2.0)
    assert welford.variance == pytest.approx(2.0)  # Sample variance of [1, 3]
    assert welford.std == pytest.approx(1.414, abs=1e-3)


def test_welford_variance_identical_values() -> None:
    """Test Welford with identical values (variance = 0)."""
    welford = WelfordVariance()

    for _ in range(5):
        welford.update(1.0)

    assert welford.mean == pytest.approx(1.0)
    assert welford.variance == pytest.approx(0.0, abs=1e-12)
    assert welford.std == pytest.approx(0.0, abs=1e-12)


def test_welford_variance_state_roundtrip() -> None:
    """Test Welford state serialization/deserialization."""
    welford = WelfordVariance()

    for v in [1.0, 2.0, 3.0, 2.5, 1.5]:
        welford.update(v)

    mean_before = welford.mean
    var_before = welford.variance
    state = welford.get_state()

    # Create new instance and restore state
    welford_restored = WelfordVariance()
    welford_restored.set_state(*state)

    assert welford_restored.mean == pytest.approx(mean_before, rel=1e-6, abs=1e-9)
    assert welford_restored.variance == pytest.approx(var_before, rel=1e-6, abs=1e-9)


def test_p2_quantile_invalid_p() -> None:
    """Test P² constructor validation."""
    with pytest.raises(ValueError, match="p must be in \\(0, 1\\)"):
        P2Quantile(p=0.0)

    with pytest.raises(ValueError, match="p must be in \\(0, 1\\)"):
        P2Quantile(p=1.0)

    with pytest.raises(ValueError, match="p must be in \\(0, 1\\)"):
        P2Quantile(p=-0.1)
