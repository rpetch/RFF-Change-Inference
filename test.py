"""Basic sanity tests for the rffmmd package."""

import numpy as np
import pytest

from rffmmd import (RFF, med, estimate_sigma, get_thresh, get_window_sizes,
                    generate_grid, diff_statistic, localise, build_feature_cumsum,
                    detect_sparse, detect_mosum, make_scenario1, make_scenario2)


def test_rff_approximates_gaussian_kernel():
    """Inner product of two RFF vectors should track exp(-gamma||x-y||^2)."""
    rng = np.random.default_rng(0)
    gamma = 0.5
    rff = RFF(dd=3, rr=20000, gamma=gamma, seed=1)
    xx, yy = rng.normal(size=3), rng.normal(size=3)
    approx = rff.zz(xx) @ rff.zz(yy)
    exact = np.exp(-gamma * np.sum((xx - yy) ** 2))
    assert abs(approx - exact) < 0.02


def test_med_is_positive():
    XX = np.random.default_rng(0).normal(size=(50, 2))
    assert med(XX) > 0


def test_estimate_sigma_recovers_noise_scale():
    yy = np.random.default_rng(0).normal(0, 2.0, 5000)
    assert 1.5 < estimate_sigma(yy) < 2.5


def test_get_window_sizes_are_dyadic():
    assert get_window_sizes(3) == []
    assert get_window_sizes(1000) == [2, 4, 8, 16, 32, 64, 128, 256]


def test_generate_grid_windows_fit_inside_segment():
    for ll, ww in generate_grid(10, 60, 1000):
        assert ll >= 10 and ll + ww <= 60


def test_get_thresh_decreases_with_alpha():
    assert get_thresh(1000, 1.0, 5) < get_thresh(1000, 1.0, 1)


def test_diff_statistic_zero_on_constant_data():
    """A window with no distribution change should give a near-zero statistic."""
    XX = np.zeros((64, 1))
    _, _, _, Y_cumsum = build_feature_cumsum(XX, rr=100, gamma=1.0, seed=0)
    assert diff_statistic(Y_cumsum, 0, 32) < 1e-9
    assert diff_statistic(Y_cumsum, 0, 1) == 0.0


def test_localise_finds_a_planted_change():
    """A single mean shift at index 50 should be localised close to 50."""
    rng = np.random.default_rng(0)
    XX = np.concatenate([rng.normal(0, 0.3, 50), rng.normal(5, 0.3, 50)])[:, None]
    _, nn, _, Y_cumsum = build_feature_cumsum(XX, rr=300, gamma=None, seed=0)
    cp, stat = localise(Y_cumsum, 0, nn)
    assert abs(cp - 50) <= 3
    assert stat > 0


def test_localise_short_interval_returns_zero():
    Y_cumsum = np.zeros((5, 4))
    assert localise(Y_cumsum, 2, 3) == (2, 0.0)


@pytest.mark.parametrize("detector", [detect_sparse, detect_mosum])
def test_detector_output_shape(detector):
    """Both detectors return the same result dict shape."""
    XX, _, _ = make_scenario1(seed=5)
    res = detector(XX, alpha=1, rr=100, seed=42)
    assert set(res) >= {"intervals", "cps", "thresh", "sigma", "gamma"}
    assert len(res["cps"]) == len(res["intervals"])
    assert all(0 <= c <= len(XX) for c in res["cps"])


@pytest.mark.parametrize("detector", [detect_sparse, detect_mosum])
def test_detector_is_reproducible(detector):
    XX, _, _ = make_scenario2(seed=5)
    a = detector(XX, alpha=1, rr=100, seed=42)
    b = detector(XX, alpha=1, rr=100, seed=42)
    assert a["cps"] == b["cps"]


@pytest.mark.parametrize("gen", [make_scenario1, make_scenario2])
def test_scenario_shape(gen):
    XX, true_cps, seg_names = gen(seed=5)
    assert XX.shape == (1000, 1)
    assert len(true_cps) == 10
    assert len(seg_names) == 11


def test_scenario_neighbouring_segments_differ():
    _, _, seg_names = make_scenario1(seed=3)
    assert all(a != b for a, b in zip(seg_names[:-1], seg_names[1:]))
