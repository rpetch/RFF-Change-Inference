"""Tests for the rffmmd package."""

import sys
from math import sqrt, log, lgamma

import numpy as np
import pytest

from rffmmd import (RFF, med, whitening_matrix, get_thresh, default_lam,
                    get_window_sizes, generate_grid, diff_statistic, localise,
                    detect_sparse, make_scenario1, make_scenario2)

sys.setrecursionlimit(20000)          # the search recurses deeply


# --- feature map --------------------------------------------------------------

def test_rff_approximates_gaussian_kernel():
    """The inner product of two feature vectors approximates exp(-gamma||x-y||^2)."""
    rng = np.random.default_rng(0)
    gamma = 0.5
    rff = RFF(dd=3, rr=20000, gamma=gamma, seed=1)
    xx, yy = rng.normal(size=3), rng.normal(size=3)
    assert abs(rff.zz(xx) @ rff.zz(yy) - np.exp(-gamma * np.sum((xx - yy) ** 2))) < 0.02


def test_med_is_positive():
    assert med(np.random.default_rng(0).normal(size=(50, 2))) > 0


# --- whitening ----------------------------------------------------------------

def test_whitening_matrix_is_the_inverse_square_root():
    """W @ W @ (Sigma + lam*I) must equal the identity."""
    ZZ = np.random.default_rng(0).normal(size=(300, 12))
    lam = 0.3
    W = whitening_matrix(ZZ, lam)
    A = ZZ.T @ ZZ / 300 + lam * np.eye(12)
    assert np.allclose(W, W.T)
    assert np.allclose(W @ W @ A, np.eye(12), atol=1e-8)


def test_whitening_gives_identity_covariance_as_lam_shrinks():
    ZZ = np.random.default_rng(1).normal(size=(2000, 6)) @ np.diag([0.2, 0.5, 1, 2, 3, 4])
    Zw = ZZ @ whitening_matrix(ZZ, 1e-10).T
    assert np.allclose(Zw.T @ Zw / 2000, np.eye(6), atol=1e-6)


# --- threshold ----------------------------------------------------------------

def test_threshold_matches_the_formula():
    nn, ww, RR, alpha = 3000, 55, 5, 0.10
    L = log(nn / ww)
    expected = sqrt(2 * L) + (max(0.0, RR * log(L) - lgamma(RR / 2))
                              + log(1 / log(1 / (1 - alpha)))) / sqrt(2 * L)
    assert np.isclose(get_thresh(nn, ww, RR, alpha), expected)


def test_smaller_alpha_gives_higher_threshold():
    assert (get_thresh(3000, 55, 5, 0.01) > get_thresh(3000, 55, 5, 0.05)
            > get_thresh(3000, 55, 5, 0.25))


def test_auxiliary_term_is_clamped_at_zero():
    """At R = 100 the gamma term exceeds R*loglog(n/w), so the clamp removes it."""
    nn, ww, alpha = 1000, 32, 0.10
    L = log(nn / ww)
    without_aux = sqrt(2 * L) + log(1 / log(1 / (1 - alpha))) / sqrt(2 * L)
    assert np.isclose(get_thresh(nn, ww, 100, alpha), without_aux)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0])
def test_threshold_rejects_alpha_outside_zero_one(alpha):
    with pytest.raises(ValueError):
        get_thresh(1000, 32, 5, alpha)


def test_default_lam_is_one_over_log_n_over_w():
    assert np.isclose(default_lam(1000, 32), 1 / log(1000 / 32))


# --- window grid --------------------------------------------------------------

def test_window_sizes_double_from_w():
    assert get_window_sizes(3000, aa=0.5) == [55, 110, 220, 440, 880]


def test_w_equal_two_recovers_the_original_sizes():
    assert get_window_sizes(1000, WW=2) == [2, 4, 8, 16, 32, 64, 128, 256]


def test_windows_stay_at_or_below_half_the_series():
    for nn in (100, 1000, 3000):
        assert max(get_window_sizes(nn)) <= nn / 2


def test_grid_windows_fit_inside_the_segment():
    for ll, ww in generate_grid(10, 400, 1000):
        assert ll >= 10 and ll + ww <= 400


# --- statistic and localisation -------------------------------------------------

def test_statistic_is_zero_on_constant_features():
    Y_cumsum = np.zeros((65, 4))
    np.cumsum(np.ones((64, 4)), axis=0, out=Y_cumsum[1:])
    assert diff_statistic(Y_cumsum, 0, 32) == 0.0
    assert diff_statistic(Y_cumsum, 0, 1) == 0.0


def test_localise_finds_a_planted_change():
    ZZ = np.concatenate([np.zeros((50, 3)), np.ones((50, 3))])
    Y_cumsum = np.zeros((101, 3))
    np.cumsum(ZZ, axis=0, out=Y_cumsum[1:])
    cp, stat = localise(Y_cumsum, 0, 100)
    assert cp == 51 and stat > 0


# --- detector -----------------------------------------------------------------

def test_detector_returns_the_documented_keys():
    X, _, _ = make_scenario1(seed=800)
    res = detect_sparse(X, rr=5)
    assert set(res) >= {"intervals", "cps", "thresh", "lam", "gamma", "W", "alpha", "R"}
    assert len(res["cps"]) == len(res["intervals"])


def test_r_defaults_to_rr():
    X, _, _ = make_scenario1(seed=800)
    assert detect_sparse(X, rr=7)["R"] == 7


def test_detector_is_reproducible():
    X, _, _ = make_scenario2(seed=800)
    assert detect_sparse(X, rr=5)["cps"] == detect_sparse(X, rr=5)["cps"]


def test_detector_finds_a_strong_change():
    rng = np.random.default_rng(3)
    X = np.concatenate([rng.normal(0, 1, 600), rng.normal(8, 1, 600)])[:, None]
    cps = detect_sparse(X, rr=5)["cps"]
    assert cps and min(abs(c - 600) for c in cps) <= 20


def test_detector_accepts_1d_input():
    X, _, _ = make_scenario1(seed=800)
    assert detect_sparse(X[:, 0], rr=5)["cps"] == detect_sparse(X, rr=5)["cps"]


# --- scenarios ----------------------------------------------------------------

@pytest.mark.parametrize("gen", [make_scenario1, make_scenario2])
def test_scenario_shape_and_change_points(gen):
    X, true_cps, names = gen(seed=800)
    assert X.shape == (3000, 1)
    assert true_cps == [500, 1000, 2000]
    assert len(names) == 4


@pytest.mark.parametrize("gen", [make_scenario1, make_scenario2])
def test_neighbouring_segments_never_share_a_distribution(gen):
    for seed in range(20):
        _, _, names = gen(seed=seed)
        assert all(a != b for a, b in zip(names[:-1], names[1:]))
