"""
Synthetic benchmark scenarios with known change points.
"""

import numpy as np


# Fixed segmentation from Arlot, Celisse & Harchaoui (2019), Section 6.1:
# n = 1000, 11 segments, 10 change points.
tau_star = [500, 1000, 2000]
_N = 3000                      # series length
_D = 1                         # number of dimensions per observation
_bounds = [0] + tau_star + [_N]

# Distribution names, index-aligned with the draw() branches below.
S1_NAMES = ["Bi(10,0.2)", "Neg_Bi(3,0.7)", "Hyper(10,5,2)", "Normal(2.5,0.25)",
            "gamma(0.5,5)", "Weibull(5,2)", "Pareto(1.5,3)"]
S2_NAMES = ["Bernoulli(0.5)", "Normal(0.5,0.25)", "Exp(0.5)"]


def _labels(n_dists, rng):
    """
    Random segment labels following the paper's selection rule.

    S_1 is uniform over {0..n_dists-1}; each S_{l+1} is uniform over that set
    minus S_l, so consecutive segments never share a distribution.

    Args:
        n_dists (int) : number of candidate distributions.
        rng (Generator) : a numpy random generator.
    Returns:
        list[int] : one distribution index per segment.
    """
    out = [rng.integers(n_dists)]
    for _ in range(1, len(_bounds) - 1):
        out.append(rng.choice([d for d in range(n_dists) if d != out[-1]]))
    return out


def make_scenario1(seed=0):
    """
    Scenario 1: real-valued data with changing (mean, variance).

    Seven distributions (Table B.1), one drawn per segment. The mean usually
    shifts across boundaries -> the easier case.

    Args:
        seed (int) : RNG seed (fixes which distribution lands in each segment).
    Returns:
        X         (array, shape (_N, _D)) : the generated series. Each of the
                   _D coordinates is an independent draw from that segment's
                   distribution, using the same parameters, so all coordinates
                   change at the same points.
        tau_star  (list[int])             : the true change points.
        seg_names (list[str])             : distribution name per segment.
    """
    rng = np.random.default_rng(seed)
    labels = _labels(7, rng)

    def draw(dist, size):
        if dist == 0: return rng.binomial(10, 0.2, size).astype(float)
        if dist == 1: return rng.negative_binomial(3, 0.7, size).astype(float)
        if dist == 2: return rng.hypergeometric(5, 5, 2, size).astype(float)
        if dist == 3: return rng.normal(2.5, 0.5, size)
        if dist == 4: return rng.gamma(0.5, 5.0, size)
        if dist == 5: return 5.0 * rng.weibull(2.0, size)
        if dist == 6: return 1.5 * (1.0 + rng.pareto(3.0, size))

    xx = np.empty((_N, _D))
    for s, e, lab in zip(_bounds[:-1], _bounds[1:], labels):
        xx[s:e] = draw(lab, (e - s, _D))          # (rows, dimensions)
    seg_names = [S1_NAMES[d] for d in labels]
    return xx, list(tau_star), seg_names


def make_scenario2(seed=0):
    """
    Scenario 2: constant mean 0.5 and variance 0.25; only the shape changes.

    Three distributions (Bernoulli, Gaussian, exponential), one per segment,
    all sharing mean 0.5 and variance 0.25 -> the hard, shape-only case.

    Args:
        seed (int) : RNG seed (fixes which distribution lands in each segment).
    Returns:
        X         (array, shape (_N, _D)) : the generated series.
        tau_star  (list[int])             : the true change points.
        seg_names (list[str])             : distribution name per segment.
    """
    rng = np.random.default_rng(seed)
    labels = _labels(3, rng)

    def draw(dist, size):
        if dist == 0: return rng.binomial(1, 0.5, size).astype(float)
        if dist == 1: return rng.normal(0.5, 0.5, size)
        if dist == 2: return rng.exponential(0.5, size)

    xx = np.empty((_N, _D))
    for s, e, lab in zip(_bounds[:-1], _bounds[1:], labels):
        xx[s:e] = draw(lab, (e - s, _D))          # (rows, dimensions)
    seg_names = [S2_NAMES[d] for d in labels]
    return xx, list(tau_star), seg_names
