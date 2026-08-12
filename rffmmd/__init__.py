"""
rffmmd — two Random-Fourier-Feature MMD change point detectors.

Public API:
    detect_sparse(X, ...)  dyadic greedy search over 2^k windows; finds an
                           unknown number of change points.
    detect_mosum(X, ...)   a single fixed-width window slid along the series.

Both return coarse intervals plus the exact change-point index pinned inside
each one by the shared `localise` post-processing step.
"""

from .core import (RFF, med, build_feature_cumsum,
                   estimate_sigma, get_thresh, get_window_sizes, generate_grid,
                   diff_statistic, localise,
                   greedy_interval_search, detect_sparse, detect_mosum)
from .simulate import (TAU_STAR, S1_NAMES, S2_NAMES,
                       make_scenario1, make_scenario2)
from .plotting import plot_detection

__version__ = "0.1.0"

__all__ = [
    "RFF", "med", "build_feature_cumsum",
    "estimate_sigma", "get_thresh", "get_window_sizes", "generate_grid",
    "diff_statistic", "localise",
    "greedy_interval_search", "detect_sparse", "detect_mosum",
    "TAU_STAR", "S1_NAMES", "S2_NAMES", "make_scenario1", "make_scenario2",
    "plot_detection",
]
