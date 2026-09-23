"""
rffmmd: sparse-grid change point detection with an RFF-MMD statistic.
"""

from .core import (RFF, med, whitening_matrix, get_thresh, default_lam,
                   get_window_sizes, generate_grid, diff_statistic,
                   greedy_interval_search, localise, detect_sparse)
from .simulate import make_scenario1, make_scenario2

__version__ = "0.2.0"

__all__ = [
    "RFF", "med", "whitening_matrix", "get_thresh", "default_lam",
    "get_window_sizes", "generate_grid", "diff_statistic",
    "greedy_interval_search", "localise", "detect_sparse",
    "make_scenario1", "make_scenario2",
]
