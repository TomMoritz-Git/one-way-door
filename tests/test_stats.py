"""Confidence-interval math (pure NumPy)."""

import numpy as np

from one_way_door import stats


def test_wilson_brackets_point_and_handles_extremes():
    ci = stats.wilson(8, 10)
    assert ci.lo < ci.point < ci.hi
    assert 0.0 <= ci.lo and ci.hi <= 1.0
    # all-success: point 1.0, upper bound capped at 1, lower below 1
    top = stats.wilson(10, 10)
    assert top.point == 1.0 and top.lo < 1.0 and top.hi == 1.0


def test_wilson_zero_trials_is_nan():
    ci = stats.wilson(0, 0)
    assert np.isnan(ci.point) and np.isnan(ci.lo) and np.isnan(ci.hi)


def test_bootstrap_ci_brackets_mean():
    vals = np.array([1, 1, 1, 0, 1, 0, 1, 1.0])
    ci = stats.bootstrap_ci(vals, n_boot=2000, seed=1)
    assert abs(ci.point - vals.mean()) < 1e-9
    assert ci.lo <= ci.point <= ci.hi


def test_diff_bootstrap_paired_positive():
    why = np.array([1, 1, 1, 1, 0, 1.0])
    demos = np.array([0, 0, 1, 0, 0, 1.0])
    ci = stats.diff_bootstrap(why, demos, n_boot=2000, seed=2, paired=True)
    assert ci.point > 0
    assert ci.lo <= ci.point <= ci.hi


def test_diff_bootstrap_paired_length_mismatch():
    import pytest

    with pytest.raises(ValueError):
        stats.diff_bootstrap(np.array([1.0]), np.array([1.0, 0.0]), paired=True)
