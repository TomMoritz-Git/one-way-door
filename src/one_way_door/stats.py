"""Small statistics core: Wilson and bootstrap confidence intervals (pure NumPy).

The headline numbers are accuracies (a model recommended the safe ordering on a
held-out dilemma or not) and the *gap* between the why-arm and the demos-arm. A
single percentage hides its own uncertainty, so every number carries an interval.
Resampling functions here treat their inputs as exchangeable, so callers must
pass one value per *independent* unit -- for this experiment that is the dilemma
(seed replicas share a training set and are averaged within each dilemma before
they get here; see ``metrics``).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate with a (lo, hi) confidence interval."""

    point: float
    lo: float
    hi: float

    @property
    def half_width(self) -> float:
        """Half the interval width (a symmetric-ish error bar for plotting)."""
        return (self.hi - self.lo) / 2.0


def wilson(successes: int, n: int, z: float = 1.96) -> Interval:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes.
        n: Number of trials.
        z: Normal quantile (1.96 for 95%).

    Returns:
        An :class:`Interval`; a degenerate ``(nan, nan, nan)`` when ``n == 0``.
    """
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return Interval(point=p, lo=max(0.0, center - margin), hi=min(1.0, center + margin))


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Percentile bootstrap interval for the mean of ``values``.

    Args:
        values: 1-D array (e.g. per-item 0/1 correctness).
        n_boot: Bootstrap resamples.
        alpha: Two-sided significance (0.05 -> 95% interval).
        seed: RNG seed for reproducibility.

    Returns:
        An :class:`Interval` around the sample mean.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(point=float(values.mean()), lo=float(lo), hi=float(hi))


def diff_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    paired: bool = True,
) -> Interval:
    """Bootstrap interval for ``mean(a) - mean(b)`` (the why-minus-demos gap).

    Args:
        a: 1-D array for group A (e.g. why-arm correctness).
        b: 1-D array for group B (e.g. demos-arm correctness).
        n_boot: Bootstrap resamples.
        alpha: Two-sided significance.
        seed: RNG seed.
        paired: If True, ``a`` and ``b`` are aligned per item and resampled with a
            shared index (the natural design when both arms answer the same
            held-out dilemmas); requires ``len(a) == len(b)``. Items must be
            independent units (one entry per dilemma, seeds pre-averaged).

    Returns:
        An :class:`Interval` for the difference of means.

    Raises:
        ValueError: If ``paired`` and the arrays differ in length.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean()) if a.size and b.size else float("nan")
    if paired:
        if a.size != b.size:
            raise ValueError(f"paired diff needs equal lengths, got {a.size} and {b.size}")
        if a.size == 0:
            return Interval(float("nan"), float("nan"), float("nan"))
        idx = rng.integers(0, a.size, size=(n_boot, a.size))
        diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    else:
        if a.size == 0 or b.size == 0:
            return Interval(float("nan"), float("nan"), float("nan"))
        ia = rng.integers(0, a.size, size=(n_boot, a.size))
        ib = rng.integers(0, b.size, size=(n_boot, b.size))
        diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return Interval(point=point, lo=float(lo), hi=float(hi))
