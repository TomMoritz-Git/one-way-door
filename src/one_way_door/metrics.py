"""Result tables: held-out accuracy and the why-minus-demos gap (pure).

Consumes one tidy DataFrame of scored eval responses with at least the columns
``model, arm, seed, dilemma_id, distribution, structure, correct_action, correct``
(``correct`` is bool: did the judged action match the answer key). Produces
accuracy and balanced-accuracy tables with dilemma-clustered bootstrap intervals
and gap tables with paired bootstrap intervals. No I/O; the pipeline writes the
frames.

Clustering: every eval dilemma is answered by all training seeds, and the seeds
share one training set (only LoRA init / shuffle differ), so per-seed responses
to the same dilemma are not independent draws. All intervals therefore resample
*dilemmas* (carrying each dilemma's seed responses together) rather than
individual responses -- otherwise three correlated replicas per dilemma would
shrink every interval by roughly sqrt(3).
"""

import numpy as np
import pandas as pd

from . import config, stats

N_BOOT = 10_000


def _balanced_acc(correct: np.ndarray, label: np.ndarray) -> float:
    """Macro-recall over ``config.LABELS`` (base-rate-proof: constant policy -> 0.5)."""
    recalls = [correct[label == lbl].mean() for lbl in config.LABELS if (label == lbl).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def _per_dilemma(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-dilemma success/trial counts (``k``, ``m``) and the dilemma's label."""
    return sub.groupby("dilemma_id", sort=False).agg(
        k=("correct", "sum"), m=("correct", "size"), label=("correct_action", "first")
    )


def _cluster_indices(n_clusters: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))


def balanced_accuracy(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Per-group balanced accuracy (mean per-action recall) with a clustered CI.

    This is the headline metric. Labels are sampled 1:1 by construction, so
    balanced accuracy tracks raw accuracy closely; it guards the residual
    imbalance left after control filtering and floors every constant policy at
    0.5 regardless, so beating it requires genuinely discriminating.

    The 95% interval is a percentile bootstrap over *dilemmas*: each resampled
    dilemma carries all of its seed responses, so correlated seed replicas do
    not narrow the interval.
    """
    rows = []
    for keys, sub in df.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        correct = sub["correct"].to_numpy(dtype=float)
        label = sub["correct_action"].to_numpy()
        point = _balanced_acc(correct, label)

        per = _per_dilemma(sub)
        idx = _cluster_indices(len(per))
        recalls = []
        with np.errstate(invalid="ignore", divide="ignore"):
            for lbl in config.LABELS:
                sel = (per["label"] == lbl).to_numpy()
                if not sel.any():
                    continue
                k = np.where(sel, per["k"].to_numpy(dtype=float), 0.0)
                m = np.where(sel, per["m"].to_numpy(dtype=float), 0.0)
                recalls.append(k[idx].sum(axis=1) / m[idx].sum(axis=1))
        boots = np.nanmean(np.stack(recalls), axis=0) if recalls else np.array([np.nan])
        lo, hi = np.nanquantile(boots, [0.025, 0.975])

        row = dict(zip(group_cols, keys, strict=True))
        row["n"] = len(sub)
        row["n_dilemmas"] = len(per)
        for lbl in config.LABELS:
            sel = label == lbl
            row[f"recall_{lbl}"] = float(correct[sel].mean()) if sel.any() else float("nan")
        row.update({"bacc": point, "lo": float(lo), "hi": float(hi)})
        rows.append(row)
    return pd.DataFrame(rows)


def _accuracy(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Per-group accuracy with a dilemma-clustered bootstrap 95% interval."""
    rows = []
    for keys, sub in df.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        k = int(sub["correct"].sum())
        n = int(len(sub))
        per = _per_dilemma(sub)
        idx = _cluster_indices(len(per))
        pk = per["k"].to_numpy(dtype=float)
        pm = per["m"].to_numpy(dtype=float)
        accs = pk[idx].sum(axis=1) / pm[idx].sum(axis=1)
        lo, hi = np.quantile(accs, [0.025, 0.975])
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n": n,
                "correct": k,
                "acc": k / n if n else float("nan"),
                "lo": float(lo),
                "hi": float(hi),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def overall(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out accuracy per (model, arm), pooled over seeds and dilemmas."""
    return _accuracy(df, ["model", "arm"])


def by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (model, arm, seed) -- the robustness view."""
    return _accuracy(df, ["model", "arm", "seed"])


def by_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (model, arm, distribution).

    The headline generalization table: the paper's claim is that the demos arm is
    accurate in-distribution but drops out-of-distribution, while the why arm
    holds up. Only visible with both ``in_dist`` and ``ood`` measured.
    """
    return _accuracy(df, ["model", "arm", "distribution"])


def by_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out accuracy per (model, arm, domain)."""
    return _accuracy(df, ["model", "arm", "domain"])


def by_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (model, arm, structure) -- the diagnostic stratum view.

    The shortcut-breaker check: a model that only learned "urgency -> act" passes
    ``delay_is_the_door`` but fails ``false_door`` (calm, feels final, but
    recoverable). The why-model should hold up across all three structures.
    """
    return _accuracy(df, ["model", "arm", "structure"])


def by_structure_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (model, arm, structure, distribution).

    Side-by-side in-dist and OOD accuracy per structure: shows whether a
    shortcut is already visible in-distribution or only bites out of it.
    """
    return _accuracy(df, ["model", "arm", "structure", "distribution"])


def by_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (model, arm, perspective) -- transfer across framing."""
    return _accuracy(df, ["model", "arm", "perspective"])


def by_trap(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out accuracy per (model, arm, trap)."""
    return _accuracy(df, ["model", "arm", "trap"])


def _paired_arrays(sub: pd.DataFrame, arm_a: str, arm_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Align ``arm_a``/``arm_b`` by dilemma, averaging each arm's seeds first.

    One pair per dilemma: seeds share a training set, so their responses to the
    same dilemma are collapsed to a per-dilemma mean before pairing. The
    downstream paired bootstrap then resamples dilemmas, the independent unit.
    """
    pivot = sub.pivot_table(index="dilemma_id", columns="arm", values="correct", aggfunc="mean")
    if arm_a not in pivot.columns or arm_b not in pivot.columns:
        empty = np.array([], dtype=float)
        return empty, empty
    pivot = pivot.dropna(subset=[arm_a, arm_b])
    return pivot[arm_a].to_numpy(dtype=float), pivot[arm_b].to_numpy(dtype=float)


def gap(
    df: pd.DataFrame,
    arm_a: str = "why",
    arm_b: str = "demos",
    breakdown: str | None = None,
) -> pd.DataFrame:
    """``arm_a``-minus-``arm_b`` accuracy gap with a paired bootstrap 95% interval.

    Defaults to the total why-minus-demos gap. With the placebo arm, the gap is
    decomposable: ``("placebo", "demos")`` isolates the effect of longer output,
    and ``("why", "placebo")`` isolates the effect of the reasoning content.

    Pairs are dilemmas (each arm's seeds averaged within the dilemma first), so
    the bootstrap resamples the independent unit; ``n_pairs`` counts dilemmas.

    Args:
        df: Scored responses (must contain both ``arm_a`` and ``arm_b``).
        arm_a: The arm whose accuracy is the minuend.
        arm_b: The arm whose accuracy is the subtrahend.
        breakdown: Optional extra grouping column (e.g. ``"domain"``,
            ``"structure"``, ``"distribution"``); ``None`` pools all dilemmas.

    Returns:
        One row per (model[, breakdown]) with paired counts, both arm accuracies,
        and the gap with its interval. Columns name the arms explicitly.
    """
    arms = df[df["arm"].isin([arm_a, arm_b])]
    group_cols = ["model"] + ([breakdown] if breakdown else [])
    rows = []
    for keys, sub in arms.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        a, b = _paired_arrays(sub, arm_a, arm_b)
        if a.size == 0:
            continue
        ci = stats.diff_bootstrap(a, b, paired=True)
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "arm_a": arm_a,
                "arm_b": arm_b,
                "n_pairs": int(a.size),
                f"{arm_b}_acc": float(b.mean()),
                f"{arm_a}_acc": float(a.mean()),
                "gap": ci.point,
                "gap_lo": ci.lo,
                "gap_hi": ci.hi,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
