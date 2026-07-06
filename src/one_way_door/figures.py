"""Plots: the why-vs-demos held-out comparison (matplotlib, Agg backend).

Reads the metric tables from ``results/metrics`` and writes each figure as a
paired ``.pdf`` (vector) and ``.png`` (300 dpi) into ``figures/``, using a
colorblind-safe Okabe-Ito palette. Importing this module has no side effects
beyond selecting the Agg backend; call :func:`make_all` to render.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

# Okabe-Ito (colorblind-safe). One color per training arm.
ARM_COLOR = {"base": "#999999", "demos": "#E69F00", "placebo": "#009E73", "why": "#0072B2"}
ARM_ORDER = ("base", "demos", "placebo", "why")
DIST_HATCH = {"in_dist": "", "ood": "//"}
DIST_ORDER = ("in_dist", "ood")
STRUCTURE_ORDER = ("one_way_door_with_escape", "delay_is_the_door", "false_door")
STRUCTURE_LABEL = {
    "one_way_door_with_escape": "one-way door with escape\n(→ reversible_first)",
    "delay_is_the_door": "delay is the door\n(→ commit_now)",
    "false_door": "false door\n(→ commit_now)",
}


def _metrics_dir() -> Path:
    return config.RESULTS_DIR / "metrics"


def _save(fig, name: str) -> list[Path]:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, dpi in (("pdf", None), ("png", 300)):
        path = config.FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out


def _arms_present(df: pd.DataFrame) -> list[str]:
    return [a for a in ARM_ORDER if a in set(df["arm"])]


def fig_headline(model: str, by_dist: pd.DataFrame) -> list[Path]:
    """In-distribution vs OOD balanced accuracy per arm -- the generalization plot.

    The signature of the paper's claim: the demos arm is accurate in-distribution
    but drops on unseen domains, while the why arm holds up. Balanced accuracy
    floors every constant policy at 0.5 (the dashed line).
    """
    sub = by_dist[by_dist["model"] == model]
    arms = _arms_present(sub)
    dists = [d for d in DIST_ORDER if d in set(sub["distribution"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.8 / max(len(dists), 1)
    for j, dist in enumerate(dists):
        xs, ys, errs = [], [], [[], []]
        for i, arm in enumerate(arms):
            r = sub[(sub["arm"] == arm) & (sub["distribution"] == dist)]
            if r.empty:
                continue
            r = r.iloc[0]
            xs.append(i + j * width)
            ys.append(r["bacc"])
            # Clamp at 0: a bootstrap/Wilson bound can round a hair past the point
            # estimate (e.g. hi just under a perfect 1.0), and matplotlib rejects
            # negative error-bar lengths.
            errs[0].append(max(0.0, r["bacc"] - r["lo"]))
            errs[1].append(max(0.0, r["hi"] - r["bacc"]))
        ax.bar(
            xs,
            ys,
            width=width,
            color=[ARM_COLOR[a] for a in arms],
            hatch=DIST_HATCH[dist],
            edgecolor="black",
            label=dist,
        )
        ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks([i + width * (len(dists) - 1) / 2 for i in range(len(arms))])
    ax.set_xticklabels(arms)
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("in-distribution (solid) vs unseen domains (hatched)")
    ax.axhline(0.5, ls="--", lw=1, color="#555555", label="constant-policy floor")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, f"headline_{model}")


def fig_by_structure(model: str, by_struct: pd.DataFrame) -> list[Path]:
    """Grouped demos/placebo/why accuracy by structure (shortcut-breaker view).

    When the table carries a ``distribution`` column, each arm shows an in-dist
    (solid) and an OOD (hatched) bar: the shortcut is visible as an arm that is
    merely weakest on ``one_way_door_with_escape`` in-distribution but collapses
    there out of distribution.
    """
    sub = by_struct[by_struct["model"] == model].copy()
    if "distribution" not in sub.columns:
        sub["distribution"] = "ood"
    arms = [a for a in ("demos", "placebo", "why") if a in set(sub["arm"])]
    structs = [s for s in STRUCTURE_ORDER if s in set(sub["structure"])]
    dists = [d for d in DIST_ORDER if d in set(sub["distribution"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    n_bars = max(len(arms) * len(dists), 1)
    width = 0.84 / n_bars
    for j, arm in enumerate(arms):
        for k, dist in enumerate(dists):
            xs, ys, errs = [], [], [[], []]
            for i, s in enumerate(structs):
                r = sub[
                    (sub["arm"] == arm) & (sub["structure"] == s) & (sub["distribution"] == dist)
                ]
                if r.empty:
                    continue
                r = r.iloc[0]
                xs.append(i + (j * len(dists) + k) * width)
                ys.append(r["acc"])
                errs[0].append(max(0.0, r["acc"] - r["lo"]))
                errs[1].append(max(0.0, r["hi"] - r["acc"]))
            ax.bar(
                xs,
                ys,
                width=width,
                color=ARM_COLOR[arm],
                hatch=DIST_HATCH[dist],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor="black", capsize=2)
    ax.set_xticks([i + width * (n_bars - 1) / 2 for i in range(len(structs))])
    ax.set_xticklabels([STRUCTURE_LABEL.get(s, s) for s in structs], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    title = "accuracy by structure (shortcut-breaker)"
    if len(dists) > 1:
        title += " — in-dist solid, OOD hatched"
    ax.set_title(title)
    handles = [mpatches.Patch(color=ARM_COLOR[a], label=a) for a in arms]
    if len(dists) > 1:
        handles.append(
            mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="OOD")
        )
    ax.legend(handles=handles, fontsize=8)
    return _save(fig, f"by_structure_{model}")


def fig_by_domain(model: str, by_domain: pd.DataFrame) -> list[Path]:
    """Grouped demos-vs-why accuracy across all eval domains (train + held-out)."""
    sub = by_domain[by_domain["model"] == model]
    arms = [a for a in ("demos", "why") if a in set(sub["arm"])]
    domains = sorted(set(sub["domain"]))
    fig, ax = plt.subplots(figsize=(max(6, len(domains)), 4))
    width = 0.8 / max(len(arms), 1)
    for j, arm in enumerate(arms):
        xs = [i + j * width for i in range(len(domains))]
        ys = [
            (
                sub[(sub["arm"] == arm) & (sub["domain"] == d)]["acc"].iloc[0]
                if not sub[(sub["arm"] == arm) & (sub["domain"] == d)].empty
                else 0.0
            )
            for d in domains
        ]
        ax.bar(xs, ys, width=width, color=ARM_COLOR[arm], label=arm)
    ax.set_xticks([i + width * (len(arms) - 1) / 2 for i in range(len(domains))])
    ax.set_xticklabels(domains, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("accuracy per domain (trained + unseen)")
    ax.legend(fontsize=8)
    return _save(fig, f"by_domain_{model}")


def fig_decomposition(model: str, gaps: dict[str, pd.DataFrame]) -> list[Path]:
    """Decompose the OOD demos->why gap into a length effect and a reasoning effect.

    Length = placebo - demos (both reasoning-free), reasoning = why - placebo (both
    long). If the reasoning bar is the one that clears zero, the transfer is the
    reasoning content, not just more output.
    """
    labels = [
        ("length\n(placebo - demos)", gaps["length"], "#009E73"),
        ("reasoning\n(why - placebo)", gaps["reasoning"], "#0072B2"),
        ("total\n(why - demos)", gaps["total"], "#555555"),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (_label, table, color) in enumerate(labels):
        sub = table[(table["model"] == model) & (table["distribution"] == "ood")]
        if sub.empty:
            continue
        r = sub.iloc[0]
        err = [[max(0.0, r["gap"] - r["gap_lo"])], [max(0.0, r["gap_hi"] - r["gap"])]]
        ax.bar(i, r["gap"], color=color, width=0.6)
        ax.errorbar(i, r["gap"], yerr=err, fmt="none", ecolor="black", capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lbl for lbl, _, _ in labels])
    ax.set_ylabel("OOD accuracy gap")
    ax.set_title("what drives the gap on unseen domains?")
    return _save(fig, f"decomposition_{model}")


def fig_order_ablation(model: str, gaps_by_order: dict[str, dict[str, pd.DataFrame]]) -> list[Path]:
    """OOD length/reasoning gaps side by side for reason-first vs action-first.

    The internalization test: with ``action_first`` targets the model must emit
    the action before any reasoning, so a surviving why-minus-placebo gap cannot
    be test-time chain-of-thought. One group per ordering, one bar per effect.
    """
    orders = [o for o in ("reason_first", "action_first") if o in gaps_by_order]
    effects = [("length", "#009E73"), ("reasoning", "#0072B2")]
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.8 / len(effects)
    for k, (effect, color) in enumerate(effects):
        xs, ys, errs = [], [], [[], []]
        for i, order in enumerate(orders):
            table = gaps_by_order[order][effect]
            sub = table[(table["model"] == model) & (table["distribution"] == "ood")]
            if sub.empty:
                continue
            r = sub.iloc[0]
            xs.append(i + k * width)
            ys.append(r["gap"])
            errs[0].append(max(0.0, r["gap"] - r["gap_lo"]))
            errs[1].append(max(0.0, r["gap_hi"] - r["gap"]))
        ax.bar(xs, ys, width=width, color=color, label=effect)
        ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor="black", capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([i + width / 2 for i in range(len(orders))])
    ax.set_xticklabels(
        [
            "reasoning before action\n(model may re-derive at test time)",
            "action before reasoning\n(no room for test-time reasoning)",
        ][: len(orders)],
        fontsize=8,
    )
    ax.set_ylabel("OOD accuracy gap")
    ax.set_title("does the reasoning gap survive when the action comes first?")
    ax.legend(fontsize=8)
    return _save(fig, f"order_ablation_{model}")


def _load_gaps(mdir: Path) -> dict[str, pd.DataFrame]:
    return {
        "length": pd.read_parquet(mdir / "gap_length_by_distribution.parquet"),
        "reasoning": pd.read_parquet(mdir / "gap_reasoning_by_distribution.parquet"),
        "total": pd.read_parquet(mdir / "gap_total_by_distribution.parquet"),
    }


def make_all(ablation_root: Path | None = None) -> list[Path]:
    """Render every figure for every model found in the metric tables.

    Args:
        ablation_root: Optional results root of an ``action_first`` run; when
            given, also renders the order-ablation comparison (this run's gaps
            as ``reason_first`` vs the ablation root's as ``action_first``).
    """
    mdir = _metrics_dir()
    if not (mdir / "balanced_by_distribution.parquet").exists():
        raise FileNotFoundError("no metric tables; run the metrics stage first")
    by_dist = pd.read_parquet(mdir / "balanced_by_distribution.parquet")
    struct_path = mdir / "by_structure_by_distribution.parquet"
    if struct_path.exists():
        by_struct = pd.read_parquet(struct_path)
    else:  # older metric runs only have the OOD slice
        by_struct = pd.read_parquet(mdir / "by_structure_ood.parquet")
    by_domain = pd.read_parquet(mdir / "by_domain.parquet")
    gaps = _load_gaps(mdir)

    out: list[Path] = []
    for model in sorted(set(by_dist["model"])):
        out += fig_headline(model, by_dist)
        out += fig_by_structure(model, by_struct)
        out += fig_by_domain(model, by_domain)
        out += fig_decomposition(model, gaps)
        if ablation_root is not None:
            gaps_ablation = _load_gaps(ablation_root / "metrics")
            out += fig_order_ablation(model, {"reason_first": gaps, "action_first": gaps_ablation})
    return out
