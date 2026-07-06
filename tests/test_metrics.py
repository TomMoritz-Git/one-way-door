"""Accuracy, balanced-accuracy, and gap aggregation on small synthetic frames."""

import pandas as pd

from one_way_door import metrics


def _frame():
    # OOD eval: one reversible_first structure and one commit_now structure, x2
    # seeds, x {demos, why}. why is perfect; demos only gets the reversible_first
    # item right (the "always reversible_first" failure mode).
    rows = []
    structs = (
        ("one_way_door_with_escape", "reversible_first"),
        ("false_door", "commit_now"),
    )
    for seed in (0, 1):
        for struct, action in structs:
            rows.append(
                {
                    "model": "m",
                    "arm": "why",
                    "seed": seed,
                    "dilemma_id": struct,
                    "domain": "cooking",
                    "distribution": "ood",
                    "structure": struct,
                    "perspective": "first_person",
                    "trap": "hot_emotion",
                    "correct_action": action,
                    "correct": True,
                }
            )
            rows.append(
                {
                    "model": "m",
                    "arm": "demos",
                    "seed": seed,
                    "dilemma_id": struct,
                    "domain": "cooking",
                    "distribution": "ood",
                    "structure": struct,
                    "perspective": "first_person",
                    "trap": "hot_emotion",
                    "correct_action": action,
                    "correct": action == "reversible_first",
                }
            )
    return pd.DataFrame(rows)


def test_by_structure_splits_the_arms():
    out = metrics.by_structure(_frame()).set_index(["arm", "structure"])
    assert out.loc[("demos", "one_way_door_with_escape"), "acc"] == 1.0
    assert out.loc[("demos", "false_door"), "acc"] == 0.0
    assert out.loc[("why", "false_door"), "acc"] == 1.0


def test_balanced_accuracy_floors_constant_policy():
    out = metrics.balanced_accuracy(_frame(), ["model", "arm"]).set_index("arm")
    # demos = "always reversible_first" -> recall 1.0 on reversible_first, 0.0 on
    # commit_now -> balanced accuracy 0.5 (not the flattering raw 0.5-or-more).
    assert out.loc["demos", "bacc"] == 0.5
    assert out.loc["why", "bacc"] == 1.0


def test_gap_is_positive_and_paired():
    row = metrics.gap(_frame()).iloc[0]
    assert row["n_pairs"] == 2  # pairs are dilemmas; seeds are averaged within
    assert row["why_acc"] == 1.0 and row["demos_acc"] == 0.5
    assert row["gap"] == 0.5


def test_gap_by_structure():
    out = metrics.gap(_frame(), breakdown="structure").set_index("structure")
    assert out.loc["one_way_door_with_escape", "gap"] == 0.0
    assert out.loc["false_door", "gap"] == 1.0


def test_by_distribution():
    out = metrics.by_distribution(_frame()).set_index(["arm", "distribution"])
    assert out.loc[("why", "ood"), "acc"] == 1.0
    assert out.loc[("demos", "ood"), "acc"] == 0.5


def _decomp_frame():
    # demos 0.0, placebo 0.5 (length helps a bit), why 1.0 (reasoning helps more):
    # total gap 1.0 = length 0.5 + reasoning 0.5.
    accs = {"demos": [False, False], "placebo": [True, False], "why": [True, True]}
    rows = []
    for seed in (0, 1):
        for k, did in enumerate(("d0", "d1")):
            for arm, vals in accs.items():
                rows.append(
                    {
                        "model": "m",
                        "arm": arm,
                        "seed": seed,
                        "dilemma_id": did,
                        "distribution": "ood",
                        "structure": "false_door",
                        "correct_action": "commit_now",
                        "correct": vals[k],
                    }
                )
    return pd.DataFrame(rows)


def test_gap_decomposition_adds_up():
    df = _decomp_frame()
    length = metrics.gap(df, "placebo", "demos").iloc[0]
    reasoning = metrics.gap(df, "why", "placebo").iloc[0]
    total = metrics.gap(df, "why", "demos").iloc[0]
    assert length["gap"] == 0.5
    assert reasoning["gap"] == 0.5
    assert total["gap"] == 1.0
    assert length["gap"] + reasoning["gap"] == total["gap"]


def _replicate_seeds(df: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """Copy every seed-0 row to ``n_seeds`` seeds with identical correctness."""
    base = df[df["seed"] == 0]
    return pd.concat([base.assign(seed=s) for s in range(n_seeds)], ignore_index=True)


def _noisy_frame(n_dilemmas: int = 24) -> pd.DataFrame:
    # Mixed correctness so bootstrap intervals have nonzero width.
    rows = []
    for k in range(n_dilemmas):
        action = "reversible_first" if k % 2 else "commit_now"
        for arm, ok in (("why", k % 3 != 0), ("demos", k % 2 == 0)):
            rows.append(
                {
                    "model": "m",
                    "arm": arm,
                    "seed": 0,
                    "dilemma_id": f"d{k}",
                    "distribution": "ood",
                    "structure": "false_door",
                    "correct_action": action,
                    "correct": ok,
                }
            )
    return pd.DataFrame(rows)


def test_seed_replication_does_not_shrink_gap_ci():
    # Seeds share one training set, so copying each dilemma's outcome across 3
    # seeds adds no information; a clustered interval must not narrow.
    one = metrics.gap(_replicate_seeds(_noisy_frame(), 1)).iloc[0]
    three = metrics.gap(_replicate_seeds(_noisy_frame(), 3)).iloc[0]
    assert three["gap"] == one["gap"]
    assert (three["gap_hi"] - three["gap_lo"]) == (one["gap_hi"] - one["gap_lo"])


def test_seed_replication_does_not_shrink_bacc_ci():
    one = metrics.balanced_accuracy(_replicate_seeds(_noisy_frame(), 1), ["model", "arm"])
    three = metrics.balanced_accuracy(_replicate_seeds(_noisy_frame(), 3), ["model", "arm"])
    for arm in ("demos", "why"):
        o = one.set_index("arm").loc[arm]
        t = three.set_index("arm").loc[arm]
        assert t["bacc"] == o["bacc"]
        assert (t["hi"] - t["lo"]) == (o["hi"] - o["lo"])
        assert t["n_dilemmas"] == o["n_dilemmas"]
