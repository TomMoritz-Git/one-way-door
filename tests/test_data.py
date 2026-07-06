"""Dataclass round-trips and validate-early loaders."""

import json

import pytest

from one_way_door import config, data


def _dilemma_record(**over):
    rec = {
        "id": "false_door/cooking/0001",
        "domain": "cooking",
        "split": "heldout",
        "role": "eval",
        "structure": "false_door",
        "label": "commit_now",
        "perspective": "first_person",
        "irreversibility_type": "mixing",
        "trap": "hot_emotion",
        "reversible_move": "sample",
        "text": "...",
        "action_text": "go ahead now",
    }
    rec.update(over)
    return rec


def test_dilemma_round_trip_and_correct_action():
    item = data.dilemma_from_record(_dilemma_record())
    assert item.correct_action == "commit_now"  # == label
    assert item.label in config.LABELS
    assert data.dilemma_to_record(item)["id"] == item.id


def test_distribution_from_split():
    assert data.dilemma_from_record(_dilemma_record(split="heldout")).distribution == "ood"
    assert (
        data.dilemma_from_record(_dilemma_record(split="train", domain="gardening")).distribution
        == "in_dist"
    )


def test_dilemma_bad_label_rejected():
    with pytest.raises(ValueError):
        data.dilemma_from_record(_dilemma_record(label="maybe"))


def test_dilemma_bad_role_rejected():
    with pytest.raises(ValueError):
        data.dilemma_from_record(_dilemma_record(role="holdout"))


def test_pair_target_selects_arm():
    pair = data.pair_from_record(
        {"dilemma_id": "x", "demos": "do it", "placebo": "do it, really", "why": "do it because"}
    )
    assert pair.target("demos") == "do it"
    assert pair.target("placebo") == "do it, really"
    assert pair.target("why") == "do it because"
    with pytest.raises(ValueError):
        pair.target("bogus")


def test_gold_loads_and_validates():
    gold = data.load_gold()
    assert gold
    assert all(g.action in config.JUDGE_ACTIONS for g in gold)


def test_gold_bad_action_rejected(tmp_path):
    bad = tmp_path / "gold.json"
    bad.write_text(json.dumps([{"id": "g", "dilemma": "d", "response": "r", "action": "nope"}]))
    with pytest.raises(ValueError):
        data.load_gold(bad)


def test_seed_examples_load():
    ex = data.load_seed_examples()
    assert ex and all({"structure", "demos", "placebo", "why"} <= set(e) for e in ex)
