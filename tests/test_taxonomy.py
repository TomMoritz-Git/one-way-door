"""Taxonomy loading and the balanced scenario sampler."""

import pytest

from one_way_door import config, taxonomy


def test_loads_structures_and_label_coverage():
    tax = taxonomy.load_taxonomy()
    actions = {s.action for s in tax.structures}
    assert actions <= set(config.LABELS)
    for label in config.LABELS:
        assert tax.structures_for_label(label)  # every label reachable


def test_action_for_matches_label():
    tax = taxonomy.load_taxonomy()
    for s in tax.structures:
        assert tax.action_for(s.name) == s.action


def test_sample_is_label_balanced_and_split_disjoint():
    tax = taxonomy.load_taxonomy()
    sc = taxonomy.sample_scenarios(tax)
    by_label = {label: sum(s.action == label for s in sc) for label in config.LABELS}
    assert by_label[config.LABELS[0]] == by_label[config.LABELS[1]]  # 50/50
    train_vals = {s.domain for s in sc if s.split == "train"}
    ood_vals = {s.domain for s in sc if s.split == "heldout"}
    assert train_vals.isdisjoint(ood_vals)
    assert len({s.uid for s in sc}) == len(sc)  # unique ids


def test_sample_is_deterministic():
    tax = taxonomy.load_taxonomy()
    a = [s.uid for s in taxonomy.sample_scenarios(tax, seed=1)]
    b = [s.uid for s in taxonomy.sample_scenarios(tax, seed=1)]
    assert a == b


def test_split_rejects_bad_axis_and_empty_side():
    tax = taxonomy.load_taxonomy()
    with pytest.raises(ValueError):
        taxonomy.split_heldout_axis(tax, "structure", ())  # not a decoy axis
    with pytest.raises(ValueError):
        taxonomy.split_heldout_axis(tax, "domain", ("not_a_domain",))
    all_domains = tax.axis_values("domain")
    with pytest.raises(ValueError):
        taxonomy.split_heldout_axis(tax, "domain", all_domains)  # empties train side


def test_balance_report_shape():
    tax = taxonomy.load_taxonomy()
    rep = taxonomy.balance_report(taxonomy.sample_scenarios(tax))
    assert set(rep) == set(config.LABELS)
    assert set(rep[config.LABELS[0]]) == {"irreversibility", "trap", "perspective", "domain"}
