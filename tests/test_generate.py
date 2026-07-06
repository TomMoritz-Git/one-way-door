"""Generator unit tests with a stubbed Anthropic client (no network)."""

from dataclasses import dataclass

from one_way_door import data, taxonomy
from one_way_door.generate import Generator


@dataclass
class _Block:
    type: str
    input: dict


@dataclass
class _Msg:
    content: list


class _StubClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = self
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Msg(content=[_Block("tool_use", self.payload)])


_PAYLOAD = {
    "text": " I want to dump all the salt in. ",
    "action_text": " Taste first. ",
    "demos": " Taste first, add a little at a time. ",
    "placebo": " Taste first, add a little at a time, you've got this. ",
    "why": " Taste first; salt is a one-way door. ",
    "reversible_move": "sample",
}


def _generator(payload):
    tax = taxonomy.load_taxonomy()
    seeds = data.load_seed_examples()
    return Generator(tax, seeds, client=_StubClient(payload)), tax


def test_generate_parses_and_carries_scenario():
    gen, tax = _generator(_PAYLOAD)
    sc = taxonomy.sample_scenarios(tax)[0]
    out = gen.generate(sc)
    assert out.scenario.uid == sc.uid
    assert out.text == "I want to dump all the salt in."  # stripped
    assert out.action_text == "Taste first."
    assert out.reversible_move == "sample"
    assert out.placebo.startswith("Taste first")


def test_generate_prompt_includes_structure_and_perspective():
    gen, tax = _generator(_PAYLOAD)
    sc = taxonomy.sample_scenarios(tax)[0]
    gen.generate(sc)
    sent = gen.client.last_kwargs["messages"][0]["content"]
    assert sc.structure in sent
    assert sc.perspective in sent
    assert sc.trap in sent


def test_reason_first_order_in_system_prompt():
    gen, _tax = _generator(_PAYLOAD)  # default order (reason_first)
    assert "reasoning FIRST" in gen.system


def test_action_first_order_and_bad_order():
    import pytest

    from one_way_door import data, taxonomy
    from one_way_door.generate import Generator

    tax = taxonomy.load_taxonomy()
    seeds = data.load_seed_examples()
    g = Generator(tax, seeds, client=_StubClient(_PAYLOAD), order="action_first")
    assert "action FIRST" in g.system
    with pytest.raises(ValueError):
        Generator(tax, seeds, client=_StubClient(_PAYLOAD), order="sideways")
