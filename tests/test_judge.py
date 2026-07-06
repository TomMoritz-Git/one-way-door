"""Judge unit tests with a stubbed Anthropic client (no network)."""

from dataclasses import dataclass

from one_way_door.data import GoldItem
from one_way_door.judge import Judge


@dataclass
class _Block:
    type: str
    input: dict


@dataclass
class _Msg:
    content: list


class _StubClient:
    """Returns the action keyed by a substring lookup on the user content."""

    def __init__(self, by_text: dict[str, str]):
        self.by_text = by_text
        self.messages = self
        self.calls = 0

    def create(self, **kwargs):  # mimics client.messages.create
        self.calls += 1
        user = kwargs["messages"][0]["content"]
        action = next((v for k, v in self.by_text.items() if k in user), "unclear")
        return _Msg(content=[_Block("tool_use", {"action": action, "rationale": "stub"})])


def test_judge_parses_tool_call():
    j = Judge(client=_StubClient({"WAITFLAG": "reversible_first"}))
    out = j.judge("a dilemma", "you should WAITFLAG and sleep on it")
    assert out.action == "reversible_first"


def test_judge_many_preserves_order():
    client = _StubClient({"AAA": "reversible_first", "BBB": "commit_now"})
    out = Judge(client=client).judge_many([("q", "xAAAx"), ("q", "yBBBy"), ("q", "plain")])
    assert [o.action for o in out] == ["reversible_first", "commit_now", "unclear"]


def test_validate_on_gold_counts_disagreements():
    gold = [
        GoldItem("g0", "d", "resp with HIT", "reversible_first"),
        GoldItem("g1", "d", "resp plain", "commit_now"),  # stub will say unclear
    ]
    client = _StubClient({"HIT": "reversible_first"})
    report = Judge(client=client).validate_on_gold(gold, threshold=0.95)
    assert report.n == 2
    assert report.agreement == 0.5
    assert report.passed is False
    assert report.disagreements == (("g1", "commit_now", "unclear"),)
    assert report.recall("reversible_first") == 1.0
