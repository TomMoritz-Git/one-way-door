"""Controller unit tests with a stubbed Anthropic client (no network)."""

from dataclasses import dataclass

from one_way_door.control import Controller


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

    def create(self, **kwargs):
        return _Msg(content=[_Block("tool_use", self.payload)])


def test_control_ok_requires_all_four():
    good = _StubClient(
        {
            "same_action": True,
            "demos_is_clean": True,
            "placebo_is_clean": True,
            "why_explains": True,
            "correct_action": "reversible_first",
        }
    )
    out = Controller(client=good).validate("d", "do it", "do it, really", "do it because")
    assert out.control_ok is True
    assert out.correct_action == "reversible_first"


def test_control_fails_when_demos_leaks():
    leaky = _StubClient(
        {
            "same_action": True,
            "demos_is_clean": False,
            "placebo_is_clean": True,
            "why_explains": True,
            "correct_action": "commit_now",
        }
    )
    out = Controller(client=leaky).validate("d", "do it because reversible", "do it", "do it bc")
    assert out.control_ok is False


def test_control_fails_when_placebo_leaks_reasoning():
    leaky = _StubClient(
        {
            "same_action": True,
            "demos_is_clean": True,
            "placebo_is_clean": False,  # placebo smuggled in reversibility reasoning
            "why_explains": True,
            "correct_action": "commit_now",
        }
    )
    out = Controller(client=leaky).validate("d", "do it", "do it because you can undo it", "do it")
    assert out.control_ok is False
