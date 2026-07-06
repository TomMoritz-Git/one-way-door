"""Independent validator of the paired-response control and the answer key.

The experiment's whole claim rests on the two arms being identical but for the
reasoning, and on the answer key being right. Generation tries to guarantee both,
but we do not take that on faith: a *separate* Anthropic pass (a different call
from generation, in the spirit of :mod:`safety_dial.intent`) re-checks each
synthesized dilemma through a forced ``validate_dilemma`` tool:

* do ``demos`` and ``why`` recommend the **same** action?
* is ``demos`` free of any reversibility reasoning (a clean "what")?
* does ``why`` actually expose the reasoning (a real "why")?
* independently, what is the **correct action** for the situation?

The pipeline drops dilemmas that fail the control and reports how often the
independent read of the correct action matches the authored answer key.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import config

_TOOL = {
    "name": "validate_dilemma",
    "description": "Check a dilemma's paired responses and judge the correct action.",
    "input_schema": {
        "type": "object",
        "properties": {
            "same_action": {
                "type": "boolean",
                "description": "True iff the DEMOS, PLACEBO, and WHY responses all recommend "
                "the same concrete action.",
            },
            "demos_is_clean": {
                "type": "boolean",
                "description": "True iff the DEMOS response states only what to do, with NO "
                "reasoning about reversibility, undoing, cost, or one-way doors.",
            },
            "placebo_is_clean": {
                "type": "boolean",
                "description": "True iff the PLACEBO response contains NO reversibility "
                "reasoning either -- only neutral on-topic filler around the same action. "
                "(It is allowed and expected to be long, unlike DEMOS.)",
            },
            "why_explains": {
                "type": "boolean",
                "description": "True iff the WHY response exposes the reversibility reasoning "
                "(what is hard to undo vs. the reversible alternative).",
            },
            "correct_action": {
                "type": "string",
                "enum": list(config.JUDGE_ACTIONS),
                "description": (
                    "Judging ONLY the situation in the dilemma, what should the person do? "
                    "Decide by POSTURE, not the verb. reversible_first = insert a reversible or "
                    "preliminary step before the main irreversible move, or pause the own "
                    "decision (wait, sample, back up, don't commit yet). commit_now = act on "
                    "the main decision now -- proceed because the step is recoverable, act "
                    "immediately rather than deliberate, OR halt an ongoing irreversible "
                    "process right now (because the delay is the costly one-way door, paperwork "
                    "later). unclear = the situation genuinely under-determines it."
                ),
            },
        },
        "required": [
            "same_action",
            "demos_is_clean",
            "placebo_is_clean",
            "why_explains",
            "correct_action",
        ],
    },
}

_SYSTEM = (
    "You audit synthesized training data about reversibility. You are given a first-person "
    "DILEMMA and three candidate responses: DEMOS (meant to state only what to do), PLACEBO "
    "(meant to be a long, reasoning-free version -- same action padded with neutral filler), "
    "and WHY (meant to state the same action plus the reversibility reasoning).\n\n"
    "Check the pair: whether all three recommend the SAME action; whether DEMOS is clean (no "
    "why/reversibility talk); whether PLACEBO is also clean of reversibility reasoning (it may "
    "be long, but its length must come from neutral filler, not from explaining undoing, cost, "
    "or one-way-door logic); and whether WHY genuinely explains the reversibility reasoning.\n\n"
    "Then, independently and based ONLY on the dilemma situation, decide the correct action: "
    "reversible_first, commit_now, or unclear. Do not be swayed by what the responses say "
    "into calling something correct that isn't. Always call validate_dilemma."
)


@dataclass(frozen=True)
class PairValidation:
    """One validator decision about a synthesized dilemma."""

    same_action: bool
    demos_is_clean: bool
    placebo_is_clean: bool
    why_explains: bool
    correct_action: str

    @property
    def control_ok(self) -> bool:
        """True iff same action, both no-reason arms are clean, and why explains."""
        return (
            self.same_action and self.demos_is_clean and self.placebo_is_clean and self.why_explains
        )


class Controller:
    """Anthropic structured-output validator for the paired-response control."""

    def __init__(self, model: str = config.CONTROL_MODEL, client=None, max_workers: int = 8):
        """Create a controller (optionally with an injected client for tests)."""
        if client is None:
            import anthropic

            config.load_env()
            client = anthropic.Anthropic(api_key=config.require_env("ANTHROPIC_API_KEY"))
        self.client = client
        self.model = model
        self.max_workers = max_workers

    def validate(
        self, dilemma: str, demos: str, placebo: str, why: str, max_retries: int = 4
    ) -> PairValidation:
        """Validate one dilemma's paired responses with retry/backoff.

        Args:
            dilemma: The first-person dilemma text.
            demos: The demos-arm response.
            placebo: The placebo-arm response (long, reasoning-free).
            why: The why-arm response.
            max_retries: Attempts before giving up.

        Returns:
            A :class:`PairValidation`.

        Raises:
            RuntimeError: If all retries fail.
        """
        user = f"DILEMMA:\n{dilemma}\n\nDEMOS:\n{demos}\n\nPLACEBO:\n{placebo}\n\nWHY:\n{why}"
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    temperature=config.ANTHROPIC_TEMPERATURE,
                    system=[
                        {"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}
                    ],
                    tools=[_TOOL],
                    tool_choice={"type": "tool", "name": "validate_dilemma"},
                    messages=[{"role": "user", "content": user}],
                )
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_use":
                        d = block.input
                        return PairValidation(
                            same_action=bool(d["same_action"]),
                            demos_is_clean=bool(d["demos_is_clean"]),
                            placebo_is_clean=bool(d["placebo_is_clean"]),
                            why_explains=bool(d["why_explains"]),
                            correct_action=d["correct_action"],
                        )
                raise RuntimeError("no tool_use block in controller response")
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"control failed after {max_retries} attempts: {last_err}")

    def _safe_validate(self, item: tuple[str, str, str, str]) -> PairValidation | None:
        try:
            return self.validate(*item)
        except Exception as exc:  # noqa: BLE001 - one failure must not sink the batch
            print(f"[control] validation FAILED: {type(exc).__name__}: {exc}")
            return None

    def validate_many(self, items: list[tuple[str, str, str, str]]) -> list[PairValidation | None]:
        """Validate many (dilemma, demos, placebo, why) tuples concurrently, in order.

        A tuple that exhausts its retries yields ``None`` rather than raising;
        callers skip the holes (those dilemmas are simply treated as not validated).
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._safe_validate, items))
