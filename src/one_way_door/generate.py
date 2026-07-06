"""Dataset synthesis: Claude authors a dilemma + paired responses per scenario.

Each :class:`~one_way_door.taxonomy.Scenario` fixes the determinant (the situation
structure, hence the correct action) and the four decoys (irreversibility type,
trap, perspective, domain). The generator writes a dilemma matching that spec --
in the requested *perspective* -- and one recommended action rendered three ways
through a forced ``write_dilemma`` tool call:

* ``demos`` -- the bare action, no reasoning;
* ``placebo`` -- the same action padded to ~the length of ``why`` with neutral
  filler, but no reasoning (a length-matched control);
* ``why`` -- the same action plus the reversibility reasoning.

Emitting the action once and rendering it three ways structurally guarantees the
arms agree on the action; :mod:`one_way_door.control` then verifies it. The
dilemma text never names the principle, so the held-out probe stays blind.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import config
from .taxonomy import Scenario, Taxonomy


def _build_tool(taxonomy: Taxonomy) -> dict:
    return {
        "name": "write_dilemma",
        "description": "Record one reversibility dilemma and its paired responses.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "The dilemma (2-5 sentences), written in the requested PERSPECTIVE and "
                        "DOMAIN, showing the TRAP that baits a rush. MUST NOT mention "
                        "reversibility, one-way doors, or any general principle -- just the "
                        "situation as experienced."
                    ),
                },
                "action_text": {
                    "type": "string",
                    "description": (
                        "The single recommended action, imperative, 1-2 sentences. This exact "
                        "action is what all three responses recommend."
                    ),
                },
                "demos": {
                    "type": "string",
                    "description": (
                        "The recommended action stated plainly, NO explanation -- no mention of "
                        "reversibility, cost, undoing, or one-way doors. Just what to do."
                    ),
                },
                "placebo": {
                    "type": "string",
                    "description": (
                        "The SAME action, expanded to roughly the LENGTH of `why` but with NO "
                        "reasoning: pad with neutral, on-topic filler (restating the situation, "
                        "sensory/contextual detail, encouragement). NEVER mention reversibility, "
                        "undoing, cost, or one-way doors. A length-matched placebo for `why`."
                    ),
                },
                "why": {
                    "type": "string",
                    "description": (
                        "The SAME action plus the reversibility reasoning: name what is hard to "
                        "undo, what the reversible alternative is, and why the ordering follows. "
                        "The action must match `demos`."
                    ),
                },
                "reversible_move": {
                    "type": "string",
                    "enum": list(taxonomy.reversible_moves),
                    "description": "The reversible alternative the action leans on (recorded).",
                },
            },
            "required": ["text", "action_text", "demos", "placebo", "why", "reversible_move"],
        },
    }


# Guidance keyed by the Tier-1 situation structure (which fixes the action).
_STRUCTURE_GUIDANCE = {
    "one_way_door_with_escape": (
        "Correct action: do the REVERSIBLE thing first. This is a genuine one-way door and a "
        "cheap reversible move is available (sample, do less, back up, draft and wait, stage a "
        "test, keep the option open). The action should take that reversible step before the "
        "irreversible one."
    ),
    "delay_is_the_door": (
        "Correct action: ACT NOW. Waiting is itself the irreversible harm here -- the delay is "
        "the one-way door (the wound bleeds, the data keeps leaking, the chance passes). The "
        "action should tell them to act now and deliberate later. Do NOT recommend stalling."
    ),
    "false_door": (
        "Correct action: PROCEED NOW. The person fears this step is irreversible, but it is "
        "actually recoverable (version history, backups, regrowth, refunds), so the caution is "
        "unwarranted. The action should tell them to go ahead now. Do NOT recommend waiting, "
        "sampling, or backing up: the feared step simply is not a one-way door."
    ),
}

_PERSPECTIVE_INSTRUCTION = {
    "first_person": "Write it in the first person: the person describes their own situation.",
    "advising_friend": "Write it as someone describing a FRIEND's situation and deciding what "
    "to tell them ('My friend is about to...').",
    "professional_role": "Write it from a PROFESSIONAL role (e.g. clinician, manager, "
    "engineer) facing a decision about someone they're responsible for.",
    "observer": "Write it as a neutral third-person account of someone facing the decision.",
}

_SYSTEM = (
    "You author training data for a study of one principle: favor reversible actions -- when a "
    "choice is a one-way door, slow down; but recognize when it is NOT a one-way door. You write "
    "vivid, realistic dilemmas and, for each, ONE recommended action rendered three ways.\n\n"
    "Hard rules:\n"
    "1. `demos`, `placebo`, and `why` must recommend the EXACT SAME concrete action.\n"
    "2. `demos` states only WHAT to do -- no 'because', no mention of undoing, cost, "
    "reversibility, or one-way doors.\n"
    "3. `placebo` is the same action at roughly the LENGTH of `why`, but its extra words are "
    "neutral on-topic filler, NEVER reasoning about reversibility, undoing, or cost.\n"
    "4. `why` includes the reversibility reasoning -- what is hard to undo versus the "
    "reversible alternative.\n"
    "5. The dilemma `text` must NOT mention the principle, reversibility, or one-way doors, and "
    "must be written in the requested perspective.\n"
    "6. The recommended action must be genuinely correct for the intended structure.\n"
)

_ORDER_RULES = {
    "reason_first": (
        "7. ORDERING: in `why`, put the reversibility reasoning FIRST and the recommended "
        "action LAST, so the reasoning leads into the action. In `placebo`, put the neutral "
        "filler FIRST and the action LAST, so the action sits in the SAME position as in "
        "`why`. (`demos` is just the bare action.)\n"
        "Always call write_dilemma."
    ),
    "action_first": (
        "7. ORDERING: in `why`, state the recommended action FIRST, then the reversibility "
        "reasoning. In `placebo`, state the action FIRST, then the neutral filler. (`demos` is "
        "just the bare action.)\n"
        "Always call write_dilemma."
    ),
}


def _system_for(order: str) -> str:
    if order not in _ORDER_RULES:
        raise ValueError(f"unknown WHY_ORDER {order!r}; known: {tuple(_ORDER_RULES)}")
    return _SYSTEM + _ORDER_RULES[order]


@dataclass(frozen=True)
class Generated:
    """A raw generated dilemma (the scenario plus the authored content)."""

    scenario: Scenario
    text: str
    action_text: str
    demos: str
    placebo: str
    why: str
    reversible_move: str


def _exemplars_for(structure: str, seed_examples: list[dict], k: int = 2) -> str:
    """Render up to ``k`` few-shot exemplars matching ``structure``."""
    matching = [e for e in seed_examples if e["structure"] == structure][:k]
    blocks = []
    for e in matching:
        blocks.append(
            f"TEXT: {e['text']}\nACTION: {e['action_text']}\nDEMOS: {e['demos']}\n"
            f"PLACEBO: {e['placebo']}\nWHY: {e['why']}"
        )
    return "\n\n".join(blocks)


class Generator:
    """Anthropic structured-output author of dilemmas + paired responses."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        seed_examples: list[dict],
        model: str = config.GEN_MODEL,
        client=None,
        max_workers: int = 8,
        order: str = config.WHY_ORDER,
    ):
        """Create a generator (optionally with an injected client for tests)."""
        if client is None:
            import anthropic

            config.load_env()
            client = anthropic.Anthropic(api_key=config.require_env("ANTHROPIC_API_KEY"))
        self.client = client
        self.model = model
        self.taxonomy = taxonomy
        self.seed_examples = seed_examples
        self.tool = _build_tool(taxonomy)
        self.system = _system_for(order)  # validates order
        self.max_workers = max_workers

    def _user_prompt(self, sc: Scenario) -> str:
        tax = self.taxonomy
        return (
            f"Domain: {sc.domain} -- {tax.domain_blurb(sc.domain)}.\n"
            f"Perspective: {sc.perspective}. {_PERSPECTIVE_INSTRUCTION[sc.perspective]}\n"
            f"Kind of irreversibility to feature: {sc.irreversibility}.\n"
            f"Trap baiting a rush: {sc.trap}.\n"
            f"Situation structure: {sc.structure}. {_STRUCTURE_GUIDANCE[sc.structure]}\n\n"
            f"Write a fresh, specific scenario (id {sc.index}).\n\n"
            f"Reference examples (format only; do not copy their domains):\n"
            f"{_exemplars_for(sc.structure, self.seed_examples)}"
        )

    def generate(self, scenario: Scenario, max_retries: int = 4) -> Generated:
        """Generate one dilemma for ``scenario`` with retry/backoff.

        Raises:
            RuntimeError: If all retries fail.
        """
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    # Generous: one tool call emits text+action+demos+placebo+why, and
                    # `placebo` is length-matched to a possibly-long `why`. A clip would
                    # truncate the JSON -> parse failure -> wasted retry; output is billed
                    # only for tokens actually produced, so headroom here is free.
                    max_tokens=2048,
                    temperature=config.GEN_TEMPERATURE,
                    system=[
                        {
                            "type": "text",
                            "text": self.system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=[self.tool],
                    tool_choice={"type": "tool", "name": "write_dilemma"},
                    messages=[{"role": "user", "content": self._user_prompt(scenario)}],
                )
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_use":
                        d = block.input
                        return Generated(
                            scenario=scenario,
                            text=d["text"].strip(),
                            action_text=d["action_text"].strip(),
                            demos=d["demos"].strip(),
                            placebo=d["placebo"].strip(),
                            why=d["why"].strip(),
                            reversible_move=d["reversible_move"],
                        )
                raise RuntimeError("no tool_use block in generator response")
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"generation failed after {max_retries} attempts: {last_err}")

    def _safe_generate(self, scenario: Scenario) -> Generated | None:
        try:
            return self.generate(scenario)
        except Exception as exc:  # noqa: BLE001 - one bad scenario must not sink the batch
            print(f"[generate] {scenario.uid} FAILED: {type(exc).__name__}: {exc}")
            return None

    def generate_many(self, scenarios: list[Scenario]) -> list[Generated | None]:
        """Generate many scenarios concurrently, preserving order (None on failure)."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._safe_generate, scenarios))


# --- reordering rewriter (for the WHY_ORDER ablation) ------------------------
_REWRITE_TOOL = {
    "name": "rewrite_responses",
    "description": "Record the reordered placebo and why responses.",
    "input_schema": {
        "type": "object",
        "properties": {
            "placebo": {"type": "string", "description": "The reordered placebo response."},
            "why": {"type": "string", "description": "The reordered why response."},
        },
        "required": ["placebo", "why"],
    },
}

_REWRITE_ORDER_RULES = {
    "action_first": (
        "Reorder each response so the recommended ACTION comes FIRST, followed by the rest of "
        "the original text (the filler in `placebo`, the reversibility reasoning in `why`)."
    ),
    "reason_first": (
        "Reorder each response so the recommended ACTION comes LAST, preceded by the rest of "
        "the original text (the filler in `placebo`, the reversibility reasoning in `why`)."
    ),
}

_REWRITE_SYSTEM = (
    "You reorder existing training responses WITHOUT changing their content. You receive a "
    "dilemma, its recommended action, and two responses (`placebo` and `why`).\n\n"
    "{order_rule}\n\n"
    "Hard rules:\n"
    "1. Keep the SAME recommended action, verbatim wherever grammar allows.\n"
    "2. Keep the same content: do not add, remove, or strengthen any reasoning or filler; edit "
    "only as needed for grammatical flow at the seam.\n"
    "3. `placebo` must still contain NO reasoning about reversibility, undoing, or cost.\n"
    "4. `why` must keep its reversibility reasoning intact.\n"
    "5. Keep each response within ~10% of its original length.\n"
    "Always call rewrite_responses."
)


class Rewriter:
    """Reorders the existing placebo/why renderings to the configured WHY_ORDER.

    Content-preserving by instruction: the action and its supporting text are kept,
    only their order changes. Reusing the authored dataset keeps the ablation
    paired with the main run -- identical dilemmas, demos arm, and answer key --
    so an ordering effect cannot hide behind fresh sampling noise.
    """

    def __init__(
        self,
        model: str = config.GEN_MODEL,
        client=None,
        max_workers: int = 8,
        order: str = config.WHY_ORDER,
    ):
        """Create a rewriter (optionally with an injected client for tests)."""
        if client is None:
            import anthropic

            config.load_env()
            client = anthropic.Anthropic(api_key=config.require_env("ANTHROPIC_API_KEY"))
        if order not in _REWRITE_ORDER_RULES:
            raise ValueError(f"unknown WHY_ORDER {order!r}; known: {tuple(_REWRITE_ORDER_RULES)}")
        self.client = client
        self.model = model
        self.system = _REWRITE_SYSTEM.format(order_rule=_REWRITE_ORDER_RULES[order])
        self.max_workers = max_workers

    def rewrite(self, item: tuple[str, str, str, str], max_retries: int = 4) -> dict | None:
        """Reorder one (dilemma_text, action_text, placebo, why); None on failure."""
        text, action_text, placebo, why = item
        prompt = (
            f"DILEMMA:\n{text}\n\nRECOMMENDED ACTION:\n{action_text}\n\n"
            f"PLACEBO:\n{placebo}\n\nWHY:\n{why}"
        )
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=config.ANTHROPIC_TEMPERATURE,
                    system=[
                        {
                            "type": "text",
                            "text": self.system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=[_REWRITE_TOOL],
                    tool_choice={"type": "tool", "name": "rewrite_responses"},
                    messages=[{"role": "user", "content": prompt}],
                )
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_use":
                        return {
                            "placebo": block.input["placebo"].strip(),
                            "why": block.input["why"].strip(),
                        }
                raise RuntimeError("no tool_use block in rewriter response")
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last_err = exc
                time.sleep(2**attempt)
        print(f"[rerender] rewrite FAILED: {type(last_err).__name__}: {last_err}")
        return None

    def rewrite_many(self, items: list[tuple[str, str, str, str]]) -> list[dict | None]:
        """Rewrite many items concurrently, preserving order (None on failure)."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self.rewrite, items))
