"""Dataset model and loaders for dilemmas, paired responses, and the gold set.

Two kinds of data live in different places:

* **Committed inputs** in ``data/`` — ``taxonomy.json`` (see :mod:`taxonomy`),
  ``seed_examples.json`` (few-shot exemplars steering generation), and
  ``gold.json`` (hand-labeled action judgments validating the judge).
* **Generated artifacts** in ``results/`` — the synthesized :class:`Dilemma` and
  :class:`ResponsePair` records. This module owns their dataclasses and the
  dict<->record conversion used to (de)serialize them to parquet/json.

Everything is validated on load so a malformed dataset fails loudly and early
rather than halfway through a multi-hour run.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Dilemma:
    """One dilemma plus its answer-key and taxonomy fields.

    The answer key is :attr:`label` (one of ``config.LABELS``), fixed by the
    Tier-1 ``structure``. The remaining taxonomy fields (``perspective``,
    ``irreversibility_type``, ``trap``, ``domain``) are balanced decoys, kept for
    by-stratum breakdowns. ``action_text`` is the canonical recommended action.
    """

    id: str
    domain: str
    split: str  # "train" | "heldout" -- membership on the held-out axis
    role: str  # "train" (used for SFT) | "eval" (probed, never trained)
    structure: str  # Tier-1 situation structure (determines the label)
    label: str  # the correct action class (one of config.LABELS)
    perspective: str
    irreversibility_type: str
    trap: str
    reversible_move: str
    text: str
    action_text: str

    @property
    def correct_action(self) -> str:
        """The judge action a correct response must recommend (== :attr:`label`)."""
        return self.label

    @property
    def distribution(self) -> str:
        """``"in_dist"`` for training-axis eval items, ``"ood"`` for held-out ones."""
        return "in_dist" if self.split == "train" else "ood"


@dataclass(frozen=True)
class ResponsePair:
    """The training targets for one dilemma: same action, differing only by arm.

    ``demos`` is the bare action; ``placebo`` is the same action padded to ~the
    length of ``why`` with reasoning-free filler; ``why`` is the same action with
    the reversibility reasoning exposed.
    """

    dilemma_id: str
    demos: str
    placebo: str
    why: str

    def target(self, arm: str) -> str:
        """Return the completion text for training ``arm``."""
        targets = {"demos": self.demos, "placebo": self.placebo, "why": self.why}
        if arm not in targets:
            raise ValueError(f"unknown arm {arm!r}; known: {tuple(targets)}")
        return targets[arm]


@dataclass(frozen=True)
class GoldItem:
    """A hand-labeled (dilemma, response, action) triple validating the judge."""

    id: str
    dilemma: str
    response: str
    action: str
    note: str = ""


def _require(d: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{where}: missing keys {missing}")


def dilemma_from_record(rec: dict) -> Dilemma:
    """Build a :class:`Dilemma` from a flat record, validating fields."""
    _require(
        rec,
        (
            "id",
            "domain",
            "split",
            "role",
            "structure",
            "label",
            "perspective",
            "irreversibility_type",
            "trap",
            "reversible_move",
            "text",
            "action_text",
        ),
        "dilemma",
    )
    if rec["label"] not in config.LABELS:
        raise ValueError(f"{rec['id']}: bad label {rec['label']!r}")
    if rec["split"] not in ("train", "heldout"):
        raise ValueError(f"{rec['id']}: bad split {rec['split']!r}")
    if rec["role"] not in ("train", "eval"):
        raise ValueError(f"{rec['id']}: bad role {rec['role']!r}")
    return Dilemma(
        id=rec["id"],
        domain=rec["domain"],
        split=rec["split"],
        role=rec["role"],
        structure=rec["structure"],
        label=rec["label"],
        perspective=rec["perspective"],
        irreversibility_type=rec["irreversibility_type"],
        trap=rec["trap"],
        reversible_move=rec["reversible_move"],
        text=rec["text"],
        action_text=rec["action_text"],
    )


def dilemma_to_record(item: Dilemma) -> dict:
    """Flatten a :class:`Dilemma` to a JSON/parquet-friendly record."""
    return asdict(item)


def pair_from_record(rec: dict) -> ResponsePair:
    """Build a :class:`ResponsePair` from a record, validating fields."""
    _require(rec, ("dilemma_id", "demos", "placebo", "why"), "response pair")
    return ResponsePair(
        dilemma_id=rec["dilemma_id"],
        demos=rec["demos"],
        placebo=rec["placebo"],
        why=rec["why"],
    )


def pair_to_record(pair: ResponsePair) -> dict:
    """Flatten a :class:`ResponsePair` to a record."""
    return asdict(pair)


def load_seed_examples(path: Path | None = None) -> list[dict]:
    """Load ``seed_examples.json`` (few-shot exemplars for the generator).

    Each exemplar is a dict with ``structure``, ``text``, ``action_text``,
    ``demos``, ``placebo``, and ``why``. Validated lightly; these only steer
    generation.
    """
    raw = json.loads(Path(path or config.SEED_EXAMPLES_PATH).read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("seed_examples.json must be a non-empty list")
    for ex in raw:
        _require(
            ex, ("structure", "text", "action_text", "demos", "placebo", "why"), "seed example"
        )
    return raw


def load_gold(path: Path | None = None) -> list[GoldItem]:
    """Load and validate ``gold.json`` (list of labeled action judgments)."""
    raw = json.loads(Path(path or config.GOLD_PATH).read_text())
    items: list[GoldItem] = []
    for row in raw:
        _require(row, ("id", "dilemma", "response", "action"), "gold item")
        if row["action"] not in config.JUDGE_ACTIONS:
            raise ValueError(f"{row['id']}: bad action {row['action']!r}")
        items.append(
            GoldItem(
                id=row["id"],
                dilemma=row["dilemma"],
                response=row["response"],
                action=row["action"],
                note=row.get("note", ""),
            )
        )
    return items
