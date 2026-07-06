"""Two-tier taxonomy and a balanced scenario sampler (pure, no I/O side effects).

Tier 1 -- ``situation_structures`` -- is the only axis the correct action depends
on; it sets the 2-class label (``reversible_first`` vs ``commit_now``). Tier 2 --
``irreversibility_types``, ``traps``, ``perspectives``, ``domains`` -- are decoys:
the sampler draws each one independently of the structure, so within each label
every decoy is (in expectation) uniform and none of them predicts the answer. The
only way to score is to read the structure, which is the principle.

The train / held-out split is a *run* choice, not baked into the data:
:func:`sample_scenarios` reserves the configured ``heldout_values`` of one decoy
axis for the out-of-distribution eval and draws everything else from the rest.
Picking which axis to hold out sets the generalization distance.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from . import config

# The decoy axes that may be held out for the OOD eval (structure can't be: it
# determines the label).
HELDOUT_AXES: tuple[str, ...] = ("domain", "perspective", "irreversibility_type", "trap")


@dataclass(frozen=True)
class SituationStructure:
    """A Tier-1 structure: the latent shape that fixes the correct action."""

    name: str
    action: str  # one of config.LABELS
    gloss: str


@dataclass(frozen=True)
class Named:
    """A simple named axis value with a human gloss (perspective, domain)."""

    name: str
    gloss: str


@dataclass(frozen=True)
class Scenario:
    """One sampled scenario spec: the determinant plus the four decoys."""

    index: int
    structure: str
    action: str  # label, == action_for(structure); carried for convenience
    irreversibility: str
    trap: str
    perspective: str
    domain: str
    role: str  # "train" (feeds SFT) | "eval" (probed, never trained)
    split: str  # "train" | "heldout" -- membership on the held-out axis

    @property
    def uid(self) -> str:
        """Stable, unique id, e.g. ``false_door/cooking/0042``."""
        return f"{self.structure}/{self.domain}/{self.index:04d}"

    @property
    def distribution(self) -> str:
        """``"in_dist"`` for train-axis values, ``"ood"`` for held-out ones."""
        return "in_dist" if self.split == "train" else "ood"


@dataclass(frozen=True)
class Taxonomy:
    """The full axis vocabulary loaded from ``taxonomy.json``."""

    structures: tuple[SituationStructure, ...]
    irreversibility_types: tuple[str, ...]
    traps: tuple[str, ...]
    perspectives: tuple[Named, ...]
    domains: tuple[Named, ...]
    reversible_moves: tuple[str, ...]

    def action_for(self, structure: str) -> str:
        """Return the label fixed by ``structure``."""
        for s in self.structures:
            if s.name == structure:
                return s.action
        raise KeyError(f"unknown structure {structure!r}")

    def structures_for_label(self, label: str) -> tuple[str, ...]:
        """Names of the structures whose action is ``label``."""
        return tuple(s.name for s in self.structures if s.action == label)

    def axis_values(self, axis: str) -> tuple[str, ...]:
        """All value names for a held-out-able decoy ``axis``."""
        if axis == "domain":
            return tuple(d.name for d in self.domains)
        if axis == "perspective":
            return tuple(p.name for p in self.perspectives)
        if axis == "irreversibility_type":
            return self.irreversibility_types
        if axis == "trap":
            return self.traps
        raise ValueError(f"unknown held-out axis {axis!r}; known: {HELDOUT_AXES}")

    def perspective_gloss(self, name: str) -> str:
        """Human gloss for a perspective value (for the generator prompt)."""
        return next((p.gloss for p in self.perspectives if p.name == name), name)

    def domain_blurb(self, name: str) -> str:
        """Human blurb for a domain value (for the generator prompt)."""
        return next((d.gloss for d in self.domains if d.name == name), name)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Load and validate ``taxonomy.json``.

    Raises:
        ValueError: If a structure's action is not a configured label, if any
            label has no structure, or if any axis is empty.
    """
    raw = json.loads(Path(path or config.TAXONOMY_PATH).read_text())

    structures = tuple(
        SituationStructure(name=s["name"], action=s["action"], gloss=s["gloss"])
        for s in raw["situation_structures"]
    )
    for s in structures:
        if s.action not in config.LABELS:
            raise ValueError(f"structure {s.name!r}: action {s.action!r} not in {config.LABELS}")
    tax = Taxonomy(
        structures=structures,
        irreversibility_types=tuple(raw["irreversibility_types"]),
        traps=tuple(raw["traps"]),
        perspectives=tuple(Named(p["name"], p["gloss"]) for p in raw["perspectives"]),
        domains=tuple(Named(d["name"], d["blurb"]) for d in raw["domains"]),
        reversible_moves=tuple(raw["reversible_moves"]),
    )
    for label in config.LABELS:
        if not tax.structures_for_label(label):
            raise ValueError(f"no situation structure maps to label {label!r}")
    for axis in HELDOUT_AXES:
        if not tax.axis_values(axis):
            raise ValueError(f"axis {axis!r} is empty")
    return tax


def split_heldout_axis(
    tax: Taxonomy, axis: str, heldout_values: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """Partition ``axis`` values into (train_values, heldout_values), validating.

    Raises:
        ValueError: If the axis is not held-out-able, the held-out values are not
            all valid, or either side ends up empty.
    """
    if axis not in HELDOUT_AXES:
        raise ValueError(f"held-out axis {axis!r} must be one of {HELDOUT_AXES}")
    values = list(tax.axis_values(axis))
    unknown = [v for v in heldout_values if v not in values]
    if unknown:
        raise ValueError(f"held-out {axis} values not in taxonomy: {unknown}")
    heldout = [v for v in values if v in heldout_values]
    train = [v for v in values if v not in heldout_values]
    if not heldout or not train:
        raise ValueError(f"held-out split on {axis!r} leaves an empty side")
    return train, heldout


def sample_scenarios(
    tax: Taxonomy,
    *,
    heldout_axis: str = config.HELDOUT_AXIS,
    heldout_values: tuple[str, ...] = config.HELDOUT_VALUES,
    n_train_per_label: int = config.N_TRAIN_PER_LABEL,
    n_indist_per_label: int = config.N_INDIST_EVAL_PER_LABEL,
    n_ood_per_label: int = config.N_OOD_EVAL_PER_LABEL,
    seed: int = config.SAMPLER_SEED,
) -> list[Scenario]:
    """Sample a balanced scenario set: 50/50 labels, decoys uniform within label.

    For each label we draw ``n_train`` SFT scenarios + ``n_indist`` in-distribution
    eval scenarios (both from the training-axis values) and ``n_ood`` OOD eval
    scenarios (from the held-out values). Every decoy axis is drawn independently
    and uniformly, so the decoy distribution is identical across labels by
    construction; :func:`balance_report` lets you verify it on the realised draw.

    Returns:
        A deterministic (seeded) list of :class:`Scenario`.
    """
    train_vals, heldout_vals = split_heldout_axis(tax, heldout_axis, heldout_values)
    rng = random.Random(seed)

    base_pools: dict[str, list[str]] = {
        "domain": [d.name for d in tax.domains],
        "perspective": [p.name for p in tax.perspectives],
        "irreversibility_type": list(tax.irreversibility_types),
        "trap": list(tax.traps),
    }
    scenarios: list[Scenario] = []
    counter = 0

    def draw(label: str, split: str, role: str) -> Scenario:
        nonlocal counter
        pools = dict(base_pools)
        pools[heldout_axis] = train_vals if split == "train" else heldout_vals
        sc = Scenario(
            index=counter,
            structure=rng.choice(tax.structures_for_label(label)),
            action=label,
            irreversibility=rng.choice(pools["irreversibility_type"]),
            trap=rng.choice(pools["trap"]),
            perspective=rng.choice(pools["perspective"]),
            domain=rng.choice(pools["domain"]),
            role=role,
            split=split,
        )
        counter += 1
        return sc

    for label in config.LABELS:
        for _ in range(n_train_per_label):
            scenarios.append(draw(label, "train", "train"))
        for _ in range(n_indist_per_label):
            scenarios.append(draw(label, "train", "eval"))
        for _ in range(n_ood_per_label):
            scenarios.append(draw(label, "heldout", "eval"))
    return scenarios


def balance_report(scenarios: list[Scenario]) -> dict[str, dict[str, dict[str, int]]]:
    """Per-label marginal counts of each decoy axis (to verify no decoy leaks).

    Returns a nested dict ``{label: {axis: {value: count}}}``; a well-balanced
    draw has near-uniform counts that look the same across the two labels.
    """
    axes = ("irreversibility", "trap", "perspective", "domain")
    out: dict[str, dict[str, dict[str, int]]] = {}
    for label in config.LABELS:
        sub = [s for s in scenarios if s.action == label]
        out[label] = {}
        for axis in axes:
            counts: dict[str, int] = {}
            for s in sub:
                v = getattr(s, axis)
                counts[v] = counts.get(v, 0) + 1
            out[label][axis] = dict(sorted(counts.items()))
    return out
