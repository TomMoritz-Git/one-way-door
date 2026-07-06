"""Held-out probe: generate recommendations from each adapter, on unseen domains.

For one ``(arm, seed)`` we load the base model with that arm's trained adapter
(or no adapter for the ``base`` zero-shot baseline) and greedily answer every
held-out dilemma. The prompt is the dilemma alone -- no mention of reversibility
or the principle -- so we measure transfer, not recall. Rows are returned for the
pipeline to cache; this module only touches the GPU.
"""

import pandas as pd

from . import sft
from .config import ModelSpec
from .data import Dilemma
from .model import ModelRunner


def probe_row_id(spec_key: str, arm: str, seed: int, dilemma_id: str) -> str:
    """Stable id for one held-out generation, e.g. ``qwen3.5-4b|why|0|cooking/...``."""
    return f"{spec_key}|{arm}|{seed}|{dilemma_id}"


def generate_for(
    spec: ModelSpec, dilemmas: list[Dilemma], arm: str, seed: int, device: str = "cuda"
) -> pd.DataFrame:
    """Generate held-out recommendations for one ``(spec, arm, seed)``.

    Args:
        spec: Base model.
        dilemmas: Held-out dilemmas (already filtered to the held-out split).
        arm: ``"why"``, ``"demos"``, or ``"base"`` (zero-shot, no adapter).
        seed: Adapter seed (ignored for ``base``).
        device: Torch device.

    Returns:
        A DataFrame with one row per dilemma, carrying the answer-key fields and
        the raw ``response`` for the judge.
    """
    adapter = None if arm == "base" else sft.adapter_dir(spec.key, arm, seed)
    runner = ModelRunner.load(spec, adapter_dir=adapter, device=device)
    try:
        rows = []
        for item in dilemmas:
            response, truncated = runner.generate_with_meta(item.text)
            rows.append(
                {
                    "row_id": probe_row_id(spec.key, arm, seed, item.id),
                    "model": spec.key,
                    "arm": arm,
                    "seed": seed,
                    "dilemma_id": item.id,
                    "domain": item.domain,
                    "distribution": item.distribution,
                    "structure": item.structure,
                    "perspective": item.perspective,
                    "trap": item.trap,
                    "irreversibility_type": item.irreversibility_type,
                    "reversible_move": item.reversible_move,
                    "correct_action": item.correct_action,
                    "dilemma": item.text,
                    "response": response,
                    "truncated": truncated,
                }
            )
        return pd.DataFrame(rows)
    finally:
        runner.unload()
