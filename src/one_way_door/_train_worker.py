"""Subprocess worker: train exactly one ``(model, arm, seed)`` adapter, then exit.

Run as ``python -m one_way_door._train_worker <model_key> <arm> <seed> [--force]``.

Each adapter is trained in its own process on purpose: bitsandbytes 4-bit + TRL
do not fully release VRAM across successive in-process trainings, so on an 8GB
card adapter N+1 starts starved and OOMs. Letting the OS reclaim all GPU memory
on process exit is the robust fix; :func:`one_way_door.pipeline.train_stage`
drives one of these per adapter.
"""

import sys

from . import config, pipeline, sft


def main(argv: list[str] | None = None) -> int:
    """Train one adapter named by ``argv = [model_key, arm, seed, [--force]]``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3:
        print("usage: _train_worker <model_key> <arm> <seed> [--force]")
        return 2
    model_key, arm, seed = argv[0], argv[1], int(argv[2])
    force = "--force" in argv[3:]

    spec = config.MODELS_BY_KEY[model_key]
    _, pairs = pipeline.load_dataset()
    train = pipeline.training_dilemmas()
    path = sft.train_adapter(spec, train, pairs, arm, seed, force=force)
    print(f"[train-worker] {model_key} {arm} seed{seed} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
